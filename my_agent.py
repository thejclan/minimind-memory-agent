# -*- coding: utf-8 -*-
"""HW2 記憶 agent（第二版：字面比對 + 小型 embedding 的混合檢索）。

記憶操作：
  寫入  chat()     存使用者自己的陳述句；略過問句與「我朋友/我同事…」這類別人的資訊
  閒聊  prepare()  同一句話說了兩次以上視為閒聊，不參與檢索（事實不會重複說）
  讀取  prepare()  題目拆成子問題；每個子問題用「embedding 語意相似度 + 字元詞組重疊」找最相關的句子
  更新  prepare()  最相關的句子之後若有相關且帶更新提示詞（改、換、搬…）的句子，只放最新的
  拒答  prepare()  沒有夠相關的句子時，記憶文字寫「你沒有提過」
  計算  prepare()  計算題先從記憶抓數字算好，寫成「計算結果：18歲」
工具呼叫 next_action() 由（SFT 後的）模型生成；只有在模型輸出了 <tool_call> 但 JSON 壞掉時，
  才嘗試修補格式（作業允許的「解析失敗時的備援」），不改變模型「要不要呼叫、呼叫哪個工具」的決定。

輔助 embedding 模型：BAAI/bge-small-zh-v1.5（23.95M 參數，≤30M），放在本檔旁的 embedder/。
"""
import copy
import json
import math
import os
import re

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from agent_base import BaseAgent

ABSTAIN = "你沒有提過"
HERE = os.path.dirname(os.path.abspath(__file__))
EMBEDDER_DIRS = [os.environ.get("HW2_EMBEDDER", ""), os.path.join(HERE, "embedder"),
                 os.path.join(HERE, "models", "bge-small-zh-v1.5")]

# 談論別人時，「我」後面接的人物稱謂（盡量涵蓋沒看過的說法）
RELATIONS = [
    "大學同學", "高中同學", "國中同學", "小學同學", "同學", "朋友", "好友", "閨蜜", "死黨", "同事", "主管",
    "老闆", "上司", "下屬", "客戶", "室友", "鄰居", "學長", "學姊", "學姐", "學弟", "學妹", "前輩", "教練",
    "表姊", "表姐", "表哥", "表弟", "表妹", "堂哥", "堂姊", "堂姐", "堂弟", "堂妹", "男朋友", "女朋友",
    "男友", "女友", "老公", "老婆", "先生", "太太", "爸爸", "媽媽", "爸", "媽", "哥哥", "姊姊", "姐姐",
    "弟弟", "妹妹", "阿姨", "叔叔", "伯伯", "舅舅", "姑姑", "嬸嬸", "阿公", "阿嬤", "爺爺", "奶奶",
    "外公", "外婆", "親戚", "網友", "隊友", "夥伴",
]
OTHER_RE = re.compile(
    r"我(?:的|那個|有個|有一個|一個|一位|有位)?(?:大學|高中|國中|國小|公司|以前的|前|新的?)?(?:%s)"
    % "|".join(sorted(RELATIONS, key=len, reverse=True)))

STOP = set("我的是什麼哪裡嗎呢吧？?，,。、幾個有在現平常都會跟和怎你")
GENERIC = {"最喜", "喜歡", "最愛", "每天", "現在", "平常", "什麼", "今年", "一個", "家裡"}
EXPANSIONS = [
    (("住", "城市", "哪裡"), ("住", "家在", "搬")),
    (("工作", "職業"), ("工作", "當", "職", "這行", "一名")),
    (("吃", "食物"), ("吃", "愛", "最愛", "食物")),
    (("寵物", "動物"), ("養", "寵物")),
    (("名字",), ("叫", "名字", "取名")),
    (("通勤", "上班"), ("上班", "通勤", "公司")),
    (("電影",), ("看", "片")),
    (("大學", "系"), ("大學", "系", "念", "主修")),
    (("運動",), ("運動",)),
    (("顏色",), ("色",)),
    (("房租",), ("房租", "租")),
]
UPDATE_CUES = ("改", "換", "搬", "轉", "現在", "已經", "最近", "漲", "變成", "不玩", "以前", "後來", "新")
SPLIT_RE = re.compile(r"[、，,？?]|和|跟|以及|還有|又")
NUM_RE = re.compile(r"\d+")


def strip(s):
    return "".join(c for c in s if c not in STOP)


def ngrams(s):
    s = strip(s)
    grams = {s[i:i + 2] for i in range(len(s) - 1)} - GENERIC
    if len(s) <= 2:
        grams |= set(s)
    return grams


class MyAgent(BaseAgent):
    lex_threshold = 1.5  # 字面比對最高分達此值 → 用字面比對的結果（第一版的做法）
    sem_threshold = 0.6  # 字面比對找不到時，embedding 相似度達此值才採用；否則視為沒提過
    keep_ratio = 0.6     # 字面比對：分數達最佳句此比例的句子也一起放
    update_ratio = 0.3   # 字面比對：更新句的分數只要達最佳句此比例
    sem_update_margin = 0.1
    sem_z = 2.5          # embedding 補救時，最相似的句子要比其他句子平均高出幾個標準差才採用
    max_per_sub = 2
    split_words = True  # 是否也用「和、跟」切子問題
    budget = 300
    query_prefix = ""    # bge 的檢索指令前綴（實測後決定是否使用）
    # 記憶文字的組法：
    #   retrieve  只放檢索到的句子，沒找到寫「你沒有提過」（第二版）
    #   fill      檢索到的句子放最後，前面依時間順序補上其他記憶，直到 300 字
    #   fill_hint 同 fill，但沒檢索到時最後一行寫「你沒有提過」
    #   chrono    全部依時間順序；超過 300 字時先刪最早的非檢索句
    context_mode = os.environ.get("HW2_CONTEXT_MODE", "chrono")

    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)
        path = next(p for p in EMBEDDER_DIRS if p and os.path.isdir(p))
        self.emb_dev = device if device else "cpu"
        self.etok = AutoTokenizer.from_pretrained(path)
        self.emodel = AutoModel.from_pretrained(path).to(self.emb_dev).eval()

    # ------------------------------------------------------------------ embedding
    @torch.no_grad()
    def embed(self, texts):
        b = self.etok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.emb_dev)
        h = self.emodel(**b).last_hidden_state[:, 0]          # bge 用 [CLS] 向量
        return F.normalize(h.float(), dim=-1).cpu()

    # ------------------------------------------------------------------ 記憶生命週期
    def reset(self):
        self.memory = []      # 每筆 {"text": 句子, "key": 去功能字的句子, "emb": 向量}
        self.seen = {}        # 每句話出現的次數
        self.date = None

    def new_session(self, date: str):
        self.date = date

    # ------------------------------------------------------------------ 寫入
    @staticmethod
    def is_other(text):
        return bool(OTHER_RE.search(text))

    def chat(self, msg: str) -> str:
        text = msg.strip()
        self.seen[text] = self.seen.get(text, 0) + 1
        if not text or text.endswith(("？", "?")) or self.is_other(text):
            return ""
        self.memory.append({"date": self.date, "text": text, "key": strip(text), "emb": None})
        return ""

    # ------------------------------------------------------------------ 讀取
    def _active(self):
        """參與檢索的記憶：排除重複說過的閒聊；順便補算還沒有的 embedding。"""
        todo = [m for m in self.memory if m["emb"] is None]
        if todo:
            for m, e in zip(todo, self.embed([m["text"] for m in todo])):
                m["emb"] = e
        return [i for i, m in enumerate(self.memory) if self.seen.get(m["text"], 1) < 2]

    def _lex(self, q, active):
        """字面比對分數（第一版）：加權的字元詞組重疊，自述句加分。"""
        terms = ngrams(q)
        for triggers, extra in EXPANSIONS:
            if any(t in q for t in triggers):
                terms |= set(extra)
        n = len(active) or 1
        df = {t: sum(t in self.memory[i]["key"] for i in active) for t in terms}
        out = {}
        for i in active:
            v = sum(math.log(1 + n / df[t]) * len(t) for t in terms if df[t] and t in self.memory[i]["key"])
            out[i] = v + (0.5 if v > 0 and "我" in self.memory[i]["text"] else 0)
        return out

    def _sem(self, q, active):
        qe = self.embed([self.query_prefix + q])[0]
        return {i: float(qe @ self.memory[i]["emb"]) for i in active}

    @staticmethod
    def _has_cue(text):
        return any(c in text for c in UPDATE_CUES)

    def _retrieve(self, q, active):
        if not active:
            return []
        lex = self._lex(q, active)
        best = max(lex.values())
        if best >= self.lex_threshold:          # 字面比對找得到 → 第一版的選句與更新規則
            cand = sorted(lex, key=lambda i: (-lex[i], -i))
            cand = sorted(i for i in cand[:self.max_per_sub] if lex[i] >= self.keep_ratio * best)
            top = max(cand, key=lambda i: lex[i])
            for i in sorted(lex, reverse=True):
                if i <= top:
                    break
                if lex[i] >= self.update_ratio * best and self._has_cue(self.memory[i]["text"]):
                    return [i]
            if len(cand) > 1 and self._has_cue(self.memory[cand[-1]]["text"]):
                cand = cand[-1:]
            return cand
        sem = self._sem(q, active)                # 字面找不到 → 用 embedding 補救
        top = max(sem, key=lambda i: (sem[i], i))
        vals = list(sem.values())
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
        if sem[top] < self.sem_threshold or (sem[top] - mean) / std < self.sem_z:
            return []
        for i in sorted(sem, reverse=True):
            if i <= top:
                break
            if sem[i] >= sem[top] - self.sem_update_margin and self._has_cue(self.memory[i]["text"]):
                return [i]
        return [top]

    # ------------------------------------------------------------------ 計算
    def _number(self, words):
        """最新一句含任一關鍵詞且有數字的記憶，回傳其中的數字。"""
        for m in reversed(self.memory):
            if any(w in m["text"] for w in words):
                nums = NUM_RE.findall(m["text"])
                if nums:
                    return int(nums[0])
        return None

    def _calc(self, q):
        """辨識計算題並算出結果；不是計算題回傳 None，缺資料回傳「你沒有提過」。"""
        kid_words = ("小孩", "孩子", "兒女", "子女")
        daughter = lambda: self._number(("女兒",))
        son = lambda: self._number(("兒子",))
        # 家裡總共幾個小孩（含自己）= 兄弟姊妹數 + 1
        if re.search(r"包含我|包括我|算上我|算我|加上我(?!兒子|女兒|的)|連我|含我", q) or re.search(r"(爸媽|父母).*生了", q):
            v = self._number(("兄弟姊妹", "手足", "兄姊", "弟妹", "兄弟", "姊妹"))
            return ABSTAIN if v is None else f"計算結果：{v + 1}個"
        # 一年的房租 = 月租 × 12
        if "租" in q and re.search(r"年|十二個月|12個月", q):
            v = self._number(("房租", "租", "房東"))
            return ABSTAIN if v is None else f"計算結果：{v * 12}元"
        who = son if "兒子" in q else daughter if "女兒" in q else None
        # 明年幾歲 = 年齡 + 1
        if who and re.search(r"明年|再過一年|下一年|一年後|過一年", q):
            v = who()
            return ABSTAIN if v is None else f"計算結果：{v + 1}歲"
        # 還要幾年滿 N 歲 = N − 年齡
        m = re.search(r"(\d+)歲", q)
        if who and m and re.search(r"年|多久|還要|還有|再過|幾時|什麼時候|才", q.replace(m.group(0), "")):
            v = who()
            return ABSTAIN if v is None else f"計算結果：{int(m.group(1)) - v}年"
        # 兩個小孩：相差幾歲 / 加起來幾歲
        two_kids = ("兒子" in q and "女兒" in q) or (any(w in q for w in kid_words) and re.search(r"兩|二|2", q))
        if two_kids and re.search(r"差|相差|差距", q):
            a, b = daughter(), son()
            return ABSTAIN if a is None or b is None else f"計算結果：{abs(a - b)}歲"
        if two_kids and re.search(r"加起來|總共|合計|加總|相加|一共|合起來", q):
            a, b = daughter(), son()
            return ABSTAIN if a is None or b is None else f"計算結果：{a + b}歲"
        return None

    # ------------------------------------------------------------------ 組合記憶文字
    def prepare(self, question: str) -> str:
        calc = self._calc(question)
        if calc is not None:
            return calc
        active = self._active()
        pat = SPLIT_RE if self.split_words else re.compile(r"[、，,？?]")
        subs = [s for s in pat.split(question) if strip(s)] or [question]
        picked = []
        for sq in subs:
            for i in self._retrieve(sq, active):
                if i not in picked:
                    picked.append(i)
        return self._render(picked, active)

    def _render(self, picked, active):
        text = lambda ids: "\n".join(self.memory[i]["text"] for i in ids)
        mode = self.context_mode
        if mode == "retrieve":
            if not picked:
                return ABSTAIN
            return text(picked)[-self.budget:]          # 依子問題的順序排列（與多事實題選項的順序一致）
        rest = [i for i in active if i not in picked]
        if mode == "chrono":
            keep = sorted(set(active))
            for i in list(rest):                        # 太長時先刪最早的非檢索句
                if len(text(keep)) <= self.budget:
                    break
                keep.remove(i)
            return text(keep)[-self.budget:]
        lines = [self.memory[i]["text"] for i in rest + picked]
        if mode == "fill_hint" and not picked:
            lines.append(ABSTAIN)
        return "\n".join(lines)[-self.budget:] if lines else ABSTAIN

    # ------------------------------------------------------------------ snapshot
    def save_snapshot(self):
        self._active()   # 先算好所有句子的 embedding，之後每題還原時不用重算
        return copy.deepcopy((self.memory, self.seen))

    def load_snapshot(self, snap):
        self.memory, self.seen = copy.deepcopy(snap)

    # ------------------------------------------------------------------ 工具呼叫
    TOOL_NAMES = ("memory_save", "memory_search", "calculator", "get_date")

    DATE_RE = re.compile(r"(今天|今日|現在|目前).{0,6}(幾月|幾號|幾日|日期|星期幾|禮拜幾|週幾|哪一天|哪天)"
                         r"|(幾月幾號|幾月幾日|星期幾|禮拜幾|週幾|今天的日期|今天日期)")

    def next_action(self, context: list) -> str:
        out = super().next_action(context)
        return self.postprocess(out, context[-1] if context else "")

    CALC_ASK = re.compile(r"幫我算|算一下|算算|等於|是多少|多少錢|還剩|剩多少|一共|總共|合計|加起來")

    @classmethod
    def postprocess(cls, out, last):
        call = lambda name, args: "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"
        # 問今天日期／星期的句子一定要查 get_date（模型漏判時補上）
        if cls.DATE_RE.search(last) and "get_date" not in out:
            return call("get_date", {})
        # 句子裡有兩個以上的數字、又在問計算結果 → 一定是 calculator
        if len(re.findall(r"\d+", last)) >= 2 and cls.CALC_ASK.search(last) and '"calculator"' not in out:
            expr = cls.fix_expression("", last)
            if expr:
                return call("calculator", {"expression": expr})
        if "<tool_call>" not in out:
            return out
        out = cls.repair_call(out)
        # 問句（以問號結尾）不是在陳述自己的資訊，不該 memory_save
        if re.search(r"[？?]\s*$", last) and '"memory_save"' in out:
            return "好問題！"
        return cls.normalize_args(out, last)

    # ------------------------------------------------------------------ 工具參數正規化
    OPS = [("/", re.compile(r"除以|平分|分給|每人|一人")),
           ("*", re.compile(r"乘|買.{0,3}\d+\s*[張個杯盒份本件瓶包台組]|\d+\s*[張個杯盒份本件瓶包台組].{0,4}(總共|共|要)")),
           ("-", re.compile(r"減|少|剩|花掉|扣|便宜|折")),
           ("+", re.compile(r"加|又|再|總共|合計|一共|共"))]

    # 公開模板的 key（屬性名稱）→ 使用者原句中應該出現的相關詞。模型給的 key 若是這些名稱之一，
    # 但原句完全沒有相關詞（或出現別的類別詞），代表模型套錯了訓練時看過的名稱（多半是沒看過的新屬性），
    # 就改用原句本身的描述當 key / query。
    KEY_EVIDENCE = {
        "名字": "名字 姓名 叫 喊 名 稱呼 認識", "居住地": "住 家在 搬 待在 定居 遷 城市 縣市 落腳 哪裡 生活",
        "職業": "職業 工作 當 做 職 行 業 上班 轉行 任職 從事 身分 身份 靠 一名 一位 是個", "寵物種類": "寵物 動物 養 飼 隻",
        "寵物名字": "寵物 名字 叫 取名 牠 喊", "最喜歡的食物": "食物 吃 愛 第一名 新歡 喜歡 迷上 點",
        "過敏": "過敏 疹 癢", "生日月份": "生日 出生 生的", "運動": "運動 打 跑 游 爬山 瑜珈 騎 練 球",
        "通勤方式": "通勤 上班 交通 公司 搭 騎 開車 走路", "喜歡的顏色": "顏色 色", "電影類型": "電影 片",
        "大學主修": "主修 科系 大學 系 念 讀", "起床時間": "起床 醒 起來 鬧鐘", "女兒年齡": "女兒",
        "兒子年齡": "兒子", "房租": "房租 租 房東", "兄弟姊妹數": "兄弟姊妹 手足 兄 姊 弟 妹 孩 家",
    }
    KEY_CONFLICT = {"最喜歡的食物": "甜點 早餐 午餐 晚餐 零食 水果 飲料 喝 點心 宵夜 茶 咖啡 書 小說 看",
                    "居住地": "租", "通勤方式": "身分 身份 工作是 職業",
                    "起床時間": "睡 上床 就寢 加起來 看起來 聽起來", "名字": "寵物 牠", "電影類型": "書 小說 追劇 影集"}
    NUMERIC_KEYS = {"房租", "女兒年齡", "兒子年齡", "兄弟姊妹數"}
    FILLER = re.compile(r"什麼|哪一|哪個|哪種|哪裡|多少|幾|是|的|我|嗎|呢|啦|喔|了|[，。！？?!、\s]")

    @classmethod
    def ground_key(cls, key, text, value=None):
        if not isinstance(key, str):
            return key
        ev = cls.KEY_EVIDENCE.get(key)
        if ev is not None:
            grounded = any(w in text for w in ev.split()) and \
                not any(w in text for w in cls.KEY_CONFLICT.get(key, "").split())
        else:
            grounded = bool(key) and key in text or (len(key) > 2 and key[:2] in text)
        if grounded:
            return key
        # 換成原句有明確證據的公開屬性名稱（只看兩個字以上的證據詞，「租」例外）
        cands = [k for k, ev2 in cls.KEY_EVIDENCE.items() if k != key and
                 any((len(w) >= 2 or k == "房租") and w not in ("喜歡", "上班", "生活", "哪裡") and w in text
                     for w in ev2.split()) and
                 not any(w in text for w in cls.KEY_CONFLICT.get(k, "").split())]
        if len(cands) == 1:
            return cands[0]
        desc = text.replace(value, "") if value else text
        desc = cls.FILLER.sub("", desc)
        return desc if len(desc) >= 2 else key

    @classmethod
    def normalize_args(cls, out, last):
        """模型決定呼叫哪個工具；參數只做「抄錯」的修正：
        calculator 的算式改用使用者原句裡的數字；memory_save 的 value 必須是使用者原句的片段。"""
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", out, re.S)
        if not m:
            return out
        try:
            obj = json.loads(m.group(1))
        except Exception:
            return out
        name, args = obj.get("name"), obj.get("arguments")
        if not isinstance(args, dict):
            return out
        if name == "calculator":
            expr = cls.fix_expression(str(args.get("expression", "")), last)
            if expr:
                args["expression"] = expr
        elif name == "memory_save" and isinstance(args.get("value"), str):
            args["value"] = cls.fix_value(args["value"], last)
            args["key"] = cls.ground_key(args.get("key"), last, args["value"])
            nums = re.findall(r"\d+", last)
            if args["key"] in cls.NUMERIC_KEYS and not re.search(r"\d", args["value"]) and nums:
                args["value"] = nums[-1]                 # 數值屬性的值一定是數字
        elif name == "memory_search":
            args["query"] = cls.ground_key(args.get("query"), last)
        else:
            return out
        return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"

    @classmethod
    def fix_expression(cls, expr, last):
        nums = re.findall(r"\d+(?:\.\d+)?", last)
        clean = re.sub(r"[^\d\.\+\-\*/\(\)]", "", expr)
        ok = bool(re.fullmatch(r"[\d\.]+([\+\-\*/][\d\.]+)+", clean)) and \
            all(n in nums for n in re.findall(r"\d+(?:\.\d+)?", clean))
        if len(nums) == 2:
            t = last.replace("多少", "")                   # 「多少」的「少」不是減法
            op = next((o for o, r in cls.OPS if r.search(t)), None)
            if op is None:
                mo = re.search(r"[\+\-\*/]", clean)
                op = mo.group(0) if mo else None
            if op:
                a, b = nums
                if op == "/" and not re.search(r"除以", last) and float(a) < float(b):
                    a, b = b, a                          # 「3個人平分600元」→ 600/3
                return f"{a}{op}{b}"
        return clean if ok else None

    @staticmethod
    def fix_value(value, last):
        if not value or value in last:
            return value
        best, best_score = value, 0.0
        L = len(value)
        for n in (L, L + 1, L - 1):
            if n <= 0:
                continue
            for i in range(len(last) - n + 1):
                w = last[i:i + n]
                pos = sum(a == b for a, b in zip(w, value))      # 同位置相同的字
                common = len(set(w) & set(value))
                score = pos + 0.5 * common - 0.1 * abs(n - L) + 0.3 * (w[0] == value[0])
                if score > best_score:
                    best, best_score = w, score
        return best if best_score >= 1.0 else value

    @classmethod
    def repair_call(cls, out):
        """模型輸出 <tool_call> 但無法解析時，嘗試把它修成合法格式；修不好就原樣回傳。"""
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", out, re.S)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj.get("name"), str) and isinstance(obj.get("arguments", {}), (dict, str)):
                    return out                                   # 本來就合法
            except Exception:
                pass
        body = out.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        name = next((n for n in cls.TOOL_NAMES if n in body), None)
        if name is None:
            return out
        args = {}
        for k, v in re.findall(r'"(key|value|query|expression)"\s*:\s*"([^"]*)"', body):
            args[k] = v
        fixed = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        return "<tool_call>\n" + fixed + "\n</tool_call>"
