# -*- coding: utf-8 -*-
"""HW2 記憶 agent（第一版：純程式的記憶框架，不需要訓練）。

記憶操作：
  寫入  chat()     存使用者自己的陳述句；略過問句與「我朋友/我同事…」這類別人的資訊
  讀取  prepare()  把題目拆成子問題，用字元 n-gram 加權重疊找最相關的句子
  更新  prepare()  同一子問題命中多句時只放最新的一句，舊值不放進記憶
  拒答  prepare()  找不到夠相關的句子時，記憶文字寫「你沒有提過」
  計算  prepare()  計算題先從記憶抓數字算好，寫成「計算結果：18歲」
工具呼叫 next_action() 目前沿用 BaseAgent（之後以 SFT 改進）。
"""
import copy
import math
import re

from agent_base import BaseAgent

ABSTAIN = "你沒有提過"

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

# 題目裡的功能詞，不拿來比對
STOP = set("我的是什麼哪裡嗎呢吧？?，,。、幾個有在現平常都會跟和怎你")
# 各種屬性共用的泛用詞，比對時不計分（避免「最喜歡」把不同主題的句子連在一起）
GENERIC = {"最喜", "喜歡", "最愛", "每天", "現在", "平常", "什麼", "今年", "一個", "家裡"}

# 題目用詞 → 陳述句常見的說法（補字面不一致的情況）
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

# 更新的提示詞
UPDATE_CUES = ("改", "換", "搬", "轉", "現在", "已經", "最近", "漲", "變成", "不玩", "以前")

NUM_RE = re.compile(r"\d+")


def strip(s):
    return "".join(c for c in s if c not in STOP)


def ngrams(s, unigram=False):
    s = "".join(c for c in s if c not in STOP)
    grams = {s[i:i + 2] for i in range(len(s) - 1)} - GENERIC
    if unigram or len(s) <= 2:
        grams |= set(s)
    return grams


class MyAgent(BaseAgent):
    threshold = 1.5      # 子問題的最高分低於此值 → 視為沒提過
    self_bonus = 0.5     # 句子含「我」（使用者自述）時加分，壓低閒聊句
    keep_ratio = 0.6     # 分數達最佳句此比例的句子也一起放進記憶
    max_per_sub = 2
    update_ratio = 0.3   # 更新句與題目的相關程度只要達最佳句的此比例即可
    budget = 300

    # ------------------------------------------------------------------ 記憶生命週期
    def reset(self):
        self.memory = []      # 每筆 {"i": 順序, "date": 日期, "text": 句子}
        self.seen = {}        # 每句話出現的次數（閒聊會重複出現，事實不會）
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
        self.memory.append({"i": len(self.memory), "date": self.date, "text": text, "key": strip(text)})
        return ""

    # ------------------------------------------------------------------ 讀取
    def _query_terms(self, q):
        terms = ngrams(q)
        for triggers, extra in EXPANSIONS:
            if any(t in q for t in triggers):
                terms |= set(extra)
        return terms

    def _score_all(self, q):
        terms = self._query_terms(q)
        # 同一句話說了兩次以上 → 閒聊，不參與檢索
        active = [self.seen.get(m["text"], 1) < 2 for m in self.memory]
        n = len(self.memory) or 1
        df = {t: sum(t in m["key"] for m in self.memory) for t in terms}
        out = []
        for m, ok in zip(self.memory, active):
            if not ok:
                out.append(0.0)
                continue
            s = sum(math.log(1 + n / df[t]) * len(t) for t in terms if df[t] and t in m["key"])
            if s > 0 and "我" in m["text"]:
                s += self.self_bonus
            out.append(s)
        return out

    def _retrieve(self, q):
        """回傳子問題最相關的句子（依時間排序；有更新時只留更新後的），找不到回傳 []。"""
        if not self.memory:
            return []
        scores = self._score_all(q)
        best = max(scores)
        if best < self.threshold:
            return []
        cand = sorted(range(len(scores)), key=lambda i: (-scores[i], -i))
        cand = [i for i in cand if scores[i] >= self.keep_ratio * best][:self.max_per_sub]
        cand.sort()
        # 更新：最佳句之後若有與題目相關（分數達 update_ratio）且帶更新提示詞的句子，
        # 只放最新的那句，舊值不放進記憶
        top = cand[0] if len(cand) == 1 else max(cand, key=lambda i: scores[i])
        for i in range(len(scores) - 1, top, -1):
            if scores[i] >= self.update_ratio * best and any(c in self.memory[i]["text"] for c in UPDATE_CUES):
                return [self.memory[i]["text"]]
        if len(cand) > 1 and any(c in self.memory[cand[-1]]["text"] for c in UPDATE_CUES):
            cand = cand[-1:]
        return [self.memory[i]["text"] for i in cand]

    # ------------------------------------------------------------------ 計算
    def _number(self, words):
        """找最新一句含任一關鍵詞且有數字的句子，回傳其中的數字。"""
        for m in reversed(self.memory):
            if any(w in m["text"] for w in words):
                nums = NUM_RE.findall(m["text"])
                if nums:
                    return int(nums[0])
        return None

    def _calc(self, q):
        """辨識計算題並算出結果；不是計算題回傳 None，缺資料回傳 ABSTAIN。"""
        d = lambda: self._number(("女兒",))
        s = lambda: self._number(("兒子",))
        if "包含我" in q or ("總共" in q and "小孩" in q and "兩個" not in q):
            v = self._number(("兄弟姊妹", "手足"))
            return ABSTAIN if v is None else f"計算結果：{v + 1}個"
        if "房租" in q and ("一年" in q or "整年" in q):
            v = self._number(("房租", "租"))
            return ABSTAIN if v is None else f"計算結果：{v * 12}元"
        if "明年" in q:
            v = s() if "兒子" in q else d() if "女兒" in q else None
            return ABSTAIN if v is None else f"計算結果：{v + 1}歲"
        m = re.search(r"滿(\d+)歲", q)
        if m:
            v = d() if "女兒" in q else s() if "兒子" in q else None
            return ABSTAIN if v is None else f"計算結果：{int(m.group(1)) - v}年"
        if ("差" in q or "加起來" in q or "總共" in q) and ("小孩" in q or ("兒子" in q and "女兒" in q)):
            a, b = d(), s()
            if a is None or b is None:
                return ABSTAIN
            v = abs(a - b) if "差" in q else a + b
            return f"計算結果：{v}歲"
        return None

    # ------------------------------------------------------------------ 組合記憶文字
    def prepare(self, question: str) -> str:
        calc = self._calc(question)
        if calc is not None:
            return calc
        subs = [s for s in re.split(r"[、，,]", question) if s.strip()]
        found = []
        for sq in subs:
            for t in self._retrieve(sq):
                if t not in found:
                    found.append(t)
        if not found:
            return ABSTAIN
        return "\n".join(found)[-self.budget:]

    # ------------------------------------------------------------------ snapshot
    def save_snapshot(self):
        return copy.deepcopy((self.memory, self.seen))

    def load_snapshot(self, snap):
        self.memory, self.seen = copy.deepcopy(snap)
