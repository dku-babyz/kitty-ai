# inference/dictionary/checker.py
from __future__ import annotations
from pathlib import Path
import pandas as pd, ahocorasick, re, unicodedata

__all__ = ["DictionaryChecker"]

_BRACKETS = re.compile(r"[()\[\]{}<>]")   # (개삽년이) → 개삽년이


class DictionaryChecker:
    """
    >>> chk = DictionaryChecker("inference/dictionary/사전.csv")
    >>> chk("야 (개삽년이) 뭐하냐!")
    {
        "개삽년": {
            "언어표현": "개삽년",
            "분류": "유해",
            "유형": "Harsh",
            "품사": "명사",
            "의미": "...",
            "예문": "...",
            "비고": ""
        }
    }
    """

    # 기본 헤더 후보를 넉넉히 넣어 두었음
    _WORD_COL_CANDS = ["언어표현", "단어", "word", "표제어"]
    _TYPE_COL_CANDS = ["유형", "분류", "category", "type"]

    def __init__(
        self,
        dict_path: str | Path,
        *,
        word_col: str | None = None,
        type_col: str | None = None,
        normalize: bool = True,
    ):
        df = self._load_dict(dict_path)

        # ► 컬럼명 자동 추론 (또는 사용자가 지정)
        self.word_col = word_col or self._guess(df, self._WORD_COL_CANDS)
        self.type_col = type_col or self._guess(df, self._TYPE_COL_CANDS)

        # ► 메타·Trie 빌드
        self.meta = {row[self.word_col]: row.to_dict() for _, row in df.iterrows()}
        self.automaton = self._build_automaton(self.meta.keys())
        self.normalize = normalize

    # ───────────────────────────────────────── helpers
    @staticmethod
    def _guess(df: pd.DataFrame, cands: list[str]) -> str:
        for c in cands:
            if c in df.columns:
                return c
        raise ValueError(
            f"❌ 사전 파일에 {cands} 중 하나의 컬럼이 필요합니다.\n현재 컬럼: {list(df.columns)}"
        )

    @staticmethod
    def _load_dict(path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path, keep_default_na=False)
        return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)

    @staticmethod
    def _build_automaton(words):
        A = ahocorasick.Automaton()
        for w in words:
            if w:                       # 빈 문자열 방어
                A.add_word(w, w)
        A.make_automaton()
        return A

    @staticmethod
    def _clean(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        return _BRACKETS.sub("", text).lower().strip()

    # ───────────────────────────────────────── public API
    def __call__(self, text: str) -> dict:
        txt = self._clean(text) if self.normalize else text
        hits = {w for _, w in self.automaton.iter(txt)}
        return {w: self.meta[w] for w in hits}
