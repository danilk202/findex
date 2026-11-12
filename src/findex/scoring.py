import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int32]


class CorpusStats(Protocol):
    @property
    def num_docs(self) -> int: ...

    @property
    def avg_doc_length(self) -> float: ...


class Scorer(Protocol):
    def score(self, tf: int, df: int, doc_len: int, stats: CorpusStats) -> float: ...

    # Лаба 8: та сама формула над масивами — tf і doc_len усіх постінгів терма разом.
    def score_array(
        self, tf: IntArray, df: int, doc_len: IntArray, stats: CorpusStats
    ) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    def idf(self, df: int, stats: CorpusStats) -> float:
        return math.log(stats.num_docs / df)

    def score(self, tf: int, df: int, doc_len: int, stats: CorpusStats) -> float:
        return tf * self.idf(df, stats)

    def score_array(
        self, tf: IntArray, df: int, doc_len: IntArray, stats: CorpusStats
    ) -> FloatArray:
        return (tf * self.idf(df, stats)).astype(np.float64, copy=False)


@dataclass(frozen=True, slots=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def idf(self, df: int, stats: CorpusStats) -> float:
        return math.log(1 + (stats.num_docs - df + 0.5) / (df + 0.5))

    def score(self, tf: int, df: int, doc_len: int, stats: CorpusStats) -> float:
        idf = self.idf(df, stats)
        norm = 1 - self.b + self.b * doc_len / stats.avg_doc_length
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)

    def score_array(
        self, tf: IntArray, df: int, doc_len: IntArray, stats: CorpusStats
    ) -> FloatArray:
        # Вираз символ у символ як у score(): ті самі операції в тому самому порядку,
        # тож і округлення float64 ті самі — бали збігаються з лабою 3 до біта.
        idf = self.idf(df, stats)
        norm = 1 - self.b + self.b * doc_len / stats.avg_doc_length
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)


type ScorerName = Literal["bm25", "tfidf"]
SCORER_NAMES: tuple[ScorerName, ...] = ("bm25", "tfidf")


def make_scorer(name: ScorerName) -> Scorer:
    match name:
        case "bm25":
            return BM25()
        case "tfidf":
            return TfIdf()
