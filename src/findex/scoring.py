import math
from dataclasses import dataclass
from typing import Protocol


class CorpusStats(Protocol):

    @property
    def num_docs(self) -> int: ...

    @property
    def avg_doc_length(self) -> float: ...


class Scorer(Protocol):

    def score(self, tf: int, df: int, doc_len: int, stats: CorpusStats) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    def score(self, tf: int, df: int, doc_len: int, stats: CorpusStats) -> float:
   
        idf = math.log(stats.num_docs /df)

        return tf * idf


@dataclass(frozen=True, slots=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def score(self, tf: int, df: int, doc_len: int, stats: CorpusStats) -> float:
       
        idf = math.log(1 + (stats.num_docs - df + 0.5) / (df + 0.5))
        norm = 1 - self.b + self.b * doc_len / stats.avg_doc_length
        score = idf * tf  * (self.k1 + 1) / (tf + self.k1 * norm)

        return score 
