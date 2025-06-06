from dataclasses import dataclass, replace
from typing import Optional

@dataclass(frozen=True, slots=True)
class CitationTraceConfig:
    top_n_papers: int = 5
    max_citations_per_paper: int = 5
    max_references_per_paper: int = 5
    max_traced_papers: Optional[int] = 5      # cap after first round
    enable_second_round: bool = True
    second_round_top_n: int = 5
    second_round_cap: int = 5
    citation_only: bool = False
    year_range: Optional[str] = None

    def for_second_round(self) -> "CitationTraceConfig":
        return replace(
            self,
            top_n_papers=self.second_round_top_n,
            max_traced_papers=self.second_round_cap,
            enable_second_round=False,   # avoid infinite recursion
        ) 