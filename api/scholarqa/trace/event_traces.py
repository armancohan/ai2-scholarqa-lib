from datetime import datetime
from typing import Any, Dict, List

from scholarqa.config.config_setup import LogsConfig
from scholarqa.llms.constants import CostAwareLLMResult
from scholarqa.models import ToolRequest
from scholarqa.trace.trace_writer import LocalWriter


class EventTrace:
    def __init__(self, task_id: str, n_retrieval: int, n_rerank: int, req: ToolRequest, user_id: str = None):
        self.query = req.query
        if not user_id:
            try:
                self.user_id = req.user_id
            except Exception as e:
                self.user_id = None
        else:
            self.user_id = user_id
        self.task_id = task_id
        self.timestamp = datetime.now().isoformat()
        self.n_retrieval = n_retrieval
        self.n_retrieved = 0
        self.n_candidates = 0
        self.n_rerank = n_rerank
        self.opt_in = req.opt_in
        self.total_cost = 0.0

        self.decomposed_query = dict()
        self.candidates = []
        self.retrieved = []
        self.quotes = {"cost": 0.0, "quotes": []}
        self.cluster = dict()
        self.summary = dict()
        self.total_cost = 0.0

    def trace_decomposition_event(self, decomposed_query: CostAwareLLMResult):
        self.decomposed_query = decomposed_query.result._asdict()
        self.decomposed_query["cost"] = decomposed_query.tot_cost
        self.decomposed_query["model"] = decomposed_query.models[0]
        self.total_cost += decomposed_query.tot_cost

    def trace_retrieval_event(self, retrieved: List[Dict[str, Any]]):
        self.n_retrieved = len(retrieved)
        self.retrieved = retrieved

    def trace_rerank_event(self, candidates: List[Dict[str, Any]]):
        self.n_candidates = len(candidates)
        self.candidates = candidates

    def trace_quote_event(self, paper_summaries: CostAwareLLMResult):
        topk = [{"idx": i, "key": k, "snippets": v} for i, (k, v) in enumerate(paper_summaries.result.items())]
        topk_models = paper_summaries.models
        for idx, tk in enumerate(topk):
            tk["model"] = topk_models[idx]

        self.quotes["cost"] = paper_summaries.tot_cost
        self.quotes["quotes"] = topk
        self.total_cost += paper_summaries.tot_cost

    def trace_clustering_event(self, cluster_json: CostAwareLLMResult, plan_str: Dict[str, Any]):
        self.cluster["cost"] = cluster_json.tot_cost
        self.cluster["cot"] = cluster_json.result["cot"]
        self.cluster["plan"] = plan_str
        self.cluster["model"] = cluster_json.models[0]
        self.total_cost += cluster_json.tot_cost

    def trace_inline_citation_following_event(
        self, paper_summaries_extd: Dict[str, Any], quotes_metadata: Dict[str, List[Dict[str, Any]]]
    ):
        for quote_obj in self.quotes["quotes"]:
            quote_obj["snippets"] = paper_summaries_extd[quote_obj["key"]].get("quote", quote_obj["snippets"])
            quote_obj["inline_citations"] = paper_summaries_extd[quote_obj["key"]].get("inline_citations", dict())
            quote_obj["metadata"] = quotes_metadata.get(quote_obj["key"], [])

    def trace_summary_event(self, json_summary: List[Dict[str, Any]], cost_result: CostAwareLLMResult):
        self.summary = {"sections": json_summary, "cost": cost_result.tot_cost}
        for idx, section in enumerate(self.summary["sections"]):
            section["model"] = cost_result.models[idx]
        self.total_cost += cost_result.tot_cost

    def persist_trace(self, logs_config: LogsConfig):
        trace_writer = LocalWriter(local_dir=f"{logs_config.log_dir}/{logs_config.event_trace_loc}")
        trace_writer.write(trace_json=self, file_name=self.task_id)
