import logging
import os
import sys
from collections import namedtuple
from logging import Formatter
import pandas as pd
from typing import Any, Dict, List, Optional, Set, Union

import requests
from scholarqa.llms.litellm_helper import setup_llm_cache

logger = logging.getLogger(__name__)

S2_APIKEY = os.getenv("S2_API_KEY", "")
S2_HEADERS = {"x-api-key": S2_APIKEY}
S2_API_BASE_URL = "https://api.semanticscholar.org/graph/v1/"
CompletionResult = namedtuple("CompletionCost", ["content", "model", "cost", "input_tokens", "output_tokens", "total_tokens"])
NUMERIC_META_FIELDS = {"year", "citationCount", "referenceCount", "influentialCitationCount"}
CATEGORICAL_META_FIELDS = {"title", "abstract", "corpusId", "authors", "venue", "isOpenAccess", "openAccessPdf", "fieldsOfStudy", "citationStyles", "venue", 'citationCount', 'title', 'url', 'externalIds', 'influentialCitationCount'}
METADATA_FIELDS = ",".join(CATEGORICAL_META_FIELDS.union(NUMERIC_META_FIELDS))


# Custom exception for API errors
class APIError(Exception):
    """Custom exception for API errors with status code."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class TaskIdAwareLogFormatter(Formatter):
    def __init__(self, task_id: str = ""):
        super().__init__("%(asctime)s - %(name)s - %(levelname)s")
        self.task_id = task_id

    def format(self, record):
        og_message = super().format(record)
        task_id_part = f"[{self.task_id}] " if self.task_id else ""
        return f"{og_message} - {task_id_part}- {record.getMessage()}"


def init_settings(logs_dir: str, log_level: str = "INFO", litellm_cache_dir: str = "litellm_cache") -> TaskIdAwareLogFormatter:
    def setup_logging() -> TaskIdAwareLogFormatter:
        # If LOG_FORMAT is "google:json" emit log message as JSON in a format Google Cloud can parse
        loggers = ["LiteLLM Proxy", "LiteLLM Router", "LiteLLM"]

        for logger_name in loggers:
            litellm_logger = logging.getLogger(logger_name)
            litellm_logger.setLevel(logging.WARNING)

        fmt = os.getenv("LOG_FORMAT")
        tid_log_fmt = TaskIdAwareLogFormatter()
        if fmt == "google:json":
            handlers = [glog.Handler()]
            for handler in handlers:
                handler.setFormatter(glog.Formatter(tid_log_fmt))
        else:
            handlers = []
            # log lower levels to stdout
            stdout_handler = logging.StreamHandler(stream=sys.stdout)
            stdout_handler.addFilter(lambda rec: rec.levelno <= logging.INFO)
            handlers.append(stdout_handler)

            # log higher levels to stderr (red)
            stderr_handler = logging.StreamHandler(stream=sys.stderr)
            stderr_handler.addFilter(lambda rec: rec.levelno > logging.INFO)
            handlers.append(stderr_handler)
            for handler in handlers:
                handler.setFormatter(tid_log_fmt)

        level = log_level
        logging.basicConfig(level=level, handlers=handlers)
        return tid_log_fmt

    def setup_local_llm_cache():
        # Local logs directory for litellm caching, event traces and state management
        local_cache_dir = f"{logs_dir}/{litellm_cache_dir}"
        # create parent and subdirectories for the local cache
        os.makedirs(local_cache_dir, exist_ok=True)
        setup_llm_cache(cache_type="disk", disk_cache_dir=local_cache_dir)

    tid_log_fmt = setup_logging()
    setup_local_llm_cache()
    return tid_log_fmt


def make_int(x: Optional[Any]) -> int:
    try:
        return int(x)
    except:
        return 0


def get_ref_author_str(authors: List[Dict[str, str]]) -> str:
    if not authors:
        return "NULL"
    f_author_lname = authors[0]["name"].split()[-1]
    return f_author_lname if len(authors) == 1 else f"{f_author_lname} et al."


def query_s2_api(
    end_pt: str = "paper/batch",
    params: Dict[str, Any] = None,
    payload: Dict[str, Any] = None,
    method: str = "get",
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Query the Semantic Scholar API with error handling."""
    url = S2_API_BASE_URL + end_pt
    req_method = requests.get if method == "get" else requests.post
    response = req_method(url, headers=S2_HEADERS, params=params, json=payload)
    if response.status_code != 200:
        import time
        logging.warning(f"S2 API request to end point {end_pt} failed with status code {response.status_code}. Retrying in 5 seconds...")
        time.sleep(10)
        response = req_method(url, headers=S2_HEADERS, params=params, json=payload)
        if response.status_code != 200:
            try:
                error_text = response.text
            except Exception:
                error_text = "<could not decode response text>"
            logging.error(
                f"S2 API request to end point {end_pt} failed again with status code {response.status_code}. "
                f"Response content: {error_text}"
            )
            raise APIError(
                f"S2 API request failed with status code {response.status_code}. Response content: {error_text}",
                status_code=500
            )
    return response.json()


def get_paper_metadata(corpus_ids: Set[str], fields=METADATA_FIELDS, batch_size: int = 25) -> Dict[str, Any]:
    """
    Fetch paper metadata from S2 API in batches to avoid exceeding request limits.
    """
    if not corpus_ids:
        return {}
    corpus_ids = list(corpus_ids)
    all_paper_metadata = {}
    for i in range(0, len(corpus_ids), batch_size):
        batch_ids = corpus_ids[i:i + batch_size]
        paper_data = query_s2_api(
            end_pt="paper/batch",
            params={"fields": fields},
            payload={"ids": ["CorpusId:{0}".format(cid) for cid in batch_ids]},
            method="post",
        )
        batch_metadata = {
            str(pdata["corpusId"]): {k: make_int(v) if k in NUMERIC_META_FIELDS else pdata.get(k) for k, v in pdata.items()}
            for pdata in paper_data
            if pdata and "corpusId" in pdata
        }
        # Add individual URLs for each paper in the batch
        for pdata in paper_data:
            if pdata and "corpusId" in pdata:
                cid = str(pdata["corpusId"])
                if "externalIds" in pdata and "ArXiv" in pdata["externalIds"]:
                    batch_metadata[cid]["arxiv_url"] = f"https://arxiv.org/pdf/{pdata['externalIds']['ArXiv']}"
                if "externalIds" in pdata and "ACL" in pdata["externalIds"]:
                    batch_metadata[cid]["acl_url"] = f"https://aclanthology.org/{pdata['externalIds']['ACL']}"
                if "url" in pdata:
                    batch_metadata[cid]["url"] = pdata["url"]
        all_paper_metadata.update(batch_metadata)
    return all_paper_metadata


def init_task(self, task_id: str, tool_request: Any):
    try:
        tool_request.user_id = str(uuid5(namespace=UUID(tool_request.user_id), name=f"nora-{UUID_NAMESPACE}"))
    except (ValueError, TypeError) as e:
        logging.warning(f"Failed to generate UUID for user_id: {e}")
        pass


def format_sections_to_markdown(row: List[Dict[str, Any]]) -> str:
    # Handle cases where row might not be a list (e.g., NaN from pandas).
    if not isinstance(row, list) or len(row) == 0:
        logger.warning(f"Row is not a list or is empty: {row}")
        return ""
    # convenience function to format the sections of a paper into markdown for function below
    # Convert the list of dictionaries to a DataFrame
    print('*********************')
    print(row)
    sentences_df = pd.DataFrame(row)
    if sentences_df.empty:
        return ""
    # Sort by 'char_offset' to ensure sentences are in the correct order
    sentences_df.sort_values(by="char_start_offset", inplace=True)

    # Group by 'section_title', concatenate sentences, and maintain overall order by the first 'char_offset'
    grouped = sentences_df.groupby("section_title", sort=False)["text"].apply("\n...\n".join)

    # Exclude sections titled 'Abstract' or 'Title'
    grouped = grouped[(grouped.index != "abstract") & (grouped.index != "title")]

    # Format as Markdown
    markdown_output = "\n\n".join(f"## {title}\n{text}" for title, text in grouped.items())
    return markdown_output    


def normalize_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    paper = {k: make_int(v) if k in NUMERIC_META_FIELDS else paper.get(k) 
            for k, v in paper.items()}
    if "externalIds" in paper:
        if "ArXiv" in paper["externalIds"]:
            paper["arxivId"] = paper["externalIds"]["ArXiv"]
            paper["arxiv_url"] = f"https://arxiv.org/pdf/{paper['externalIds']['ArXiv']}"
        if "ACL" in paper["externalIds"]:
            paper["aclId"] = paper["externalIds"]["ACL"]
            paper["acl_url"] = f"https://aclanthology.org/{paper['externalIds']['ACL']}"
    if "corpusId" in paper:
        paper["corpus_id"] = str(paper["corpusId"])
    # added so that the results are consistent with the snippet_search api results
    if 'text' not in paper:
        paper["text"] = paper["abstract"]
    if 'section_title' not in paper:
        paper["section_title"] = "abstract"
        paper["char_start_offset"] = 0
    if 'sentence_offsets' not in paper:
        paper["sentence_offsets"] = []
    if 'ref_mentions' not in paper:
        paper["ref_mentions"] = []
    if 'score' not in paper:
        paper["score"] = 0.0
    if 'stype' not in paper:
        paper["stype"] = "public_api"
    if 'pdf_hash' not in paper:
        paper["pdf_hash"] = ""
    return paper