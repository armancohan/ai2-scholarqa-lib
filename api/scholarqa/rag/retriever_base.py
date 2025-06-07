from abc import ABC, abstractmethod
from typing import List, Any, Dict
import logging
from dotenv import load_dotenv
from scholarqa.utils import query_s2_api, METADATA_FIELDS, make_int, NUMERIC_META_FIELDS, get_paper_metadata, normalize_paper

load_dotenv()

logger = logging.getLogger(__name__)


class AbstractRetriever(ABC):
    @abstractmethod
    def retrieve_passages(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def retrieve_additional_papers(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_paper_citations(self, corpus_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get papers that cite the given paper."""
        pass

    @abstractmethod
    def get_paper_references(self, corpus_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get papers that are referenced by the given paper."""
        pass


class FullTextRetriever(AbstractRetriever):
    def __init__(self, n_retrieval: int = 256, n_keyword_srch: int = 20):
        self.n_retrieval = n_retrieval
        self.n_keyword_srch = n_keyword_srch

    def retrieve_passages(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        """Query the Semantic Scholar API full text search end point to retrieve papers based on the query.
        The full text search end point does not return the required metadata fields, so we will fetch metadata for
        these later."""
        snippets_list = self.snippet_search(query, **filter_kwargs)
        snippets_list = [
            snippet for snippet in snippets_list if len(snippet["text"].split(" ")) > 20
        ]
        # Use get_paper_metadata to fetch metadata for all collected corpus_ids
        corpus_ids = [snippet["corpus_id"] for snippet in snippets_list]
        if corpus_ids:
            # get_paper_metadata expects a set of strings
            meta_map = get_paper_metadata(set(str(cid) for cid in corpus_ids))
            # Merge metadata into each snippet by corpus_id
            for snippet in snippets_list:
                meta = meta_map[str(snippet["corpus_id"])]
                if "title" in meta:
                    snippet["meta_title"] = meta["title"]
                if "authors" in meta:
                    snippet["meta_authors"] = meta["authors"]
                if "year" in meta:
                    snippet["meta_year"] = meta["year"]
                if "abstract" in meta:
                    snippet["meta_abstract"] = meta["abstract"]
                if "citationCount" in meta:
                    snippet["meta_citationCount"] = meta["citationCount"]
                if "influentialCitationCount" in meta:
                    snippet["meta_influentialCitationCount"] = meta["influentialCitationCount"]
                if "url" in meta:
                    snippet["meta_url"] = meta["url"]
                if "externalIds" in meta:
                    snippet["meta_externalIds"] = meta["externalIds"]
                    if "ArXiv" in meta["externalIds"]:
                        snippet["meta_arxivId"] = meta["externalIds"]["ArXiv"]
                        snippet["meta_arxiv_url"] = f"https://arxiv.org/pdf/{meta['externalIds']['ArXiv']}"
                    if "ACL" in meta["externalIds"]:
                        snippet["meta_aclId"] = meta["externalIds"]["ACL"]
                        snippet["meta_acl_url"] = f"https://aclanthology.org/{meta['externalIds']['ACL']}"
                if "fieldsOfStudy" in meta:
                    snippet["meta_fieldsOfStudy"] = meta["fieldsOfStudy"]
        return snippets_list

    def snippet_search(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        if not self.n_retrieval:
            return []
        query_params = {fkey: fval for fkey, fval in filter_kwargs.items() if fval}
        query_params.update({"query": query, "limit": self.n_retrieval})
        snippets = query_s2_api(
            end_pt="snippet/search",
            params=query_params,
            method="get",
        )
        snippets_list = []
        res_data = snippets["data"]
        if res_data:
            for fields in res_data:
                res_map = dict()
                snippet, paper = fields["snippet"], fields["paper"]
                res_map["corpus_id"] = paper["corpusId"]
                res_map["title"] = paper["title"]
                res_map["text"] = snippet["text"]
                res_map["score"] = fields["score"]
                res_map["section_title"] = snippet["snippetKind"]
                if snippet["snippetKind"] == "body":
                    if section := snippet.get("section"):
                        res_map["section_title"] = section
                if "snippetOffset" in snippet and snippet["snippetOffset"].get("start"):
                    res_map["char_start_offset"] = snippet["snippetOffset"]["start"]
                else:
                    res_map["char_start_offset"] = 0
                if "annotations" in snippet and "sentences" in snippet["annotations"] and snippet["annotations"][
                    "sentences"]:
                    res_map["sentence_offsets"] = snippet["annotations"]["sentences"]
                else:
                    res_map["sentence_offsets"] = []

                if snippet.get("annotations") and snippet["annotations"].get("refMentions"):
                    res_map["ref_mentions"] = [rmen for rmen in
                                               snippet["annotations"]["refMentions"] if rmen.get("matchedPaperCorpusId")
                                               and rmen.get("start") and rmen.get("end")]
                else:
                    res_map["ref_mentions"] = []
                res_map["pdf_hash"] = snippet.get("extractionPdfHash","")
                res_map["stype"] = "vespa"
                if res_map:
                    snippets_list.append(res_map)
        
        return snippets_list

    def retrieve_additional_papers(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        return self.keyword_search(query, **filter_kwargs) if self.n_keyword_srch else []

    def keyword_search(self, kquery: str, **filter_kwargs) -> List[Dict[str, Any]]:
        """Query the Semantic Scholar API keyword search end point and return top n papers.
        The keyword search api also accepts filters for fields like year, venue, etc. which we obtain after decomposing
        the initial user query. This end point returns the required metadata fields as well, so we will skip fetching
        metadata for these later."""

        paper_data = []
        query_params = {fkey: fval for fkey, fval in filter_kwargs.items() if fval}
        query_params.update({"query": kquery, "limit": self.n_keyword_srch, "fields": METADATA_FIELDS})
        res = query_s2_api(
            end_pt="paper/search",
            params=query_params,
            method="get",
        )
        if "data" in res:
            paper_data = res["data"]
            paper_data = [pd for pd in paper_data if pd.get("corpusId") and pd.get("title") and pd.get("abstract")]
            paper_data = [{k: make_int(v) if k in NUMERIC_META_FIELDS else pd.get(k) for k, v in pd.items()}
                          for pd in paper_data]
            for pd in paper_data:
                pd["corpus_id"] = str(pd["corpusId"])
                pd["text"] = pd["abstract"]
                pd["section_title"] = "abstract"
                pd["char_start_offset"] = 0
                pd["sentence_offsets"] = []
                pd["ref_mentions"] = []
                pd["score"] = 0.0
                pd["stype"] = "public_api"
                pd["pdf_hash"] = ""
        return paper_data

    def get_paper_citations(self, corpus_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get papers that cite the given paper using Semantic Scholar API."""
        try:
            extended_limit = limit * 2  # make sure we include highly influential citations
            meta_data_fields = METADATA_FIELDS + ",isInfluential"
            response = query_s2_api(
                end_pt=f"paper/CorpusId:{corpus_id}/citations",
                params={"limit": extended_limit, "fields": meta_data_fields},
                method="get"
            )
            
            citations = []
            if "data" in response and response["data"]:
                for citation_entry in response["data"]:
                    if citation_entry.get("citingPaper"):
                        paper = citation_entry["citingPaper"]
                        if paper.get("corpusId") and paper.get("title"):
                            # Normalize the paper data format
                            normalized_paper = normalize_paper(paper)
                            citations.append(normalized_paper)
            
            logger.info(f"Retrieved {len(citations)} citations for paper {corpus_id}")
            return citations
        except Exception as e:
            logger.error(f"Failed to get citations for paper {corpus_id}: {e}")
            return []

    def get_paper_references(self, corpus_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get papers that are referenced by the given paper using Semantic Scholar API."""
        try:
            meta_data_fields = METADATA_FIELDS + ",isInfluential"
            extended_limit = limit * 2  # make sure we include highly influential citations
            response = query_s2_api(
                end_pt=f"paper/CorpusId:{corpus_id}/references",
                params={"limit": extended_limit, "fields": meta_data_fields},
                method="get"
            )
            
            references = []
            if "data" in response and response["data"]:
                for reference_entry in response["data"]:
                    if reference_entry.get("citedPaper"):
                        paper = reference_entry["citedPaper"]
                        if paper.get("corpusId") and paper.get("title"):
                            # Normalize the paper data format
                            normalized_paper = normalize_paper(paper)
                            references.append(normalized_paper)
            # sort by isInfluential
            references.sort(key=lambda x: x.get("isInfluential", 0), reverse=True)
            references = references[:limit]
            
            logger.info(f"Retrieved {len(references)} references for paper {corpus_id}")
            return references
        except Exception as e:
            logger.error(f"Failed to get references for paper {corpus_id}: {e}")
            return []
