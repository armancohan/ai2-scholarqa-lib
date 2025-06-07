import logging
import re
from abc import abstractmethod
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from scholarqa.rag.reranker.reranker_base import AbstractReranker
from scholarqa.rag.retriever_base import AbstractRetriever
from scholarqa.utils import make_int, get_ref_author_str, query_s2_api, format_sections_to_markdown
from anyascii import anyascii
from .citation_config import CitationTraceConfig

logger = logging.getLogger(__name__)


class AbsPaperFinder(AbstractRetriever):

    @abstractmethod
    def rerank(self, query: str, retrieved_ctxs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass


class PaperFinder(AbsPaperFinder):
    snippet_srch_fields = ["text", "snippetKind", "snippetOffset", "section", "annotations.refMentions",
                           "annotations.sentences.start", "annotations.sentences.end"]

    def __init__(self, retriever: AbstractRetriever, context_threshold: float = 0.0, n_rerank: int = -1,
                 max_date: Optional[str] = None):
        self.retriever = retriever
        self.context_threshold = context_threshold
        self.n_rerank = n_rerank
        self.max_date = max_date

    def retrieve_passages(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        """Retrieve relevant passages along with scores from an index for the given query"""
        filter_kwargs.update({
            "fields": ",".join([f"snippet.{f}" for f in self.snippet_srch_fields])
        })
        if self.max_date:
            filter_kwargs.update({"insertedBefore": '{}'.format(self.max_date)})
        return self.retriever.retrieve_passages(query, **filter_kwargs)

    def retrieve_additional_papers(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        if self.max_date:
            # S2 API doesnt have insertedBefore for this endpoint, so use publicationDateOrYear:
            if year_filter := filter_kwargs.pop("year", None):
                # if year filter is present, we need to remove it from the kwargs and resolve it to publicationDateOrYear
                date_start, date_end = year_filter.split("-") # YYYY-YYYY
                max_year = self.max_date.split("-")[0] #YYYY-MM
                # restrict both start and end to max_year and if end is max_year, set it to max_date
                date_start, date_end = min(date_start, max_year), date_end if date_end and date_end < max_year else self.max_date
                filter_kwargs.update({"publicationDateOrYear": '{}:{}'.format(date_start, date_end)})

            else:
                filter_kwargs.update({"publicationDateOrYear": ':{}'.format(self.max_date)})

        return self.retriever.retrieve_additional_papers(query, **filter_kwargs)

    def trace_citations_and_references(self, query: str, highly_ranked_papers: List[Dict[str, Any]], 
                                       current_candidates: List[Dict[str, Any]],
                                       citation_config: CitationTraceConfig) -> List[Dict[str, Any]]:
        """
        Trace citations and references of highly ranked papers and add them to results if highly relevant.
        """
        if not highly_ranked_papers:
            logger.info("No highly ranked papers provided for citation tracing")
            return []
        
        # Take the top N papers for citation tracing
        top_papers = highly_ranked_papers[:citation_config.top_n_papers]
        logger.info(f"Tracing citations and references for top {len(top_papers)} papers")
        
        traced_papers = []
        seen_corpus_ids = set()
        
        # Add existing paper corpus_ids to avoid duplicates
        for paper in current_candidates:
            seen_corpus_ids.add(paper.get("corpus_id"))
        
        for paper in top_papers:
            corpus_id = paper.get("corpus_id")
            if not corpus_id:
                continue
            logger.info(f"Tracing citations and references for paper {corpus_id}")
            # Get citations (papers that cite this paper)
            try:
                citations = self.retriever.get_paper_citations(
                    corpus_id, limit=citation_config.max_citations_per_paper
                )
                if citation_config.year_range:
                    citations = [citation for citation in citations if self._is_within_year_range(paper, citation, citation_config.year_range)]
                traced_papers.extend(citations)
            except Exception as e:
                logger.warning(f"Failed to get citations for paper {corpus_id}: {e}")
            # Get references (papers this paper cites)
            if not citation_config.citation_only:
                try:
                    references = self.retriever.get_paper_references(
                        corpus_id, limit=citation_config.max_references_per_paper
                    )
                    if citation_config.year_range:
                        references = [reference for reference in references if self._is_within_year_range(paper, reference, citation_config.year_range)]
                    traced_papers.extend(references)
                except Exception as e:
                    logger.warning(f"Failed to get references for paper {corpus_id}: {e}")
        # Remove duplicates and filter by relevance threshold
        unique_traced_papers = []
        for paper in traced_papers:
            if paper["corpus_id"] not in seen_corpus_ids:
                seen_corpus_ids.add(paper["corpus_id"])
                unique_traced_papers.append(paper)
        return unique_traced_papers
    
    def _is_within_year_range(self, original_paper: Dict[str, Any], traced_paper: Dict[str, Any], year_range: str) -> bool:
        """Check if paper is within the specified year range.
        
        Args:
            original_paper: The original paper
            traced_paper: The traced paper
            year_range: e.g., 2013-2015
            
        Returns:
        """
        try:
            year_start, year_end = year_range.split("-")
            traced_paper_year = traced_paper.get("year") or traced_paper.get("meta_year")
            if int(year_start) <= int(traced_paper_year):
                return True
            else:
                return False
        except (ValueError, AttributeError):
            return False
    
    def _is_before_max_date(self, paper: Dict[str, Any]) -> bool:
        """Check if paper is before the max_date threshold."""
        try:
            paper_date = paper.get("publicationDate")
            if not paper_date:
                return True  # If no date, assume it's valid
                
            # Simple year comparison (can be enhanced for full date comparison)
            paper_year = paper_date.get("year")
            max_year = int(self.max_date.split("-")[0])
            
            return paper_year and int(paper_year) <= max_year
        except (ValueError, AttributeError):
            return True


    def rerank(self, query: str, retrieved_ctxs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return retrieved_ctxs

    def expand_results_with_citations(self, query: str, reranked_candidates: List[Dict[str, Any]], 
                                    enable_citation_tracing: bool = True,
                                    citation_config: Optional[CitationTraceConfig] = None,
                                    is_second_round: bool = False) -> List[Dict[str, Any]]:
        """
        Expand retrieval results by tracing citations and references of highly ranked papers.
        """
        if not enable_citation_tracing:
            return reranked_candidates
        if citation_config is None:
            citation_config = CitationTraceConfig()
        logger.info("Starting citation tracing to expand search results")
        highly_ranked_papers = reranked_candidates[:citation_config.top_n_papers]
        # For second round, use the config's for_second_round method
        if is_second_round:
            citation_config = citation_config.for_second_round()
            highly_ranked_papers = reranked_candidates[:citation_config.top_n_papers]
        # Trace citations and references
        traced_papers = self.trace_citations_and_references(
            query=query,
            highly_ranked_papers=highly_ranked_papers,
            current_candidates=reranked_candidates,
            citation_config=citation_config
        )
        # Add traced papers to search_api_results (they're more like additional papers than passages)
        extended_results = reranked_candidates + traced_papers
        return extended_results

    def get_paper_citations(self, corpus_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get papers that cite the given paper by delegating to the internal retriever."""
        return self.retriever.get_paper_citations(corpus_id, limit)

    def get_paper_references(self, corpus_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get papers that are referenced by the given paper by delegating to the internal retriever."""
        return self.retriever.get_paper_references(corpus_id, limit)

    def aggregate_into_dataframe(self, snippets_list: List[Dict[str, Any]], paper_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Aggregate both snippet-level and paper-level results. Paper-level results (no 'text' and 'relevance_judgement') are included as-is and ranked by 'rerank_score'. Snippet-level are aggregated as before."""
        # Separate snippet-level and paper-level results
        snippet_level = [s for s in snippets_list if s.get("text") is not None and s["corpus_id"] in paper_metadata]
        paper_level = [s for s in snippets_list if s.get("text") is None and s["corpus_id"] in paper_metadata]

        aggregated_candidates = self.aggregate_snippets_to_papers(snippet_level, paper_metadata)
        # Only filter snippet-level by context threshold
        aggregated_candidates = [acand for acand in aggregated_candidates if acand.get("relevance_judgement", 0) >= self.context_threshold]

        # Sort snippet-level by relevance_judgement
        aggregated_candidates.sort(key=lambda x: x.get("relevance_judgement", 0), reverse=True)
        # Sort paper-level by rerank_score (default 0.0)
        paper_level_sorted = sorted(paper_level, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        # Concatenate: snippet-level first, then paper-level
        all_candidates = aggregated_candidates + paper_level_sorted
        return self.format_retrieval_response(all_candidates)

    @staticmethod
    def aggregate_snippets_to_papers(snippets_list: List[Dict[str, Any]], paper_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        logging.info(f"Aggregating {len(snippets_list)} passages at paper level with metadata")
        paper_snippets = dict()
        for snippet in snippets_list:
            corpus_id = snippet["corpus_id"]
            if corpus_id not in paper_snippets:
                paper_snippets[corpus_id] = paper_metadata[corpus_id].copy()
                paper_snippets[corpus_id]["corpus_id"] = corpus_id
                paper_snippets[corpus_id]["sentences"] = []
            if snippet.get("stype") != "public_api":
                paper_snippets[corpus_id]["sentences"].append(snippet)
            if "paperId" in paper_snippets[corpus_id]:
                del paper_snippets[corpus_id]["paperId"]
            paper_snippets[corpus_id]["relevance_judgement"] = max(
                paper_snippets[corpus_id].get("relevance_judgement", -1),
                snippet.get("rerank_score", snippet.get("score", 0)))
            if not paper_snippets[corpus_id].get("abstract") and snippet.get("section_title") == "abstract":
                paper_snippets[corpus_id]["abstract"] = snippet["text"]
        sorted_ctxs = sorted(paper_snippets.values(), key=lambda x: x["relevance_judgement"], reverse=True)
        logger.info(f"Scores after aggregation: {[s['relevance_judgement'] for s in sorted_ctxs]}")
        return sorted_ctxs

    
    def format_retrieval_response_orig(self, agg_reranked_candidates: List[Dict[str, Any]]) -> pd.DataFrame:

        df = pd.DataFrame(agg_reranked_candidates)
        try:
            df = df.drop(["text", "section_title", "ref_mentions", "score", "stype", "rerank_score"], axis=1)
        except Exception as e:
            logger.info(e)
        df = df[~df.sentences.isna() & ~df.year.isna()] if not df.empty else df
        if df.empty:
            return df
        df["corpus_id"] = df["corpus_id"].astype(int)

        # there are multiple relevance judgments in ['relevance_judgements'] for each paper
        # we will keep rows where ANY of the relevance judgments are 2 or 3
        # df = df[df["relevance_judgement"] >= self.context_threshold]

        if df.empty:
            return df

        # authors are lists of jsons. process with "name" key inside

        df["year"] = df["year"].apply(make_int)

        df["authors"] = df["authors"].fillna(value="")

        df.rename(
            columns={
                "citationCount": "citation_count",
                "referenceCount": "reference_count",
                "influentialCitationCount": "influential_citation_count",
            },
            inplace=True,
        )

        # drop corpusId
        df = df.drop(columns=["corpusId"])

        # now we need the big relevance_judgment_input_expanded
        # top of it
        # \n## Abstract\n{row['abstract']} --> Not using abstracts OR could use and not show
        prepend_text = df.apply(
            lambda
                row: f"# Title: {row['title']}\n# Venue: {row['venue']}\n"
                     f"# Authors: {', '.join([a['name'] for a in row['authors']])}\n## Abstract\n{row['abstract']}\n",
            axis=1,
        )
        section_text = df["sentences"].apply(format_sections_to_markdown)
        # update relevance_judgment_input
        df.loc[:, "relevance_judgment_input_expanded"] = prepend_text + section_text
        df["reference_string"] = df.apply(
            lambda
                row: anyascii(f"[{make_int(row.corpus_id)} | {get_ref_author_str(row.authors)} | "
                              f"{make_int(row['year'])} | Citations: {make_int(row['citation_count'])}]"),
            axis=1,
        )        
        return df    

    def format_retrieval_response(self, agg_reranked_candidates: List[Dict[str, Any]]) -> pd.DataFrame:
        import pandas as pd
        from .retrieval import get_author_info
        df = pd.DataFrame(agg_reranked_candidates)
        if df.empty:
            return df

        # Build paper_author_list for get_author_info
        paper_author_list = {}
        for idx, row in df.iterrows():
            paper_id = row.get("corpus_id")
            authors = row.get("authors")
            if paper_id is not None and authors:
                paper_author_list[paper_id] = authors
        author_info = get_author_info(paper_author_list)
        authorid_to_info = {str(a.get("authorId")): a for a in author_info["author_info_list"] if a.get("authorId")}
        paper_to_authors = author_info["paper_to_authors"]

        def get_affiliation_for_paper(paper_id, which):
            author_ids = paper_to_authors.get(paper_id, [])
            if not author_ids:
                return None
            if which == "first":
                aid = author_ids[0] if len(author_ids) > 0 else None
            else:
                aid = author_ids[-1] if len(author_ids) > 1 else (author_ids[0] if author_ids else None)
            author = authorid_to_info.get(str(aid))
            affs = author.get("affiliations") if author else None
            if affs and isinstance(affs, list) and len(affs) > 0:
                if isinstance(affs[0], dict):
                    return affs[0].get("name")
                return affs[0]
            return None
        
        df.rename(
            columns={
                "citationCount": "citation_count",
                "referenceCount": "reference_count",
                "influentialCitationCount": "influential_citation_count",
            },
            inplace=True,
        )        

        df["first_author_affiliation"] = df["corpus_id"].apply(lambda pid: get_affiliation_for_paper(pid, "first"))
        df["last_author_affiliation"] = df["corpus_id"].apply(lambda pid: get_affiliation_for_paper(pid, "last"))

        # Add "reference_string" (using the same lambda as in "format_retrieval_response_orig")
        df["reference_string"] = df.apply(
            lambda row: anyascii(f"[{make_int(row.corpus_id)} | {get_ref_author_str(row.authors)} | {make_int(row['year'])} | Citations: {make_int(row['citation_count'])}]"),
            axis=1
        )

        def get_latex_citation(row: pd.Series) -> str:
            """
            Extracts bibtex key and formats a \\citep citation string.
            We use \\citep by default as we cannot determine from here whether
            a textual citation (\\citet) is needed.
            """
            bibtex_str = row.get('citationStyles', {}).get('bibtex')
            if not bibtex_str or not isinstance(bibtex_str, str):
                return ""

            # Extract the citation key from a bibtex entry like @article{key, ...}
            match = re.search(r'@\w+\{([^,]+),', bibtex_str)
            if match:
                key = match.group(1).strip()
                return f"\\citep{{{key}}}"
            return ""             
        
        df["reference_string_latex"] = df.apply(get_latex_citation, axis=1)
        df["reference_bibtex"] = df.apply(lambda row: row.get('citationStyles', {}).get('bibtex'), axis=1)        
        
        # now we need the big relevance_judgment_input_expanded
        # top of it
        # \n## Abstract\n{row['abstract']} --> Not using abstracts OR could use and not show
        prepend_text = df.apply(
            lambda
                row: f"# Title: {row['title']}\n# Venue: {row['venue']}\n"
                     f"# Authors: {', '.join([a['name'] for a in row['authors']])}\n# Bibtex: {row['reference_bibtex']}\n# Citation Marker: {row['reference_string_latex']}\n## Abstract\n{row['abstract']}\n",
            axis=1,
        )
        section_text = df["sentences"].apply(format_sections_to_markdown)
        # update relevance_judgment_input
        df.loc[:, "relevance_judgment_input_expanded"] = prepend_text + section_text

        return df


class PaperFinderWithReranker(PaperFinder):
    def __init__(self, retriever: AbstractRetriever, reranker: AbstractReranker, n_rerank: int = -1,
                 context_threshold: float = 0.5, max_date: Optional[str] = None):
        super().__init__(retriever, context_threshold, n_rerank, max_date=max_date)
        if reranker:
            self.reranker_engine = reranker
        else:
            raise Exception(f"Reranker not initialized: {reranker}")

    def rerank(
            self, query: str, retrieved_ctxs: List[Dict[str, Any]], rerank_snippets: bool = True) -> List[Dict[str, Any]]:
        """Rerank the retrieved passages using a cross-encoder model and return the top n passages."""
        if rerank_snippets:
            passages = [doc["title"] + " " + doc["text"] if "title" in doc else doc["text"] for doc in retrieved_ctxs]
        else: # rerank papers
            passages = [doc.get("title") or "" + " " + doc.get("abstract" or "") for doc in retrieved_ctxs]
        rerank_scores = self.reranker_engine.get_scores(
            query, passages
        )
        logger.info(f"Reranker scores: {rerank_scores}")

        for doc, rerank_score in zip(retrieved_ctxs, rerank_scores):
            doc["rerank_score"] = rerank_score
        sorted_ctxs = sorted(
            retrieved_ctxs, key=lambda x: x["rerank_score"], reverse=True
        )
        sorted_ctxs = super().rerank(query, sorted_ctxs)
        sorted_ctxs = sorted_ctxs[:self.n_rerank] if self.n_rerank > 0 else sorted_ctxs
        logging.info(f"Done reranking: {len(sorted_ctxs)} passages remain")
        return sorted_ctxs

    def expand_results_with_citations(self, query: str, reranked_candidates: List[Dict[str, Any]], 
                                    enable_citation_tracing: bool = True,
                                    citation_config: Optional[CitationTraceConfig] = None) -> List[Dict[str, Any]]:
        """
        Enhanced citation expansion that also reranks the traced papers.
        """
        if not enable_citation_tracing:
            return reranked_candidates
        
        # Set default citation_trace_params if None
        if citation_config is None:
            citation_config = CitationTraceConfig()
            
        # Get enhanced results from parent method
        enhanced_retrieval_results = super().expand_results_with_citations(
            query, reranked_candidates, enable_citation_tracing, citation_config
        )
        
        # Find the newly added traced papers (those added by citation tracing)
        original_corpus_ids = {paper.get("corpus_id") for paper in reranked_candidates}
        traced_papers = [paper for paper in enhanced_retrieval_results 
                        if paper.get("corpus_id") not in original_corpus_ids]
        
        # If no traced papers were found, return original candidates
        if not traced_papers:
            logger.info("No traced papers found, returning original candidates")
            return reranked_candidates
            
        logger.info(f"Reranking {len(traced_papers)} traced papers")
        
        # Rerank the traced papers
        reranked_traced_papers = self.rerank(query, traced_papers, rerank_snippets=False)
        
        # Limit traced papers to the specified maximum
        max_traced_papers = citation_config.max_traced_papers
        reranked_traced_papers = reranked_traced_papers[:max_traced_papers]
        
        # Start with original candidates and add the reranked traced papers
        final_results = reranked_candidates + reranked_traced_papers
        
        # Optional: Second round of citation expansion (only if enabled and we have traced papers)
        enable_second_round = citation_config.enable_second_round
        if enable_second_round and reranked_traced_papers:
            logger.info("Expanding citations for second round")
            
            # Use top papers from first round for second expansion
            top_papers_for_second_round = reranked_traced_papers[:citation_config.second_round_top_n]
            
            # Create new citation params for second round (typically with smaller limits)
            second_round_config = citation_config.for_second_round()
            
            # Expand citations for second round
            second_round_results = super().expand_results_with_citations(
                query, top_papers_for_second_round, enable_citation_tracing, second_round_config, is_second_round=True
            )
            
            # Find new papers from second round
            existing_corpus_ids = {paper.get("corpus_id") for paper in final_results}
            new_second_round_papers = [paper for paper in second_round_results 
                                     if paper.get("corpus_id") not in existing_corpus_ids]
            
            if new_second_round_papers:
                logger.info(f"Reranking {len(new_second_round_papers)} second round traced papers")
                
                # Rerank second round papers
                reranked_second_round = self.rerank(query, new_second_round_papers, rerank_snippets=False)
                
                # Limit second round papers
                max_second_round = second_round_config.second_round_cap
                reranked_second_round = reranked_second_round[:max_second_round]
                logger.info(f"Reranked second round papers: {reranked_second_round} and added {len(reranked_second_round)} new papers")
                
                # Add to final results
                final_results.extend(reranked_second_round)
        
        return final_results

def get_author_info(paper_author_list: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve author information for a list of author names.

    Args:
        paper_author_list: Dict[paper_id, List[author_dict]] with 'name' and 'authorId' fields.

    Returns:
        Dict with:
            - 'author_info_list': List of unique author info dicts (name, affiliations, hIndex, citationCount, authorId)
            - 'paper_to_authors': Dict[paper_id, List[authorId]] (first and last only, or fewer if <2)
            - 'reconstruct_map': Function to reconstruct paper-author map from author_info_list
    """
    # 1. Collect first and last authorIds for each paper
    paper_to_authors = {}
    author_ids = set()
    for paper_id, authors in paper_author_list.items():
        if not authors:
            paper_to_authors[paper_id] = []
            continue
        ids = []
        if len(authors) == 1:
            aid = authors[0].get("authorId")
            if aid:
                ids.append(str(aid))
        else:
            first = authors[0].get("authorId")
            last = authors[-1].get("authorId")
            if first:
                ids.append(str(first))
            if last and last != first:
                ids.append(str(last))
        paper_to_authors[paper_id] = ids
        author_ids.update(ids)
    # 2. Batch query S2 API for all unique authorIds
    if not author_ids:
        return {"author_info_list": [], "paper_to_authors": paper_to_authors, "reconstruct_map": lambda x: {}}
    author_ids = list(author_ids)
    author_info_list = []
    batch_size = 40
    for i in range(0, len(author_ids), batch_size):
        batch = author_ids[i:i+batch_size]
        res = query_s2_api(
            end_pt="author/batch",
            params={"fields": "name,affiliations,hIndex,citationCount"},
            payload={"ids": batch},
            method="post"
        )
        if isinstance(res, dict) and "data" in res:
            res = res["data"]
        author_info_list.extend(res)
    # 3. Build a map from authorId to author info
    authorid_to_info = {str(a.get("authorId")): a for a in author_info_list if a.get("authorId")}
    # 4. Provide a function to reconstruct paper-author map
    def reconstruct_map(author_info_list):
        authorid_to_info = {str(a.get("authorId")): a for a in author_info_list if a.get("authorId")}
        return {pid: [authorid_to_info.get(aid) for aid in aids if aid in authorid_to_info] for pid, aids in paper_to_authors.items()}
    return {
        "author_info_list": author_info_list,
        "paper_to_authors": paper_to_authors,
        "reconstruct_map": reconstruct_map
    }
