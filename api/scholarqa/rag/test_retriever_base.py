from scholarqa.rag.retriever_base import FullTextRetriever
from scholarqa.rag.retrieval import PaperFinder
from scholarqa.rag.citation_config import CitationTraceConfig

def test_retrieval():
    # Initialize the retriever with smaller numbers for testing
    retriever = FullTextRetriever(n_retrieval=15, n_keyword_srch=13)
    
    # Test query
    test_query = "machine learning applications in healthcare"
    
    print("\nTesting retrieve_passages...")
    passages = retriever.retrieve_passages(test_query)
    print(f"Retrieved {len(passages)} passages")
    if passages:
        print("\nFirst passage sample:")
        print(f"Title: {passages[0].get('title', 'N/A')}")
        print(f"Text snippet: {passages[0].get('text', 'N/A')[:200]}...")
    
    print("\nTesting retrieve_additional_papers...")
    papers = retriever.retrieve_additional_papers(test_query)
    print(f"Retrieved {len(papers)} additional papers")
    if papers:
        print("\nFirst paper sample:")
        print(f"Title: {papers[0].get('title', 'N/A')}")
        print(f"Abstract snippet: {papers[0].get('text', 'N/A')[:200]}...")

def test_expand_results_with_citations():
    # Initialize the PaperFinder with retriever
    retriever = FullTextRetriever(n_retrieval=10, n_keyword_srch=5)
    paper_finder = PaperFinder(retriever, context_threshold=0.0, n_rerank=-1)
    
    # Test query
    test_query = "transformer architecture attention mechanism"
    
    print("\nTesting expand_results_with_citations...")
    
    # First get some initial results to expand
    initial_candidates = retriever.retrieve_passages(test_query)
    if not initial_candidates:
        print("No initial candidates found, skipping citation expansion test")
        return
    
    print(f"Starting with {len(initial_candidates)} initial candidates")
    
    # Test with enable_citation_tracing=False
    print("\nTesting with citation tracing disabled...")
    results_no_tracing = paper_finder.expand_results_with_citations(
        test_query, initial_candidates, enable_citation_tracing=False
    )
    print(f"Results without citation tracing: {len(results_no_tracing)} papers")
    assert len(results_no_tracing) == len(initial_candidates), "Should return same results when tracing disabled"

    # Test with enable_citation_tracing=True and default parameters
    print("\nTesting with citation tracing enabled (default params)...")
    config_default = CitationTraceConfig()
    results_with_tracing = paper_finder.expand_results_with_citations(
        test_query, initial_candidates, enable_citation_tracing=True, citation_config=config_default
    )
    print(f"Results with citation tracing: {len(results_with_tracing)} papers")
    print(f"Added {len(results_with_tracing) - len(initial_candidates)} papers through citation tracing")
    
    # Test with custom citation trace parameters
    print("\nTesting with custom citation trace parameters...")
    config_custom = CitationTraceConfig(
        top_n_papers=2,
        max_citations_per_paper=5,
        max_references_per_paper=5,
        max_traced_papers=5,
        enable_second_round=True,
        second_round_top_n=4,
        second_round_cap=4
    )
    results_custom_params = paper_finder.expand_results_with_citations(
        test_query, initial_candidates, enable_citation_tracing=True, 
        citation_config=config_custom
    )
    print(f"Results with custom params: {len(results_custom_params)} papers")
    print(f"Added {len(results_custom_params) - len(initial_candidates)} papers through custom citation tracing")
    
    # Verify structure of returned results
    if results_with_tracing:
        print("\nSample expanded result structure:")
        sample_result = results_with_tracing[0]
        print(f"Keys in result: {list(sample_result.keys())}")
        if 'corpus_id' in sample_result:
            print(f"Corpus ID: {sample_result['corpus_id']}")
        if 'title' in sample_result:
            print(f"Title: {sample_result['title']}")

if __name__ == "__main__":
    test_retrieval()
    test_expand_results_with_citations()
