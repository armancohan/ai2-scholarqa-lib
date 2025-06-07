#!/usr/bin/env python3

import argparse
import logging

from scholarqa.config.config_setup import LogsConfig
from scholarqa.llms.constants import (GPT_4_TURBO, GPT_41, GPT_41_MINI, GPT_4o, 
                                      CLAUDE_3_OPUS, CLAUDE_35_SONNET, CLAUDE_4_SONNET,
                                      LLAMA_405_TOGETHER_AI)
from scholarqa.rag.reranker.reranker_base import QWENRerankerVLLM
from scholarqa.rag.retrieval import PaperFinderWithReranker
from scholarqa.rag.retriever_base import FullTextRetriever
from scholarqa.scholar_qa import ScholarQA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available models from constants.py
AVAILABLE_MODELS = {
    GPT_4_TURBO,
    GPT_41,
    GPT_41_MINI,
    GPT_4o,
    CLAUDE_3_OPUS,
    CLAUDE_35_SONNET,
    CLAUDE_4_SONNET,
    LLAMA_405_TOGETHER_AI,
}


def setup_scholar_qa(reranker_model: str, llm_model: str = None) -> ScholarQA:
    """Initialize and return a configured ScholarQA instance.

    Returns:
        ScholarQA: A configured instance of the ScholarQA class with Qwen reranker.
    """
    # Initialize the retriever with default settings
    retriever = FullTextRetriever(n_retrieval=256, n_keyword_srch=20)

    # Initialize the Qwen reranker
    reranker = QWENRerankerVLLM(model_name=reranker_model)

    # Initialize the paper finder with the retriever and reranker
    paper_finder = PaperFinderWithReranker(
        retriever=retriever,
        reranker=reranker,
        # Number of papers to rerank
        n_rerank=50,
        # Minimum relevance score threshold
        context_threshold=0.5,
    )

    # Create logs config
    logs_config = LogsConfig(llm_cache_dir="llm_cache")
    logs_config.init_formatter()

    # Initialize ScholarQA with default settings
    scholar_qa = ScholarQA(
        paper_finder=paper_finder,
        logs_config=logs_config,
        llm_model=llm_model,
        # Disable table generation for simplicity
        run_table_generation=False,
    )
    return scholar_qa


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Query ScholarQA with a scientific question.")
    parser.add_argument("-q", "--query", required=True, help="The scientific question to ask")
    parser.add_argument("--inline-tags", action="store_true", help="Include inline paper tags in the output")
    parser.add_argument(
        "--reranker",
        type=str,
        default="Qwen/Qwen3-Reranker-4B", 
        help="The reranker to use. If you specify '0.6' or '4', it will use 'Qwen/Qwen3-Reranker-0.6B' or 'Qwen/Qwen3-Reranker-4B' respectively. You can also provide a full model string for other rerankers."
    )
    parser.add_argument(
        "--model",
        type=str,
        help=f"The LLM model to use. Available models: {', '.join(sorted(AVAILABLE_MODELS))}. If not specified, uses the default GPT_41_MINI."
    )
    args = parser.parse_args()
    
    # Validate model if provided
    if args.model and args.model not in AVAILABLE_MODELS:
        print(f"Error: Model '{args.model}' is not available.")
        print(f"Available models: {', '.join(sorted(AVAILABLE_MODELS))}")
        return
    
    #parse the reranker model
    reranker_model = args.reranker
    if reranker_model == "0.6":
        reranker_model = "Qwen/Qwen3-Reranker-0.6B"
    elif reranker_model == "4":
        reranker_model = "Qwen/Qwen3-Reranker-4B"

    # Initialize ScholarQA
    scholar_qa = setup_scholar_qa(reranker_model, args.model)

    try:
        # Process the query
        result = scholar_qa.answer_query(args.query, inline_tags=args.inline_tags, output_format="latex")

        with open("scholar_query_results.txt", "w") as f:
            # Print the results
            for section in result["sections"]:
                f.write(f"\n{section['title']}\n")
                f.write("-" * len(section["title"]) + "\n")
                if section.get("tldr"):
                    f.write(f"\nTLDR: {section['tldr']}\n\n")
                f.write(section["text"] + "\n")
                if section.get("citations"):
                    f.write("\nCitations:\n")
                    f.write("```\n")
                    for citation in section["citations"]:
                        paper = citation["paper"]
                        f.write(paper['bibtex'] + "\n")
                    f.write("```\n")
                f.write("\n" + "=" * 80 + "\n\n")
        
        # Print the results
        for section in result["sections"]:
            print(f"\n{section['title']}")
            print("-" * len(section["title"]))
            if section.get("tldr"):
                print(f"\nTLDR: {section['tldr']}\n")
            print(section["text"])
            if section.get("citations"):
                print("\nCitations:")
                print("```")
                for citation in section["citations"]:
                    paper = citation["paper"]
                    print(paper['bibtex'])
                print("```")
            print("\n" + "=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise


if __name__ == "__main__":
    main()
