#!/usr/bin/env python3

import argparse
import json
import logging
import os
from datetime import datetime

from scholarqa.config.config_setup import LogsConfig
from scholarqa.llms.constants import (GPT_4_TURBO, GPT_41, GPT_41_MINI, GPT_4o, 
                                      CLAUDE_3_OPUS, CLAUDE_35_SONNET, CLAUDE_4_SONNET,
                                      LLAMA_405_TOGETHER_AI)
from scholarqa.rag.reranker.reranker_base import QWENRerankerVLLM, LLMReranker, RERANKER_MAPPING
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


def load_config(config_path: str, config_name: str = "default") -> dict:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        return {}
    
    try:
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        if config_name in configs:
            return configs[config_name]
        else:
            print(f"Configuration '{config_name}' not found. Available configs: {list(configs.keys())}")
            return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def setup_scholar_qa(reranker_model: str, reranker_type: str = "qwen_vllm", reranker_llm_model: str = None,
                     llm_model: str = None, decomposer_model: str = None, 
                     quote_extraction_model: str = None, clustering_model: str = None,
                     summary_generation_model: str = None, fallback_model: str = None, 
                     table_column_model: str = None, table_value_model: str = None) -> ScholarQA:
    """Initialize and return a configured ScholarQA instance.

    Returns:
        ScholarQA: A configured instance of the ScholarQA class with Qwen reranker.
    """
    # Initialize the retriever with default settings
    retriever = FullTextRetriever(n_retrieval=256, n_keyword_srch=20)

    # Initialize the reranker based on type
    if reranker_type == "llm":
        # Use LLM-based reranker
        reranker = LLMReranker(
            model=reranker_llm_model or GPT_41_MINI,
            use_batch=True,
            max_tokens=10,
            temperature=0.0
        )
    elif reranker_type == "qwen_vllm":
        # Use QWEN VLLM reranker (default)
        reranker = QWENRerankerVLLM(model_name=reranker_model)
    elif reranker_type in RERANKER_MAPPING:
        # Use other reranker types from mapping
        reranker_class = RERANKER_MAPPING[reranker_type]
        if reranker_type == "qwen":
            reranker = reranker_class(model_name=reranker_model)
        else:
            reranker = reranker_class(reranker_model)
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}. Available types: {list(RERANKER_MAPPING.keys())}")

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
        decomposer_llm=decomposer_model,
        quote_extraction_llm=quote_extraction_model,
        clustering_llm=clustering_model,
        summary_generation_llm=summary_generation_model,
        fallback_llm=fallback_model,
        table_column_model=table_column_model,
        table_value_model=table_value_model,
        # Disable table generation for simplicity
        run_table_generation=False,
    )
    return scholar_qa


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Query ScholarQA with a scientific question.")
    parser.add_argument("-q", "--query", required=True, help="The scientific question to ask")
    parser.add_argument("--inline-tags", action="store_true", help="Include inline paper tags in the output")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file (default: config.json)")
    parser.add_argument("--config-name", type=str, default="default", help="Configuration name to use from the config file (default: default)")
    parser.add_argument("--output-prefix", type=str, help="Optional prefix for the output filename (e.g., 'experiment1' -> 'experiment1_scholar_query_results_...')")
    parser.add_argument(
        "--reranker",
        type=str,
        default="Qwen/Qwen3-Reranker-4B", 
        help="The reranker model to use. If you specify '0.6' or '4', it will use 'Qwen/Qwen3-Reranker-0.6B' or 'Qwen/Qwen3-Reranker-4B' respectively. You can also provide a full model string for other rerankers."
    )
    parser.add_argument(
        "--reranker-type",
        type=str,
        default="qwen_vllm",
        choices=list(RERANKER_MAPPING.keys()) + ["qwen_vllm"],
        help=f"The type of reranker to use. Available types: {', '.join(list(RERANKER_MAPPING.keys()) + ['qwen_vllm'])}. Use 'llm' for LLM-based reranking."
    )
    parser.add_argument(
        "--reranker-llm-model",
        type=str,
        help=f"The LLM model to use for LLM-based reranking (only used when --reranker-type=llm). If not specified, uses GPT_41_MINI."
    )
    parser.add_argument(
        "--model",
        type=str,
        help=f"The main LLM model to use for quote extraction, clustering, and summary generation. Available models: {', '.join(sorted(AVAILABLE_MODELS))}. If not specified, uses the default GPT_41_MINI."
    )
    parser.add_argument(
        "--decomposer-model",
        type=str,
        help=f"The LLM model to use for query decomposition/preprocessing. If not specified, uses the main model."
    )
    parser.add_argument(
        "--quote-extraction-model",
        type=str,
        help=f"The LLM model to use for extracting quotes from papers. If not specified, uses the main model."
    )
    parser.add_argument(
        "--clustering-model",
        type=str,
        help=f"The LLM model to use for clustering quotes into dimensions. If not specified, uses the main model."
    )
    parser.add_argument(
        "--summary-generation-model",
        type=str,
        help=f"The LLM model to use for generating the final summary sections. If not specified, uses the main model."
    )
    parser.add_argument(
        "--fallback-model", 
        type=str,
        help=f"The fallback LLM model to use. If not specified, uses GPT_41."
    )
    parser.add_argument(
        "--table-column-model",
        type=str, 
        help=f"The LLM model to use for table column generation. If not specified, uses GPT_41."
    )
    parser.add_argument(
        "--table-value-model",
        type=str,
        help=f"The LLM model to use for table value generation. If not specified, uses GPT_41."
    )
    args = parser.parse_args()
    
    # Load configuration from file
    config = load_config(args.config, args.config_name)
    
    if config:
        print(f"Using configuration '{args.config_name}' from {args.config}")
    
    # Apply config defaults to args (command line args override config)
    def apply_config_default(arg_name, config_key=None):
        config_key = config_key or arg_name
        if getattr(args, arg_name) is None and config_key in config:
            setattr(args, arg_name, config[config_key])
    
    # Apply configuration defaults
    apply_config_default('reranker')
    apply_config_default('reranker_type')
    apply_config_default('reranker_llm_model')
    apply_config_default('model')
    apply_config_default('decomposer_model')
    apply_config_default('quote_extraction_model')
    apply_config_default('clustering_model')
    apply_config_default('summary_generation_model')
    apply_config_default('fallback_model')
    apply_config_default('table_column_model')
    apply_config_default('table_value_model')
    
    # Validate all model arguments if provided
    models_to_validate = [
        ("--model", args.model),
        ("--decomposer-model", args.decomposer_model),
        ("--quote-extraction-model", args.quote_extraction_model),
        ("--clustering-model", args.clustering_model),
        ("--summary-generation-model", args.summary_generation_model),
        ("--fallback-model", args.fallback_model),
        ("--table-column-model", args.table_column_model),
        ("--table-value-model", args.table_value_model),
        ("--reranker-llm-model", args.reranker_llm_model),
    ]
    
    for arg_name, model in models_to_validate:
        if model and model not in AVAILABLE_MODELS:
            print(f"Error: Model '{model}' for {arg_name} is not available.")
            print(f"Available models: {', '.join(sorted(AVAILABLE_MODELS))}")
            return
    
    #parse the reranker model
    reranker_model = args.reranker
    if reranker_model == "0.6":
        reranker_model = "Qwen/Qwen3-Reranker-0.6B"
    elif reranker_model == "4":
        reranker_model = "Qwen/Qwen3-Reranker-4B"

    # Initialize ScholarQA
    scholar_qa = setup_scholar_qa(
        reranker_model,
        args.reranker_type,
        args.reranker_llm_model,
        args.model, 
        args.decomposer_model,
        args.quote_extraction_model,
        args.clustering_model,
        args.summary_generation_model,
        args.fallback_model,
        args.table_column_model, 
        args.table_value_model
    )

    try:
        # Process the query
        result = scholar_qa.answer_query(args.query, inline_tags=args.inline_tags, output_format="latex")

        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        
        # Generate output filename with timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_prefix:
            # Sanitize prefix to be filename-safe
            safe_prefix = "".join(c for c in args.output_prefix if c.isalnum() or c in '-_').strip()
            output_filename = f"outputs/{safe_prefix}_scholar_query_results_{timestamp}.txt"
        else:
            output_filename = f"outputs/scholar_query_results_{timestamp}.txt"
        
        with open(output_filename, "w") as f:
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
            
            # Write cost information to file
            if "cost" in result:
                f.write(f"\nTotal LLM Cost: ${result['cost']:.6f}\n")
        
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

        # Print cost information
        if "cost" in result:
            print(f"Total LLM Cost: ${result['cost']:.6f}")
            print("=" * 50)
        
        # Inform user about output file location
        print(f"\nResults saved to: {output_filename}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise


if __name__ == "__main__":
    main()
