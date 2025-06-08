#!/usr/bin/env python3
"""
Interactive script to create custom configurations for ScholarQA.
"""

import json
import os
from typing import Dict, Any

def get_available_models():
    """Return list of available models."""
    return [
        "openai/gpt-4-turbo-2024-04-09",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini", 
        "openai/gpt-4o-2024-08-06",
        "anthropic/claude-3-opus-20240229",
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-sonnet-4-20250514",
        "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    ]

def get_reranker_types():
    """Return list of available reranker types."""
    return ["qwen_vllm", "llm", "crossencoder", "biencoder", "flag_embedding", "qwen"]

def create_config_interactive():
    """Create a configuration interactively."""
    print("=== ScholarQA Configuration Creator ===\n")
    
    config = {}
    models = get_available_models()
    reranker_types = get_reranker_types()
    
    # Display available models
    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print()
    
    # Reranker configuration
    print("Reranker configuration:")
    print("Available reranker types:", ", ".join(reranker_types))
    config["reranker_type"] = input(f"Reranker type [qwen_vllm]: ").strip() or "qwen_vllm"
    
    if config["reranker_type"] == "qwen_vllm":
        config["reranker"] = input("Qwen reranker model [Qwen/Qwen3-Reranker-4B]: ").strip() or "Qwen/Qwen3-Reranker-4B"
    elif config["reranker_type"] == "llm":
        print("\nSelect LLM for reranking:")
        choice = input("Model number or name: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            config["reranker_llm_model"] = models[int(choice) - 1]
        else:
            config["reranker_llm_model"] = choice or "openai/gpt-4.1-mini"
    
    # Main model configuration
    print("\nMain pipeline models:")
    
    def get_model_choice(prompt, default="openai/gpt-4.1-mini"):
        choice = input(f"{prompt} [{default}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        return choice or default
    
    config["model"] = get_model_choice("Main model (fallback for all steps)")
    config["decomposer_model"] = get_model_choice("Query decomposition model", config["model"])
    config["quote_extraction_model"] = get_model_choice("Quote extraction model", config["model"])
    config["clustering_model"] = get_model_choice("Quote clustering model", config["model"])
    config["summary_generation_model"] = get_model_choice("Summary generation model", config["model"])
    config["fallback_model"] = get_model_choice("Fallback model", "openai/gpt-4.1")
    
    use_tables = input("\nInclude table generation models? [y/N]: ").strip().lower()
    if use_tables in ['y', 'yes']:
        config["table_column_model"] = get_model_choice("Table column model", "openai/gpt-4.1")
        config["table_value_model"] = get_model_choice("Table value model", "openai/gpt-4.1")
    
    return config

def save_config(config: Dict[str, Any], config_name: str, config_file: str = "config.json"):
    """Save configuration to file."""
    configs = {}
    
    # Load existing configs if file exists
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                configs = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing config file: {e}")
    
    # Add new config
    configs[config_name] = config
    
    # Save updated configs
    try:
        with open(config_file, 'w') as f:
            json.dump(configs, f, indent=2)
        print(f"\nConfiguration '{config_name}' saved to {config_file}")
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False

def main():
    print("This script will help you create a custom ScholarQA configuration.\n")
    
    config_name = input("Configuration name: ").strip()
    if not config_name:
        print("Configuration name is required.")
        return
    
    config = create_config_interactive()
    
    print(f"\nConfiguration preview:")
    print(json.dumps(config, indent=2))
    
    confirm = input(f"\nSave this configuration as '{config_name}'? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        if save_config(config, config_name):
            print(f"\nYou can now use this configuration with:")
            print(f"python query_scholar.py -q \"your question\" --config-name {config_name}")
            print(f"or")
            print(f"python run.py -q \"your question\" -c {config_name}")

if __name__ == "__main__":
    main()