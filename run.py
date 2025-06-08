#!/usr/bin/env python3
"""
Convenience script for running ScholarQA with predefined configurations.
"""

import subprocess
import sys
import argparse

def run_query(query, config_file="default", config_name="default", **kwargs):
    """Run a ScholarQA query with the specified configuration."""
    cmd = [
        sys.executable, "query_scholar.py",
        "-q", query,
        "--config-file", config_file,
        "--config-name", config_name
    ]

    print(f"Running the command: {cmd}")
    
    # Add any additional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running query: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run ScholarQA with predefined configurations")
    parser.add_argument("-q", "--query", required=True, help="The scientific question to ask")
    parser.add_argument("-c", "--config-file")
    parser.add_argument("-n", "--config-name", default="default", 
                       choices=["default", "fast", "llm_reranker", "premium"],
                       help="Configuration preset to use")
    parser.add_argument("--inline-tags", action="store_true", help="Include inline paper tags")
    parser.add_argument("--output-prefix", type=str, help="Optional prefix for the output filename")
    
    args = parser.parse_args()
    
    # Run the query with the specified configuration
    run_query(args.query, args.config_file, args.config_name, 
             inline_tags=args.inline_tags if args.inline_tags else None,
             output_prefix=args.output_prefix)

if __name__ == "__main__":
    main()