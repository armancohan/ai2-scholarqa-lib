#!/usr/bin/env python3

"""
ScholarQA Web Application Launcher

This script launches the ScholarQA web interface using FastAPI and Uvicorn.
Run with: python run_app.py
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Launch ScholarQA Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], 
                        help="Log level (default: info)")
    
    args = parser.parse_args()
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Check if required files exist
    required_files = ["web_app.py", "query_scholar.py", "templates/index.html"]
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure all required files are present.")
        sys.exit(1)
    
    # Check if config file exists
    config_file = Path("config.json")
    if not config_file.exists():
        config_example = Path("config_example.json")
        if config_example.exists():
            print("⚠️  No config.json found. Using config_example.json as reference.")
            print("   Consider copying config_example.json to config.json for custom configurations.")
        else:
            print("⚠️  No configuration file found. Using default settings.")
    
    print("🚀 Starting ScholarQA Web Interface...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print("🔧 Configuration files checked")
    print("📁 Working directory:", script_dir)
    print("\n" + "="*50)
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    try:
        uvicorn.run(
            "web_app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()