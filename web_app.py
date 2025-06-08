#!/usr/bin/env python3

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from query_scholar import load_config, setup_scholar_qa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ScholarQA Web Interface")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Load available configurations
    config_file = "config.json"
    configs = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                configs = json.load(f)
        except Exception:
            pass
    
    config_names = list(configs.keys()) if configs else ["default"]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config_names": config_names
    })

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "query":
                await process_query(message, client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        await manager.send_message({
            "type": "error",
            "message": f"An error occurred: {str(e)}"
        }, client_id)
        manager.disconnect(client_id)

async def process_query(message: dict, client_id: str):
    """Process a scholar query and send progress updates via WebSocket"""
    try:
        query = message["query"]
        config_name = message.get("config_name", "default")
        inline_tags = message.get("inline_tags", False)
        
        # Send initial status
        await manager.send_message({
            "type": "status",
            "step": "initializing",
            "message": "Loading configuration..."
        }, client_id)
        
        # Load configuration
        config_file = "config.json"
        config = load_config(config_file, config_name)
        
        if not config and config_name != "default":
            await manager.send_message({
                "type": "error",
                "message": f"Configuration '{config_name}' not found"
            }, client_id)
            return
        
        await manager.send_message({
            "type": "status",
            "step": "config_loaded",
            "message": f"Using configuration: {config_name}"
        }, client_id)
        
        # Apply config defaults with fallbacks
        def get_config_value(key: str, fallback: str = None):
            return config.get(key, fallback)
        
        reranker = get_config_value('reranker', "Qwen/Qwen3-Reranker-4B")
        reranker_type = get_config_value('reranker_type', "qwen_vllm")
        
        # Parse reranker model shortcuts
        if reranker == "0.6":
            reranker = "Qwen/Qwen3-Reranker-0.6B"
        elif reranker == "4":
            reranker = "Qwen/Qwen3-Reranker-4B"
        
        await manager.send_message({
            "type": "status",
            "step": "setup_models",
            "message": "Setting up ScholarQA models..."
        }, client_id)
        
        # Initialize ScholarQA
        scholar_qa = setup_scholar_qa(
            reranker,
            reranker_type,
            get_config_value('reranker_llm_model'),
            get_config_value('model'),
            get_config_value('decomposer_model'),
            get_config_value('quote_extraction_model'),
            get_config_value('clustering_model'),
            get_config_value('summary_generation_model'),
            get_config_value('fallback_model'),
            get_config_value('table_column_model'),
            get_config_value('table_value_model')
        )
        
        await manager.send_message({
            "type": "status",
            "step": "processing",
            "message": "Processing your query... This may take several minutes."
        }, client_id)
        
        # Process the query
        result = scholar_qa.answer_query(query, inline_tags=inline_tags, output_format="latex")
        
        await manager.send_message({
            "type": "status",
            "step": "formatting",
            "message": "Formatting results..."
        }, client_id)
        
        # Format the result
        formatted_result = ""
        for section in result["sections"]:
            formatted_result += f"\n{section['title']}\n"
            formatted_result += "-" * len(section["title"]) + "\n"
            if section.get("tldr"):
                formatted_result += f"\nTLDR: {section['tldr']}\n\n"
            formatted_result += section["text"] + "\n"
            if section.get("citations"):
                formatted_result += "\nCitations:\n"
                formatted_result += "```\n"
                for citation in section["citations"]:
                    paper = citation["paper"]
                    formatted_result += paper['bibtex'] + "\n"
                formatted_result += "```\n"
            formatted_result += "\n" + "=" * 80 + "\n\n"
        
        # Add cost information
        if "cost" in result:
            formatted_result += f"\nTotal LLM Cost: ${result['cost']:.6f}\n"
        
        # Save to outputs directory
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"outputs/web_scholar_query_results_{timestamp}.txt"
        
        with open(output_filename, "w") as f:
            f.write(formatted_result)
        
        # Send completion message
        await manager.send_message({
            "type": "complete",
            "result": formatted_result,
            "cost": result.get("cost", 0),
            "output_file": output_filename
        }, client_id)
        
    except Exception as e:
        logger.error(f"Error processing query for {client_id}: {e}")
        await manager.send_message({
            "type": "error",
            "message": f"Error processing query: {str(e)}"
        }, client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)