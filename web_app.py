#!/usr/bin/env python3

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from scholarqa.llms.constants import GPT_41_MINI
from scholarqa.state_mgmt.local_state_mgr import LocalStateMgrClient
from scholarqa.utils import format_citation

from query_scholar import AVAILABLE_MODELS, load_config, setup_scholar_qa

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


class WebSocketStateMgr(LocalStateMgrClient):
    """Custom state manager that forwards progress updates to WebSocket clients"""

    def __init__(self, client_id: str, connection_manager: ConnectionManager, logs_dir: str = "web_logs"):
        # Initialize parent with a temporary logs directory
        super().__init__(logs_dir)
        self.client_id = client_id
        self.connection_manager = connection_manager
        # Store the current event loop for later use
        try:
            self.event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.event_loop = None

    def update_task_state(
        self,
        task_id: str,
        tool_request,
        status: str,
        step_estimated_time: int = 0,
        curr_response=None,
        task_estimated_time: str = None,
    ):
        """Forward progress updates to WebSocket client and call parent"""
        # Send WebSocket message
        message = {
            "type": "progress",
            "task_id": task_id,
            "status": status,
            "step_estimated_time": step_estimated_time,
            "task_estimated_time": task_estimated_time,
        }

        # Send the progress update asynchronously
        logger.info(f"Attempting to send progress update: {status}")
        if self.event_loop and not self.event_loop.is_closed():
            try:
                # Use call_soon_threadsafe to schedule the coroutine from any thread
                future = asyncio.run_coroutine_threadsafe(
                    self.connection_manager.send_message(message, self.client_id), self.event_loop
                )
                logger.info(f"Progress message scheduled successfully: {status}")
            except Exception as e:
                logger.error(f"Error scheduling progress update: {status} - {e}")
        else:
            logger.warning(f"No event loop available, cannot send progress: {status}")

        # Also call parent method to maintain state
        try:
            super().update_task_state(task_id, tool_request, status, step_estimated_time, curr_response, task_estimated_time)
        except Exception as e:
            logger.warning(f"Error calling parent update_task_state: {e}")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        # Load available configurations
        config_file = "config_example.json"
        configs = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    configs = json.load(f)
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
                pass

        config_names = list(configs.keys()) if configs else ["default"]

        # Prepare available models for the frontend
        available_models = []
        for model in sorted(AVAILABLE_MODELS):
            # Create display names from the model strings
            if "gpt-4" in model:
                display_name = model.replace("openai/", "").replace("-", " ").title()
                if "4.1" in model:
                    display_name = display_name.replace("4.1", "4.1")
                elif "4o" in model:
                    display_name = display_name.replace("4O", "4o")
            elif "claude" in model:
                display_name = model.replace("anthropic/", "").replace("-", " ").title()
                if "3 5" in display_name:
                    display_name = display_name.replace("3 5", "3.5")
            elif "llama" in model:
                display_name = "Llama 3.1 405B (Together AI)"
            else:
                display_name = model

            available_models.append({"value": model, "display_name": display_name})

        logger.info(f"Serving index.html with configs: {config_names}")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "config_names": config_names,
                "available_models": available_models,
                "default_model": GPT_41_MINI,
            },
        )
    except Exception as e:
        logger.error(f"Error in read_root: {e}")
        return HTMLResponse(f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>", status_code=500)


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
        await manager.send_message({"type": "error", "message": f"An error occurred: {str(e)}"}, client_id)
        manager.disconnect(client_id)


async def process_query(message: dict, client_id: str):
    """Process a scholar query and send progress updates via WebSocket"""
    try:
        query = message["query"]
        config_name = message.get("config_name", "llm_reranker")
        inline_tags = message.get("inline_tags", False)

        # Extract model selections from the frontend
        main_model = message.get("main_model", "")
        decomposer_model = message.get("decomposer_model", "")
        quote_extraction_model = message.get("quote_extraction_model", "")
        clustering_model = message.get("clustering_model", "")
        summary_generation_model = message.get("summary_generation_model", "")

        # Send initial status
        await manager.send_message({"type": "status", "step": "initializing", "message": "Loading configuration..."}, client_id)

        config = load_config("config_example.json", config_name)
        if not config:
            await manager.send_message({"type": "error", "message": f"Configuration '{config_name}' not found"}, client_id)
            return

        await manager.send_message(
            {"type": "status", "step": "config_loaded", "message": f"Using configuration: {config_name}"}, client_id
        )

        # Apply config defaults with fallbacks
        reranker = config.get("reranker", None)
        reranker_type = config.get("reranker_type", "llm")

        # Parse reranker model shortcuts
        if reranker == "0.6":
            reranker = "Qwen/Qwen3-Reranker-0.6B"
        elif reranker == "4":
            reranker = "Qwen/Qwen3-Reranker-4B"

        await manager.send_message(
            {"type": "status", "step": "setup_models", "message": "Setting up ScholarQA models..."}, client_id
        )

        # Initialize ScholarQA with custom state manager for progress tracking
        custom_state_mgr = WebSocketStateMgr(client_id, manager)

        # Use user-selected models or fall back to config defaults
        final_main_model = main_model or config.get("model")
        final_decomposer_model = decomposer_model or config.get("decomposer_model")
        final_quote_extraction_model = quote_extraction_model or config.get("quote_extraction_model")
        final_clustering_model = clustering_model or config.get("clustering_model")
        final_summary_generation_model = summary_generation_model or config.get("summary_generation_model")

        # Log the models being used
        logger.info(
            f"Using models - Main: {final_main_model}, "
            f"Decomposer: {final_decomposer_model}, "
            f"Quote Extraction: {final_quote_extraction_model}, "
            f"Clustering: {final_clustering_model}, "
            f"Summary Generation: {final_summary_generation_model}"
        )

        scholar_qa = setup_scholar_qa(
            reranker_model=reranker,
            reranker_type=reranker_type,
            reranker_llm_model=config.get("reranker_llm_model"),
            llm_model=final_main_model,
            decomposer_model=final_decomposer_model,
            quote_extraction_model=final_quote_extraction_model,
            clustering_model=final_clustering_model,
            summary_generation_model=final_summary_generation_model,
            fallback_model=config.get("fallback_model"),
            table_column_model=config.get("table_column_model"),
            table_value_model=config.get("table_value_model"),
            state_mgr=custom_state_mgr,
        )

        await manager.send_message(
            {"type": "status", "step": "processing", "message": "Processing your query... This may take several minutes."},
            client_id,
        )

        # # Run in executor to avoid blocking the event loop while allowing progress updates
        # import concurrent.futures

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     result = await asyncio.get_event_loop().run_in_executor(
        #         executor, lambda: scholar_qa.answer_query(query, inline_tags=inline_tags, output_format="latex")
        #     )

        # Process the query
        result = scholar_qa.answer_query(query, inline_tags=inline_tags, output_format="latex")

        await manager.send_message({"type": "status", "step": "formatting", "message": "Formatting results..."}, client_id)

        # Format the result
        all_citations = set()
        all_citations_plain_text = set()
        formatted_result = ""
        for section in result["sections"]:
            formatted_result += f"\n{section['title']}\n"
            formatted_result += "-" * len(section["title"]) + "\n"
            if section.get("tldr"):
                formatted_result += f"\nTLDR: {section['tldr']}\n\n"
            formatted_result += section["text"] + "\n"
            if section.get("citations"):
                for citation in section["citations"]:
                    paper = citation["paper"]
                    all_citations.add(paper["bibtex"])
                    all_citations_plain_text.add(format_citation(paper))
            formatted_result += "\n" + "=" * 80 + "\n\n"
        formatted_result += "\n" + "=" * 80 + "\n\n"
        formatted_result += "\n" + "CITATIONS" + "\n\n"
        formatted_result += "\n" + "=" * 80 + "\n\n"
        formatted_result += "\n\n".join(list(all_citations_plain_text))
        formatted_result += "\n" + "=" * 80 + "\n\n"
        formatted_result += "\n" + "BIBTEX CITATIONS" + "\n\n"
        formatted_result += "\n" + "=" * 80 + "\n\n"
        formatted_result += "\n".join(list(all_citations))
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
        await manager.send_message(
            {"type": "complete", "result": formatted_result, "cost": result.get("cost", 0), "output_file": output_filename},
            client_id,
        )

    except Exception as e:
        logger.error(f"Error processing query for {client_id}: {e}")
        import traceback

        traceback.print_exc()
        await manager.send_message({"type": "error", "message": f"Error processing query: {str(e)}"}, client_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
