import logging
import multiprocessing
import os
from json import JSONDecodeError
from time import time
from typing import Union, Dict
from uuid import uuid4
import json
import asyncio
import glob
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from nora_lib.tasks.models import TASK_STATUSES, AsyncTaskState
from nora_lib.tasks.state import NoSuchTaskException

from scholarqa.config.config_setup import read_json_config
from scholarqa.models import (
    AsyncToolResponse,
    TaskResult,
    ToolRequest,
    ToolResponse,
    TaskStep
)
from scholarqa.rag.reranker.reranker_base import RERANKER_MAPPING
from scholarqa.rag.retrieval import PaperFinderWithReranker, PaperFinder
from scholarqa.rag.retriever_base import FullTextRetriever
from scholarqa.scholar_qa import ScholarQA
from scholarqa.state_mgmt.local_state_mgr import LocalStateMgrClient
from typing import Type, TypeVar

logger = logging.getLogger(__name__)

TIMEOUT = 240

async_context = multiprocessing.get_context("fork")

started_task_step = None

T = TypeVar("T", bound=ScholarQA)


class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.task_connections: Dict[str, str] = {}  # task_id -> client_id mapping

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        # Remove task connections for this client
        task_ids_to_remove = [task_id for task_id, cid in self.task_connections.items() if cid == client_id]
        for task_id in task_ids_to_remove:
            del self.task_connections[task_id]
        logger.info(f"WebSocket client {client_id} disconnected")

    def register_task(self, task_id: str, client_id: str):
        self.task_connections[task_id] = client_id

    async def send_task_update(self, task_id: str, message: dict):
        client_id = self.task_connections.get(task_id)
        if client_id and client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
                logger.debug(f"Sent task update for {task_id} to client {client_id}")
            except Exception as e:
                logger.error(f"Error sending task update to {client_id}: {e}")
                self.disconnect(client_id)

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)


websocket_manager = WebSocketManager()


async def monitor_notifications():
    """Background task to monitor notification files and send WebSocket updates"""
    notifications_dir = os.path.join(logs_config.log_dir, "async_state", "notifications")
    logger.info(f"Starting notification monitor for {notifications_dir}")
    
    while True:
        try:
            if os.path.exists(notifications_dir):
                # Find all notification files
                notification_files = glob.glob(os.path.join(notifications_dir, "*.json"))
                
                for file_path in notification_files:
                    try:
                        with open(file_path, 'r') as f:
                            message = json.load(f)
                        
                        # Send WebSocket update
                        if message.get("task_id"):
                            await websocket_manager.send_task_update(message["task_id"], message)
                            logger.debug(f"Sent WebSocket update for task {message['task_id']}")
                        
                        # Clean up the notification file
                        os.remove(file_path)
                        
                    except Exception as e:
                        logger.error(f"Error processing notification file {file_path}: {e}")
                        # Try to remove corrupted file
                        try:
                            os.remove(file_path)
                        except:
                            pass
            
            # Wait before checking again
            await asyncio.sleep(0.1)  # Check every 100ms for responsiveness
            
        except Exception as e:
            logger.error(f"Error in notification monitor: {e}")
            await asyncio.sleep(1)  # Wait longer on error


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    task = asyncio.create_task(monitor_notifications())
    logger.info("Started WebSocket notification monitor")
    yield
    # Shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("Stopped WebSocket notification monitor")


def lazy_load_state_mgr_client():
    state_mgr = LocalStateMgrClient(logs_config.log_dir, "async_state")
    state_mgr.set_websocket_callback(websocket_manager.send_task_update)
    return state_mgr


def lazy_load_scholarqa(task_id: str, sqa_class: Type[T] = ScholarQA, **sqa_args) -> T:
    retriever = FullTextRetriever(**run_config.retriever_args)
    if run_config.reranker_args:
        reranker = RERANKER_MAPPING[run_config.reranker_service](**run_config.reranker_args)
        paper_finder = PaperFinderWithReranker(retriever, reranker, **run_config.paper_finder_args)
    else:
        paper_finder = PaperFinder(retriever, **run_config.paper_finder_args)

    return sqa_class(paper_finder=paper_finder, task_id=task_id, state_mgr=app_config.state_mgr_client,
                     logs_config=logs_config, **run_config.pipeline_args, **sqa_args)


# setup logging config and local litellm cache
CONFIG_PATH = os.environ.get("CONFIG_PATH", "run_configs/default.json")

app_config = read_json_config(CONFIG_PATH)
logs_config = app_config.logs
run_config = app_config.run_config
app_config.load_scholarqa = lazy_load_scholarqa


def _do_task(tool_request: ToolRequest, task_id: str) -> TaskResult:
    """
    TODO: BYO logic here. Don't forget to define `ToolRequest` and `TaskResult`
    in `models.py`!

    The meat of whatever it is your tool or task agent actually
    does should be kicked off in here. This will be run synchonrously
    unless `_needs_to_be_async()` above returns True, in which case
    it will be run in a background process.

    If you need to update state for an asynchronously running task, you can
    use `task_state_manager.read_state(task_id)` to retrieve, and `.write_state()`
    to write back.
    """
    scholar_qa = app_config.load_scholarqa(task_id)
    return scholar_qa.run_qa_pipeline(tool_request)


def _estimate_task_length(tool_request: ToolRequest) -> str:
    """

    For telling the user how long to wait before asking for a status
    update on async tasks. This can just be a static guess, but you
    have access to the request if you want to do something fancier.
    """
    return (
        "~3 minutes"
    )


###########################################################################
### BELOW THIS LINE IS ALL TEMPLATE CODE THAT SHOULD NOT NEED TO CHANGE ###
###########################################################################


def create_app() -> FastAPI:
    app = FastAPI(root_path="/api", lifespan=lifespan)

    @app.get("/")
    def root(request: Request):
        return {"message": "Hello World", "root_path": request.scope.get("root_path")}

    @app.get("/health", status_code=204)
    def health():
        return "OK"

    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        await websocket_manager.connect(websocket, client_id)
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message["type"] == "register_task":
                    task_id = message["task_id"]
                    websocket_manager.register_task(task_id, client_id)
                    await websocket_manager.send_message({
                        "type": "task_registered",
                        "task_id": task_id
                    }, client_id)
                    
        except WebSocketDisconnect:
            websocket_manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
            websocket_manager.disconnect(client_id)

    @app.post("/query_corpusqa")
    def use_tool(
            tool_request: ToolRequest,
    ) -> Union[AsyncToolResponse, ToolResponse]:
        if not app_config.state_mgr_client:
            app_config.state_mgr_client = lazy_load_state_mgr_client()
        # Caller is asking for a status update of long-running request
        if tool_request.task_id:
            return _handle_async_task_check_in(tool_request)

        # New task
        task_id = str(uuid4())
        logs_config.task_id = task_id
        logger.info("New task")
        app_config.state_mgr_client.init_task(task_id, tool_request)
        estimated_time = _start_async_task(task_id, tool_request)

        return AsyncToolResponse(
            task_id=task_id,
            query=tool_request.query,
            estimated_time=estimated_time,
            task_status=TASK_STATUSES["STARTED"],
            task_result=None,
            steps=[started_task_step]
        )
    app.state.use_tool_fn = use_tool
    return app


def _start_async_task(task_id: str, tool_request: ToolRequest) -> str:
    global started_task_step
    estimated_time = _estimate_task_length(tool_request)
    tool_request.task_id = task_id
    task_state_manager = app_config.state_mgr_client.get_state_mgr(tool_request)
    started_task_step = TaskStep(description=TASK_STATUSES["STARTED"], start_timestamp=time(),
                                 estimated_timestamp=time() + TIMEOUT)
    task_state = AsyncTaskState(
        task_id=task_id,
        estimated_time=estimated_time,
        task_status=TASK_STATUSES["STARTED"],
        task_result=None,
        extra_state={"query": tool_request.query, "start": time(),
                     "steps": [started_task_step]},
    )
    task_state_manager.write_state(task_state)

    def _do_task_and_write_result():
        extra_state = {}
        try:
            task_result = _do_task(tool_request, task_id)
            task_status = TASK_STATUSES["COMPLETED"]
            extra_state["end"] = time()
        except Exception as e:
            task_result = None
            task_status = TASK_STATUSES["FAILED"]
            extra_state["error"] = str(e)

        state = task_state_manager.read_state(task_id)
        state.task_result = task_result
        state.task_status = task_status
        state.extra_state.update(extra_state)
        state.estimated_time = "--"
        task_state_manager.write_state(state)

    async_context.Process(
        target=_do_task_and_write_result,
        name=f"Async Task {task_id}",
        args=(),
    ).start()

    return estimated_time


def _handle_async_task_check_in(
        tool_req: ToolRequest,
) -> Union[ToolResponse | AsyncToolResponse]:
    """
    For tasks that will take a while to complete, we issue a task id
    that can be used to request status updates and eventually, results.

    This helper function is responsible for checking the state store
    and returning either the current state of the given task id, or its
    final result.
    """
    task_id = tool_req.task_id
    logs_config.task_id = task_id
    task_state_manager = app_config.state_mgr_client.get_state_mgr(tool_req)
    try:
        task_state = task_state_manager.read_state(task_id)
    except NoSuchTaskException:
        raise HTTPException(
            status_code=404, detail=f"Referenced task {task_id} does not exist."
        )
    except JSONDecodeError as e:
        logger.warning("state file is corrupted, should be updated on next poll: {e}")
        return AsyncToolResponse(
            task_id=task_id,
            query="",
            estimated_time="~3 minutes",
            task_status=f"{time()}: Processing user query",
            task_result=None,
        )

    # Retrieve data, which is just on local disk for now
    if task_state.task_status == TASK_STATUSES["FAILED"]:
        if task_state.extra_state and "error" in task_state.extra_state:
            msg = f"\nError: {task_state.extra_state['error']}"
            logger.exception(msg)
        else:
            msg = f"Referenced task failed."
        raise HTTPException(status_code=500, detail=f"{msg}")

    if task_state.task_status == TASK_STATUSES["COMPLETED"]:
        if not task_state.task_result:
            msg = f"Task marked completed but has no result."
            logger.error(msg)
            raise HTTPException(
                status_code=500,
                detail=msg,
            )

        if "start" in task_state.extra_state and "end" in task_state.extra_state:
            try:
                cost = task_state.task_result["cost"] if type(
                    task_state.task_result) == dict else task_state.task_result.cost
            except Exception as e:
                logger.warning(f"Error occurred while parsing cost from the response: {e}")
                cost = 0.0
            logger.info(
                f"completed in {task_state.extra_state['end'] - task_state.extra_state['start']} seconds, "
                f"cost: ${cost}")
        return ToolResponse(
            task_id=task_state.task_id,
            query=task_state.extra_state["query"],
            task_result=task_state.task_result,
        )

    if task_state.task_status not in {TASK_STATUSES["COMPLETED"],
                                      TASK_STATUSES["FAILED"]} and "start" in task_state.extra_state:
        elapsed = time() - task_state.extra_state["start"]
        if elapsed > TIMEOUT:
            task_state.task_status = TASK_STATUSES["FAILED"]
            task_state.extra_state["error"] = f"Task timed out after {TIMEOUT} seconds"
            task_state_manager.write_state(task_state)
            logger.info(f"timed out after {time() - task_state.extra_state['start']} seconds.")
            raise HTTPException(
                status_code=500,
                detail=f"Task timed out after {TIMEOUT} seconds.")

    return AsyncToolResponse(
        task_id=task_state.task_id,
        query=task_state.extra_state["query"],
        estimated_time=task_state.estimated_time,
        task_status=task_state.task_status,
        task_result=task_state.task_result,
        steps=task_state.extra_state.get("steps", []),
    )
