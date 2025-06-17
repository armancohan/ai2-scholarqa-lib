import os
from abc import ABC, abstractmethod
from time import time
from typing import List, Any, Optional, Callable
from uuid import uuid5, UUID
import asyncio
import json

from nora_lib.tasks.state import IStateManager, StateManager

from scholarqa.llms.constants import CompletionResult, CostReportingArgs
from scholarqa.models import TaskResult, TaskStep, AsyncTaskState, ToolRequest

UUID_NAMESPACE = os.getenv("UUID_ENCODER_KEY", "ai2-scholar-qa")


class AbsStateMgrClient(ABC):
    def __init__(self):
        self.websocket_callback: Optional[Callable[[str, dict], None]] = None
        self._async_state_dir: Optional[str] = None
    
    @abstractmethod
    def get_state_mgr(self, tool_req: ToolRequest) -> IStateManager:
        pass

    def set_websocket_callback(self, callback: Callable[[str, dict], None]):
        self.websocket_callback = callback

    def init_task(self, task_id: str, tool_request: ToolRequest):
        pass

    def update_task_state(
            self,
            task_id: str,
            tool_req: ToolRequest,
            status: str,
            step_estimated_time: int = 0,
            curr_response: Any = None,
            task_estimated_time: str = None,
    ):
        state_mgr = self.get_state_mgr(tool_req)
        curr_step = TaskStep(description=status, start_timestamp=time())
        task_state = state_mgr.read_state(task_id)
        task_state.task_status = status
        if step_estimated_time:
            curr_step.estimated_timestamp = curr_step.start_timestamp + step_estimated_time
        if task_estimated_time:
            task_state.estimated_time = task_estimated_time
        if curr_response:
            task_state.task_result = TaskResult(sections=curr_response)
        task_state.extra_state["steps"].append(curr_step)
        state_mgr.write_state(task_state)
        
        # Write WebSocket update to a notification file for the main process to pick up
        if self.websocket_callback and self._async_state_dir:
            try:
                update_message = {
                    "type": "task_update",
                    "task_id": task_id,
                    "task_status": status,
                    "estimated_time": task_state.estimated_time,
                    "current_step": {
                        "description": status,
                        "start_timestamp": curr_step.start_timestamp,
                        "estimated_timestamp": curr_step.estimated_timestamp
                    },
                    "steps": [
                        {
                            "description": step.description if hasattr(step, 'description') else step.get('description', ''),
                            "start_timestamp": step.start_timestamp if hasattr(step, 'start_timestamp') else step.get('start_timestamp', 0),
                            "estimated_timestamp": step.estimated_timestamp if hasattr(step, 'estimated_timestamp') else step.get('estimated_timestamp', None)
                        } for step in task_state.extra_state["steps"]
                    ]
                }
                if curr_response:
                    update_message["partial_result"] = {
                        "sections": [section.model_dump() if hasattr(section, 'model_dump') else section for section in curr_response]
                    }
                
                # Write notification to a file that the main process can monitor
                notifications_dir = os.path.join(self._async_state_dir, "notifications")
                os.makedirs(notifications_dir, exist_ok=True)
                notification_file = os.path.join(notifications_dir, f"{task_id}_{time()}.json")
                
                with open(notification_file, 'w') as f:
                    json.dump(update_message, f)
                    
            except Exception as e:
                # Don't let WebSocket errors break the normal flow
                pass

    def report_llm_usage(self, completion_costs: List[CompletionResult], cost_args: CostReportingArgs) -> float:
        pass


class LocalStateMgrClient(AbsStateMgrClient):
    def __init__(self, logs_dir: str, async_state_dir: str = "async_state"):
        super().__init__()
        self._async_state_dir = f"{logs_dir}/{async_state_dir}"
        os.makedirs(self._async_state_dir, exist_ok=True)
        self.state_mgr = StateManager(AsyncTaskState, self._async_state_dir)

    def get_state_mgr(self, tool_req: Optional[ToolRequest] = None) -> IStateManager:
        return self.state_mgr

    def report_llm_usage(self, completion_costs: List[CompletionResult], cost_args: CostReportingArgs) -> float:
        return sum([cost.cost for cost in completion_costs])

    def init_task(self, task_id: str, tool_request: ToolRequest):
        try:
            tool_request.user_id = str(uuid5(namespace=UUID(tool_request.user_id), name=f"nora-{UUID_NAMESPACE}"))
        except Exception as e:
            pass
