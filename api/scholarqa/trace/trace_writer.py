import json
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TraceWriter(ABC):
    @abstractmethod
    def write(self, trace_json, file_name: str) -> None:
        pass


class LocalWriter(TraceWriter):
    def __init__(self, local_dir: str):
        self.local_dir = local_dir
        if not os.path.exists(local_dir):
            logger.info(f"Creating local directory to record traces: {local_dir}")
            os.makedirs(local_dir)

    def write(self, trace_json, file_name: str) -> None:
        try:
            with open(f"{self.local_dir}/{file_name}.json", "w") as f:
                json.dump(trace_json.__dict__, f, indent=4)
            logger.info(f"Pushed event trace to local path: {self.local_dir}/{file_name}.json")
        except Exception as e:
            logger.info(f"Error pushing {file_name} to local directory: {e}")
