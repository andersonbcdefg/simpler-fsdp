import os
import json
import time
from typing import TypedDict
# need to cast entry to Entry
from typing import cast

class Entry(TypedDict):
    timestamp: float
    loss: float
    lr: float
    step: int

class Logger:
    def __init__(self, run_name: str, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.logs: list[Entry] = []
        self.run_name = run_name
        self.log_file = f"{log_dir}/{run_name}_{round(time.time())}.jsonl"
        os.makedirs(log_dir, exist_ok=True)

    def log(self, entry: dict):
        if not self.enabled:
            return
        entry['timestamp'] = time.time()
        self.logs.append(cast(Entry, entry))
        if len(self.logs) > 100:
            with open(self.log_file, "a") as f:
                for log in self.logs:
                    json.dump(log, f)
                    f.write("\n")
                self.logs = []

    def close(self):
        if not self.enabled:
            return
        with open(self.log_file, "a") as f:
            for log in self.logs:
                json.dump(log, f)
                f.write("\n")
            self.logs = []
