import os
import csv
from typing import Dict, Any, List

class Logger:
    """
    Logger class to save training metrics to a CSV file.
    """
    def __init__(self, output_dir: str, output_fname: str = "metrics.csv") -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, output_fname)
        self.file = open(self.filepath, 'w', newline='')
        self.writer = None
        self.keys: List[str] = []

    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Logs a dictionary of metrics as a row in the CSV.
        """
        if self.writer is None:
            self.keys = list(metrics.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.keys)
            self.writer.writeheader()
        
        # Ensure we only write keys that are in the header
        # This handles cases where metrics might change order or keys (though unlikely in this loop)
        row = {k: metrics.get(k, 0) for k in self.keys}
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.close()