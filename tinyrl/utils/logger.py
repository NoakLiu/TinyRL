"""
Simple logger for TinyRL
"""

import os
import json
import time
from typing import Dict, Any, Optional
import numpy as np


class Logger:
    """Simple logger for training metrics"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = {}
        self.start_time = time.time()
        
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics"""
        timestamp = time.time() - self.start_time
        
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            **metrics
        }
        
        # Store in memory
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Write to file
        log_file = os.path.join(self.log_dir, "metrics.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_stats(self, key: str, window: int = 100) -> Dict[str, float]:
        """Get statistics for a metric"""
        if key not in self.metrics:
            return {}
        
        values = self.metrics[key][-window:]
        return {
            f"{key}_mean": np.mean(values),
            f"{key}_std": np.std(values),
            f"{key}_min": np.min(values),
            f"{key}_max": np.max(values),
        }
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration"""
        config_file = os.path.join(self.log_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def close(self):
        """Close logger"""
        # Save final metrics summary
        summary_file = os.path.join(self.log_dir, "summary.json")
        summary = {}
        
        for key in self.metrics:
            summary.update(self.get_stats(key))
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2) 