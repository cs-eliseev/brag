from dataclasses import dataclass, field
from time import time
from typing import Dict, Optional
from rag.utils.logger import logger


@dataclass
class Metrics:
    start_time: float = field(default_factory=time)
    end_time: float = field(default=0.0)
    duration: float = field(default=0.0)
    success: bool = field(default=True)
    error: Optional[str] = field(default=None)
    additional_data: Dict = field(default_factory=dict)


class MetricsCollector:
    def __init__(self):
        self.metrics = {}

    def start_operation(self, operation_name: str) -> None:
        self.metrics[operation_name] = Metrics()
        logger().info(f"Starting operation: {operation_name}")

    def end_operation(self, operation_name: str, success: bool = True, error: str = None, **kwargs) -> None:
        if operation_name not in self.metrics:
            logger().warning(f"Operation {operation_name} was not started")
            return

        metric = self.metrics[operation_name]
        metric.end_time = time()
        metric.duration = metric.end_time - metric.start_time
        metric.success = success
        metric.error = error
        metric.additional_data.update(kwargs)

        log_data = {
            "operation": operation_name,
            "duration": f"{metric.duration:.3f}s",
            "success": success,
            **kwargs
        }

        if success:
            logger().info("Operation completed", **log_data)
        else:
            logger().error(f"Operation failed: {error}", **log_data)

    def get_metrics(self, operation_name: str) -> Optional[Metrics]:
        return self.metrics.get(operation_name)

    def clear_metrics(self) -> None:
        self.metrics.clear()