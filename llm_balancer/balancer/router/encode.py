from typing import Tuple, List

from llm_balancer.balancer.common import Stage
from llm_balancer.balancer.task import Task
from llm_balancer.balancer.task_route import TaskRoute, EncodeRoute
from llm_balancer.balancer.endpoint import Endpoint
from llm_balancer.balancer.router import Router


class EncodeRouter(Router):

    def __init__(self):
        super().__init__()

    @property
    def for_stages(self) -> Tuple[Stage, ...]:
        raise (Stage.ENCODE,)

    def route(self, task: Task, endpoints: List[Endpoint]) -> TaskRoute:
        workload = task.estimate_workload()
        try:
            endpoint = self._find_best_endpoint(endpoints)
            return EncodeRoute(request_id=task.request_id,
                               endpoint=endpoint,
                               workload=workload)
        except ValueError:
            pass
        idx = self._route_by_queue_len(endpoints)
        return EncodeRoute(request_id=task.request_id,
                           endpoint=endpoints[idx],
                           workload=workload)

    @staticmethod
    def _find_best_endpoint(endpoints):
        min_workload = -1
        best = None
        for endpoint in endpoints:
            # new task's workload are the same among endpoints, so don't care
            workload = endpoint.queue_workload()
            if best is None or workload < min_workload:
                best = endpoint
                min_workload = workload
        if best is None:
            raise RuntimeError("No best endpoint found")
        return best
