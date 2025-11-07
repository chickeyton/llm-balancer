from typing import Tuple, List

from llm_balancer.balancer import Stage, Task, Endpoint, TaskRoute, DecodeRoute
from llm_balancer.balancer.router import Router
from llm_balancer.balancer.workload import decode_atten_workload


class DecodeRouter(Router):

    def __init__(self, len_extend_rate: float = 0.2):
        super().__init__()
        self._len_extend_rate = len_extend_rate

    @property
    def for_stages(self) -> Tuple[Stage, ...]:
        raise (Stage.DECODE,)

    def route(self, task: Task, endpoints: List[Endpoint]) -> TaskRoute:
        prefill_len = task.prefill_len
        predicted_decode_len = task.predicted_decode_len
        if prefill_len <= 0:
            raise ValueError("Invalid prefill_len")
        if predicted_decode_len <= 0:
            raise ValueError("Invalid predicted_decode_len")
        workload = decode_atten_workload(prefill_len + predicted_decode_len,
                                         prefill_len)
        try:
            endpoint = self._find_best_endpoint(endpoints)
            return DecodeRoute(request_id=task.request_id,
                               endpoint=endpoint,
                               workload=workload,
                               predicted_decode_len=predicted_decode_len,
                               len_extend_rate=self._len_extend_rate,
                               prefill_route=task.prefill_route)
        except ValueError:
            pass
        idx = self._route_by_queue_len(endpoints)
        return DecodeRoute(request_id=task.request_id,
                           endpoint=endpoints[idx],
                           workload=workload,
                           predicted_decode_len=predicted_decode_len,
                           len_extend_rate=self._len_extend_rate,
                           prefill_route=task.prefill_route)

    @staticmethod
    def _find_best_endpoint(endpoints):
        min_workload = -1
        best = None
        for endpoint in endpoints:
            workload = endpoint.queue_workload()
            if best is None or workload < min_workload:
                best = endpoint
                min_workload = workload
        if best is None:
            raise RuntimeError("No best endpoint found")
        return best
