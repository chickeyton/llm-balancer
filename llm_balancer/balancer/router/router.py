# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from typing import List, Tuple

from llm_balancer.balancer.common import Stage
from llm_balancer.balancer.endpoint import Endpoint
from llm_balancer.balancer.task import Task
from llm_balancer.balancer.task_route import TaskRoute, EncodeRoute, PrefillRoute, DecodeRoute, PrefillThenDecodeRoute


class Router:

    def __init__(self):
        self._balancer: "Balancer" = None

    @property
    def for_stages(self) -> Tuple[Stage, ...]:
        raise NotImplementedError

    def route(self, task: Task, endpoints: List[Endpoint]) -> TaskRoute:
        raise NotImplementedError

    def on_registered(self, balancer: "Balancer"):
        self._balancer = balancer

    @staticmethod
    def _create_nonworkload_route(task: Task, endpoint: Endpoint) -> TaskRoute:
        if task.stage == Stage.ENCODE:
            return EncodeRoute(request_id=task.request_id,
                               endpoint=endpoint,
                               workload=-1)
        if task.stage == Stage.PREFILL:
            return PrefillRoute(request_id=task.request_id,
                                endpoint=endpoint,
                                workload=-1,
                                num_prompt_tokens=len(task.prompt_tokens),
                                num_cached_tokens=-1)
        if task.stage == Stage.DECODE:
            return DecodeRoute(request_id=task.request_id,
                               endpoint=endpoint,
                               workload=-1,
                               prefill_route=task.prefill_route,
                               predicted_decode_len=task.predicted_decode_len,
                               len_extend_rate=task.len_extend_rate)
        if task.stage == Stage.PREFILL_THEN_DECODE:
            return PrefillThenDecodeRoute(request_id=task.request_id,
                                          endpoint=endpoint,
                                          workload=-1,
                                          num_prompt_tokens=len(task.prompt_tokens),
                                          num_cached_tokens=-1,
                                          prefill_workload=-1,
                                          predicted_decode_len=task.predicted_decode_len,
                                          len_extend_rate=task.len_extend_rate)
        raise ValueError(f"Unsupported stage:{task.stage}")

    @staticmethod
    def _route_by_queue_len(endpoints: List[Endpoint]) -> int:
        min_queue_len = -1
        min_queue_ep_i = -1
        for i, endpoint in enumerate(endpoints):
            if min_queue_ep_i == -1 or endpoint.queue_length() < min_queue_len:
                min_queue_len = endpoint.queue_length()
                min_queue_ep_i = i
        if min_queue_ep_i == -1:
            raise RuntimeError("Found no Endpoint")
        return min_queue_ep_i
