# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from typing import Tuple, List

from llm_balancer.balancer.common import Stage
from llm_balancer.balancer.endpoint import Endpoint
from llm_balancer.balancer.router.router import Router
from llm_balancer.balancer.task import Task
from llm_balancer.balancer.task_route import TaskRoute


class QueueLenRouter(Router):

    def __init__(self):
        super().__init__()

    @property
    def for_stages(self) -> Tuple[Stage, ...]:
        return Stage.ENCODE, Stage.PREFILL, Stage.DECODE, Stage.PREFILL_THEN_DECODE

    def route(self, task: Task, endpoints: List[Endpoint]) -> TaskRoute:
        endpoint_index = self._route_by_queue_len(endpoints)
        return self._create_nonworkload_route(task, endpoints[endpoint_index])
