# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project
from dataclasses import dataclass
from typing import List, Dict, Optional

from .logger import Logger, NullLogger
from .router.batch_routing import BatchRouteOptimizer, GreedyBatchRoute
from .task_handle import TaskHandle, DecodeHandle
from .common import Stage, ServiceLevelObj
from .connector.kv_connector import KvConnector
from .dynamic_pd import DynamicPd
from .endpoint import Endpoint, EndpointListener
from .endpoint_tracker import EndpointTrackerListener, EndpointTracker
from .router.router import Router
from .task import Task
from .task_route import TaskRoute


class BalancerConfig:

    @dataclass
    class DynamicPd:
        update_on_requests: int = 100
        min_update_time: float = 10

    def __init__(self):
        self.service_level_obj: ServiceLevelObj = None
        self.dynamic_pd: DynamicPd = BalancerConfig.DynamicPd()


class Balancer(EndpointTrackerListener, EndpointListener):

    def __init__(self,
                 config: BalancerConfig,
                 tracker: EndpointTracker,
                 routers: Dict[Stage, Router],
                 kv_connector: Optional[KvConnector] = None,
                 batch_route_optimizer: Optional[BatchRouteOptimizer] = None,
                 logger: Logger = None,
                 dynamic_pd = None):
        self.config = config
        self._tracker = tracker
        self._tracker.add_listener(self)
        self._kv_connector = kv_connector
        self._batch_route_optimizer = \
            GreedyBatchRoute() if batch_route_optimizer is None else batch_route_optimizer
        self._routers = routers
        self._dynamic_pd = dynamic_pd
        self._logger = NullLogger() if logger is None else logger

        for stage, router in self._routers.items():
            if stage not in router.for_stages:
                raise ValueError(f"{router.__class__.__name__} is not for stage:{stage}")
            router.on_registered(self)

        for endpoint in self._tracker.get_up_endpoints():
            endpoint.set_listener(self)

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def kv_connector(self) -> Optional[KvConnector]:
        return self._kv_connector

    @property
    def dynamic_pd(self):
        return self._dynamic_pd

    @property
    def batch_route_optimizer(self) -> BatchRouteOptimizer:
        return self._batch_route_optimizer

    def get_candidates(self, task) -> List[Endpoint]:
        return self._tracker.get_up_endpoints(stages=(task.stage,))

    def get_up_endpoints(self) -> List[Endpoint]:
        return self._tracker.get_up_endpoints()

    def route(self, task: Task, candidates: List[Endpoint] = None) -> TaskRoute:
        router = self._routers.get(task.stage)
        if router is None:
            raise ValueError(f"{task.stage} task is not supported by routers")
        if not candidates:
            candidates = self.get_candidates(task)
        if not candidates:
            raise RuntimeError(f"No candidate for {task.stage} task")
        return router.route(task, candidates)

    def batch_route(self, tasks: List[Task], candidates: List[Endpoint] = None) -> List[TaskRoute]:
        if len(tasks) == 1:
            return [self.route(tasks[0], candidates)]
        for task in tasks:
            if task.stage != tasks[0].stage:
                raise ValueError(f"Not all tasks of the same stage")
        router = self._routers.get(tasks[0].stage)
        if router is None:
            raise ValueError(f"{tasks[0].stage} task is not supported by routers")
        if not candidates:
            candidates = self.get_candidates(tasks[0])
        if not candidates:
            raise RuntimeError(f"No candidate for {tasks[0].stage} task")
        return router.batch_route(tasks, candidates)

    def on_endpoints_changed(self, new_ups: List[Endpoint], new_downs: List[Endpoint]):
        for endpoint in new_downs:
            endpoint.set_listener(None)
        for endpoint in new_ups:
            endpoint.set_listener(self)

    def on_task_ended(self, handle: TaskHandle):
        if isinstance(handle, DecodeHandle):
            self._dynamic_pd.on_request_finished(handle.request_meta.ttft, handle.request_meta.tpot)
        self._logger.task_ended(handle)
