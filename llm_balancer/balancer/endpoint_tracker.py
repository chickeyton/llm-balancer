# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from typing import List, Optional, Tuple

from llm_balancer.balancer import Stage
from llm_balancer.balancer.endpoint import Endpoint


class EndpointTrackerListener:
    def on_endpoints_changed(self, new_ups: List[Endpoint], new_downs: List[Endpoint]):
        pass


class EndpointTracker:
    def __init__(self):
        self._listener: Optional[EndpointTrackerListener] = None

    def get_up_endpoints(self, stages: Optional[Tuple[Stage, ...], List[Stage]] = None) -> List[Endpoint]:
        raise NotImplementedError

    def set_listener(self, listener: Optional[EndpointTrackerListener]):
        self._listener = listener

    def on_endpoints_changed(self, new_ups: List[Endpoint], new_downs: List[Endpoint]):
        """To be called by the sub-classes."""
        if self._listener:
            self._listener.on_endpoints_changed(new_ups, new_downs)


class StaticEndpointTracker(EndpointTracker):
    def __init__(self, endpoints: List[Endpoint]):
        super().__init__()
        self._endpoints = endpoints

    def get_up_endpoints(self, stages: Optional[Tuple[Stage, ...], List[Stage]] = None) -> List[Endpoint]:
        if stages:
            return [ep for ep in self._endpoints if ep.stage in stages]
        return self._endpoints.copy()
