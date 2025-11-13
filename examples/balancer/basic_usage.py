import time
from dataclasses import dataclass
from threading import Thread, Lock
from typing import List, Tuple, Optional

from llm_balancer.balancer import Balancer, EndpointTracker, EndpointConfig, Endpoint, Stage, ViTEncodeTask, \
    PrefillTask, EncodeHandle, PrefillHandle, DecodeHandle, DecodeTask
from llm_balancer.balancer.router import PrefillRouter, RoundRobinRouter, QueueLenRouter
from llm_balancer.connectors.lmcache import LMCacheKvConnector


@dataclass
class ZmqEndpointConfig(EndpointConfig):
    # put extra settings here
    zmq_addr: str = ""

class ZmqEndpoint(Endpoint):
    def __init__(self, config: ZmqEndpointConfig):
        super().__init__(config)
        # put extra states here



class RedisEndpointTracker(EndpointTracker):
    def __init__(self, host: str, port: int):
        super().__init__()
        self._host = host
        self._port = port
        self._lock = Lock()
        self._is_running = True
        self._endpoints:list[ZmqEndpoint] = []
        self._thread = None

    def get_up_endpoints(self, stages: Optional[Tuple[Stage, ...], List[Stage]] = None) -> List[Endpoint]:
        with self._lock:
            if stages:
                return [ep for ep in self._endpoints if ep.stage in stages]
            return self._endpoints.copy()

    def start(self):
        self._thread = Thread(target=self._update_loop)
        self._thread.start()

    def stop(self):
        with self._lock:
            self._is_running = False

    def _parse_endpoint_list(self, endpoint_list_str) -> List[ZmqEndpointConfig]:
        # TODO
        # configs.append(
        #     ZmqEndpointConfig(
        #         endpoint_id=endpoint_id,
        #         stage=stage,
        #         cache_instance_id=cache_instance_id,
        #         zmq_addr=zmq_addr
        #     )
        # )
        # return configs
        pass

    def _update_endpoints(self, updated_configs: List[ZmqEndpointConfig]):
        with self._lock:
            # NEVER re-create endpoint object if it is still up

            # filter out newly down endpoints, keep the exiting up endpoints
            up_ids = set([c.endpoint_id for c in updated_configs])
            old_endpoints = self._endpoints
            self._endpoints = [ep for ep in old_endpoints if ep.id in up_ids]

            # create newly up endpoints
            old_up_ids = set([ep.id for ep in old_endpoints])
            new_ups: List[Endpoint] = [ZmqEndpoint(c) for c in updated_configs if c.endpoint_id not in old_up_ids]
            self._endpoints.extend(new_ups)

            # gather newly down endpoints
            new_downs: List[Endpoint] = [ep for ep in old_endpoints if ep.id not in up_ids]

        if new_ups or new_downs:
            # unlock then notify balancer
            self.on_endpoints_changed(new_ups, new_downs)

    def _update_loop(self):
        redis = redis.Redis(host=self._host, port=self._port, decode_responses=True)
        while True:
            with self._lock:
                if not self._is_running:
                    break
            endpoint_list_str = redis.get("endpoint_list.json")
            updated_configs = self._parse_endpoint_list(endpoint_list_str)
            self._update_endpoints(updated_configs)
            time.sleep(5)


def to_encode_task(request) -> ViTEncodeTask:
    # TODO
    # return VitEncodeTask(
    #     request_id=request_id,
    #     num_patches=num_patches
    # )
    pass


def send_encode(request, zmq_addr):
    # TODO
    pass


def to_prefill_task(request, encode_handle):
    # TODO
    # return PrefillTask(
    #     request_id=encode_handle.request_id,
    #     prompt_tokens=prompt_tokens
    # )
    pass


def send_prefill(request, zmq_addr, prefill_handle):
    # TODO
    # for token_chunk in respond_stream:
    #    prefill_handle.on_respond(len(token_chunk))
    pass


def to_decode_task(request, prefill_handle):y

    # TODO
    # return DecodeTask(
    #     request_id=prefill_handle.request_id,
    #     prefill_len=prefill_handle.route.num_prompt_tokens,
    #     predicted_decode_len=predicted_decode_len,
    #     prefill_route=prefill_handle.route
    # )
    pass


def send_decode(request, decode_handle):
    # TODO
    # for token_chunk in respond_stream:
    #    decode_handle.on_respond(len(token_chunk))
    pass


def initialize_balancer():
    tracker = RedisEndpointTracker(host="localhost", port=1234)
    routers = {Stage.ENCODE: RoundRobinRouter(),
               Stage.PREFILL: PrefillRouter(),
               Stage.DECODE: QueueLenRouter()}
    kv_connector = LMCacheKvConnector(ctl_mgr_port=9876, is_p2p_enabled=True)
    balancer = Balancer(tracker=tracker,
                        routers=routers,
                        kv_connector=kv_connector)

    # ALWAYS start tracker and connector update loop(if any) AFTER creating the balancer
    tracker.start()
    kv_connector.start()

    return balancer


def handle_request(balancer, request):
    # Balacner & task handles can only be used in a single thread for now

    # ENCODE
    encode_task: ViTEncodeTask = to_encode_task(request)
    encode_handle: EncodeHandle = balancer.route(encode_task).on_submit()
    send_encode(request, encode_handle.endpoint.config.zmq_addr)
    encode_handle.on_finished()

    # PREFILL
    prefill_task: PrefillTask = to_prefill_task(request, encode_handle)
    prefill_handle: PrefillHandle = balancer.route(prefill_task).on_submit()
    send_prefill(request, prefill_handle.endpoint.config.zmq_addr, prefill_handle)
    prefill_handle.on_finished()

    # DECODE
    decode_task: DecodeTask = to_decode_task(request, prefill_handle)
    decode_handle: DecodeHandle = balancer.route(decode_task).on_submit()
    send_decode(request, prefill_handle.endpoint.config.zmq_addr, decode_handle)
    decode_handle.on_finished()
