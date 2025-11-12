from threading import Thread, Lock
from typing import Dict, Set, List, Optional
import zmq
from msgspec.msgpack import Decoder
from vllm.v1.codre.kv_cache_utils import BlockHash
from vllm.distributed.kv_events import BlockStored, BlockRemoved, AllBlocksCleared, EventBatch

from llm_balancer.balancer import EndpointTracker, EndpointTrackerListener, Endpoint
from llm_balancer.api.http.config import VllmEndpointConfig


# copy from vllm as vllm_instance_id is added by the patch
class KVEventBatch(EventBatch):
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]
    vllm_instance_id: str


class KVCacheTracker(Thread, EndpointTrackerListener):

    class _Subscription:
        def __init__(self, endpoint_id, event_endpoint):
            self.endpoint_id = endpoint_id
            self.is_connected = False
            self.event_endpoint = event_endpoint
            self.block_hashes: Set[BlockHash] = set()
            self.is_endpoint_up = True

    def __init__(self, tracker: EndpointTracker, preserve_down_records: bool = False):
        super(Thread, self).__init__()
        self._preserve_down_records = preserve_down_records
        self._lock = Lock()
        self._subscriptions: Dict[str, KVCacheTracker._Subscription] = {}
        self._tracker = tracker
        self._tracker.add_listener(self)
        with self._lock:
            for endpoint in self._tracker.get_up_endpoints():
                if endpoint.config.kv_event_endpoint:
                    subscription = self._Subscription(endpoint.id,
                                                      endpoint.config.kv_event_endpoint)
                    self._subscriptions[endpoint.id] = subscription

    def query_hit_len(self,
                      block_size: int,
                      prefix_block_hashes: List[BlockHash],
                      endpoint_ids: Optional[Set[str]] = None) -> Dict[str, int]:
        result = {}
        if endpoint_ids:
            with self._lock:
                for endpoint_id in endpoint_ids:
                    subscription = self._subscriptions.get(endpoint_id)
                    if subscription:
                        result[endpoint_id] = \
                            self._find_num_hit_blocks(subscription.block_hashes,
                                                      prefix_block_hashes) * block_size
                    else:
                        result[endpoint_id] = 0
        else:
            with self._lock:
                for subscription in self._subscriptions.values():
                    result[subscription.endpoint_id] = \
                        self._find_num_hit_blocks(subscription.block_hashes,
                                                  prefix_block_hashes) * block_size
        return result

    def on_endpoints_changed(self, new_ups: List[Endpoint], new_downs: List[Endpoint]):
        with self._lock:
            for new_up in new_ups:
                if isinstance(new_up.config, VllmEndpointConfig):
                    raise ValueError("config is not VllmEndpointConfig")
                subscription = self._subscriptions.get(new_up.id)
                if subscription is None:
                    if new_up.config.kv_event_endpoint:
                        subscription = self._Subscription(new_up.id,
                                                          new_up.config.kv_event_endpoint)
                        self._subscriptions[new_up.id] = subscription
                else:
                    subscription.is_endpoint_up = True

            for new_down in new_downs:
                subscription = self._subscriptions.get(new_down.id)
                if subscription is not None:
                    subscription.is_endpoint_up = False

    def run(self):
        zmq_ctx = zmq.Context()
        zmq_sub = zmq_ctx.socket(zmq.SUB)
        zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "kv-events")
        decoder = Decoder(type=KVEventBatch)
        while True:
            with self._lock:
                subscriptions = self._subscriptions.copy()
            remove_list = []
            for subscription in subscriptions.values():
                if subscription.is_endpoint_up:
                    if not subscription.is_connected:
                        zmq_sub.connect(subscription.event_endpoint)
                        subscription.is_connected = True
                else:
                    if subscription.is_connected:
                        zmq_sub.disconnect(subscription.event_endpoint)
                        subscription.is_connected = False
                    if not self._preserve_down_records:
                        remove_list.append(subscription.endpoint_id)
            if remove_list:
                with self._lock:
                    for endpoint_id in remove_list:
                        self._subscriptions.pop(endpoint_id)

            try:
                # TODO: use blocking and catch timeout
                _, seq_bytes, payload = zmq_sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue

            event_batch = decoder.decode(payload)
            with self._lock:
                # TODO: event_batch.vllm_instance_id needs to be added by vllm
                subscription = self._subscriptions.get(event_batch.vllm_instance_id)
                if not subscription:
                    continue
                for event in event_batch.events:
                    if isinstance(event, BlockStored):
                        for block_hash in event.block_hashes:
                            subscription.block_hashes.add(block_hash)
                    elif isinstance(event, BlockRemoved):
                        for block_hash in event.block_hashes:
                            subscription.block_hashes.discard(block_hash)
                    elif isinstance(event, AllBlocksCleared):
                        subscription.block_hashes.clear()

    @staticmethod
    def _find_num_hit_blocks(cache_block_hashes, prefix_block_hashes):
        for i, prefix_block_hash in enumerate(prefix_block_hashes):
            if prefix_block_hash not in cache_block_hashes:
                return i
        return 0
