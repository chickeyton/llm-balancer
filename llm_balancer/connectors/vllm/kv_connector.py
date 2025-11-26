from typing import Optional, Set, List, Dict

from vllm.v1.core.kv_cache_utils import BlockHash

from ...balancer.connector import KvConnector
from ...balancer import EndpointTracker
from .kv_cache_tracker import VllmKvCacheTracker


class VllmKvConnector(KvConnector):

    def __init__(self, tracker: EndpointTracker, block_size: int, is_cache_shared: bool):
        self._cache_tracker = VllmKvCacheTracker(tracker)
        self._block_size = block_size
        self._is_cache_shared = is_cache_shared

    @property
    def is_cache_shared(self) -> bool:
        return self._is_cache_shared

    def query_hit_len(self, tokens: List[int], instance_ids: Optional[Set[str]] = None) -> Dict[str, int]:
        block_hashes = self._hash(tokens)
        return self._cache_tracker.query_hit_len(len(tokens), self._block_size, block_hashes, instance_ids)

    def start(self):
        self._cache_tracker.start()

    def die(self):
        self._cache_tracker.die()

    def join(self):
        self._cache_tracker.join()

    def _hash(self, tokens) -> List[BlockHash]:
        # TODO: by using vllm hasher
        pass
