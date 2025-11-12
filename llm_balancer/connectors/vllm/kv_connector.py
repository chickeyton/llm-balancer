from typing import Optional, Set, List, Dict

from vllm.v1.codre.kv_cache_utils import BlockHash

from llm_balancer.balancer.connector import KvConnector
from llm_balancer.balancer import EndpointTracker
from .kv_cache_tracker import KVCacheTracker


class VllmKvConnector(KvConnector):
    def __init__(self, tracker: EndpointTracker, block_size: int, is_p2p_enabled: bool):
        self._cache_tracker = KVCacheTracker(tracker)
        self._block_size = block_size
        self._is_p2p_enabled = is_p2p_enabled

    @property
    def is_p2p_enabled(self) -> bool:
        return self._is_p2p_enabled

    def query_hit_len(self, tokens: List[int], instance_ids: Optional[Set[str]] = None) -> Dict[str, int]:
        block_hashes = self._hash(tokens)
        return self._cache_tracker.query_hit_len(self._block_size, block_hashes, instance_ids)

    def start(self):
        self._cache_tracker.start()

    def stop(self):
        self._cache_tracker.stop()

    def join(self):
        self._cache_tracker.join()

    def _hash(self, tokens) -> List[BlockHash]:
        # TODO: by using vllm hasher
        pass
