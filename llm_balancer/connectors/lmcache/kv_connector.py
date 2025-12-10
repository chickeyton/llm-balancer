# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import asyncio
from threading import Thread
from typing import Set, List, Dict, Tuple, Optional

from lmcache.v1.cache_controller import controller_manager

from ...balancer.connector import KvConnector


class LMCacheKvConnector(KvConnector):

    def __init__(self,
                 controller_pull_port: int,
                 controller_reply_port: int,
                 is_cache_shared: bool):
        pull_adr = f"0.0.0.0:{controller_pull_port}"
        reply_adr = f"0.0.0.0:{controller_reply_port}"
        self._kv_manager = controller_manager.LMCacheControllerManager(
            controller_urls={"pull": pull_adr,
                             "reply": reply_adr},
            health_check_interval=1,
            lmcache_worker_timeout=1
        )
        self._is_cache_shared = is_cache_shared
        self._thread = None
        self._loop = None

    @property
    def is_cache_shared(self) -> bool:
        return self._is_cache_shared

    def query_hit_len(self, tokens: List[int], instance_ids: Optional[Set[str]] = None) -> Dict[str, int]:
        #print(f"LMCacheKvConnector query_hit_len 1")
        if not self._thread:
            raise RuntimeError("LMCacheKvConnector.start() is not yet called")
        #print(f"LMCacheKvConnector query_hit_len 2")
        kv = self._kv_manager.kv_controller
        #print(f"------------------------ len(kv.kv_pool) : {len(kv.kv_pool)}")
        layout_info: Dict[str, Tuple[str, int]] = {}
        last_end = -1
        for start, end, key in kv.token_database.process_tokens(
            tokens, make_key=False
        ):
            matched_pool = kv.kv_pool.get(key, None)
            if matched_pool is None:
                break
            for instance in matched_pool:
                if instance_ids and instance.instance_id not in instance_ids:
                    continue
                matched_instance = instance.instance_id
                matched_location = instance.location
                cache_info = layout_info.get(matched_instance)
                if last_end == -1:
                    if cache_info is None or cache_info[0] != "LocalCPUBackend":
                        layout_info[matched_instance] = (matched_location, end)
                else:
                    if cache_info is not None:
                        if last_end == cache_info[1]:
                            layout_info[matched_instance] = (matched_location, end)
                        elif (
                            end == cache_info[1]
                            and cache_info[0] != "LocalCPUBackend"
                        ):
                            layout_info[matched_instance] = (matched_location, end)
            last_end = end
        #print(f"LMCacheKvConnector query_hit_len 2")
        result = dict(zip(layout_info.keys(), [v[1] for v in layout_info.values()]))
        if instance_ids:
            #print(f"LMCacheKvConnector query_hit_len 3")
            for instance_id in instance_ids:
                if instance_id not in result:
                    result[instance_id] = 0
        #print(f"LMCacheKvConnector query_hit_len 4")
        return result

    def start(self):
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._loop.run_forever)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._kv_manager.start_all(), self._loop)
