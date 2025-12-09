from dataclasses import dataclass
from typing import List, Dict, Type

from llm_balancer.balancer import BalancerConfig, Stage
from llm_balancer.connectors.vllm.endpoint import VllmEndpointConfig
from .pipeline import PD_Pipeline, P_D_Pipeline
from .pipeline.p_d import P_D_BatchedPipeline
from .pipeline.pd import PD_BatchedPipeline
from ...balancer.common import ServiceLevelObj


@dataclass
class LMCacheConfig:
    contorller_pull_port: int = -1
    contorller_reply_port: int = -1
    is_cache_shared: bool = False


@dataclass
class RouterConfig:
    name: str = ""
    len_extend_rate: float = 0.2


@dataclass
class BatchRoutingConfig:
    max_batch_size: int = 8
    max_batch_time: float = 0.1


@dataclass
class AppConfig:
    http_port: int = -1
    tokenizer: str = ""
    balancer: BalancerConfig = BalancerConfig()
    routers: Dict[Stage, RouterConfig] = None
    lmcache: LMCacheConfig = None
    batch_routing: BatchRoutingConfig = None


def parse_app_config(json_dict) -> AppConfig:
    config = AppConfig()
    config.http_port = int(json_dict.get("http_port", config.http_port))
    config.tokenizer = str(json_dict.get("tokenizer", config.tokenizer))
    config.balancer = BalancerConfig()

    obj = json_dict.get("service_level_obj")
    if obj:
        if config.balancer.service_level_obj is None:
            config.balancer.service_level_obj = ServiceLevelObj()
        config.balancer.service_level_obj.ttft = \
            float(obj.get("ttft", config.balancer.service_level_obj.ttft))
        config.balancer.service_level_obj.tpot = \
            float(obj.get("tpot", config.balancer.service_level_obj.tpot))
        config.balancer.service_level_obj.p_quantile = \
            float(obj.get("p_quantile", config.balancer.service_level_obj.p_quantile))

    obj = json_dict.get("dynamic_pd")
    if obj:
        if config.balancer.dynamic_pd is None:
            config.balancer.dynamic_pd = BalancerConfig.DynamicPd()
        config.balancer.dynamic_pd.update_on_requests = \
            int(obj.get("update_on_requests", config.balancer.dynamic_pd.update_on_requests))
        config.balancer.dynamic_pd.min_update_time = \
            float(obj.get("min_update_time", config.balancer.dynamic_pd.min_update_time))

    router_objs = json_dict.get("routers")
    if router_objs:
        config.routers = {}
        for stage, obj in router_objs.items():
            if isinstance(obj, str):
                config.routers[Stage[stage]] = RouterConfig(name=obj)
            else:
                config.routers[Stage[stage]] = \
                    RouterConfig(str(obj.get("router")),
                                 float(obj.get("len_extend_rate", 0.2)))

    lmcache_obj = json_dict.get("lmcache")
    if lmcache_obj:
        config.lmcache = LMCacheConfig()
        config.lmcache.controller_pull_port = int(lmcache_obj.get("controller_pull_port"))
        config.lmcache.controller_reply_port = int(lmcache_obj.get("controller_reply_port"))
        config.lmcache.is_cache_shared = bool(lmcache_obj.get("is_cache_shared"))
        print(f"lmcache_obj {config.lmcache.contorller_pull_port}, {config.lmcache.contorller_reply_port}")

    print(f"lLLLLLLLLLLLLLLL {config.lmcache.contorller_pull_port}, {config.lmcache.contorller_reply_port}")



    batch_routing_obj = json_dict.get("batch_routing")
    if batch_routing_obj:
        config.batch_routing = BatchRoutingConfig()
        config.batch_routing.max_batch_size = int(batch_routing_obj.get("max_batch_size"))
        config.batch_routing.max_batch_time = float(batch_routing_obj.get("max_batch_time"))

    return config


def parse_endpoint_configs(json_list) -> List[VllmEndpointConfig]:
    config_list = []
    for obj in json_list:
        config = VllmEndpointConfig()
        config.endpoint_id = str(obj.get("endpoint_id"))  # i.e. VLLM_INSTANCE_ID
        config.cache_instance_id = str(obj.get("cache_instance_id", config.endpoint_id))
        config.base_url = str(obj.get("base_url"))
        config.api_key = str(obj.get("api_key", config.api_key))
        config.kv_event_endpoint = str(obj.get("kv_event_endpoint", config.kv_event_endpoint))
        stage_str = obj.get("stage")
        if stage_str == "PREFILL/":
            config.is_dynamic_pd = True
            config.stage = Stage.PREFILL
        elif stage_str == "DECODE/":
            config.is_dynamic_pd = True
            config.stage = Stage.DECODE
        else:
            config.is_dynamic_pd = False
            config.stage = Stage[stage_str]
        config.cache_instance_id = \
            str(obj.get("cache_instance_id", config.cache_instance_id))
        config_list.append(config)
    return config_list


def detect_pipeline(endpoints: List[VllmEndpointConfig], is_batched: bool) -> Type:
    if not endpoints:
        raise ValueError("No Endpoint")
    stage_counts:Dict[Stage, int] = {}
    for endpoint in endpoints:
        stage_counts[endpoint.stage] = stage_counts.get(endpoint.stage, 0) + 1

    if stage_counts.get(Stage.PREFILL_THEN_DECODE):
        if stage_counts[Stage.PREFILL_THEN_DECODE] != len(endpoints):
            raise ValueError("Not all Endpoints' are PREFILL_THEN_DECODE")
        return PD_BatchedPipeline if is_batched else PD_Pipeline

    if not stage_counts.get(Stage.PREFILL):
        raise ValueError("No PREFILLE Endpoint")

    if not stage_counts.get(Stage.DECODE):
        raise ValueError("No DECODE Endpoint")

    return P_D_BatchedPipeline if is_batched else P_D_Pipeline
