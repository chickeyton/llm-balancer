from dataclasses import dataclass
from typing import List, Dict

from llm_balancer.balancer import BalancerConfig, Stage
from llm_balancer.connectors.vllm.endpoint import VllmEndpointConfig


@dataclass
class LMCacheConfig:
    ctrl_mgr_port: int = -1
    is_p2p_enabled: bool = False


@dataclass
class RouterConfig:
    name: str = ""
    len_extend_rate: float = 0.2


@dataclass
class AppConfig:
    http_port: int = -1
    tokenizer: str = ""
    balancer: BalancerConfig = BalancerConfig()
    routers: Dict[Stage, RouterConfig] = None
    lmcache: LMCacheConfig = None


def parse_app_config(json_dict) -> AppConfig:
    config = AppConfig()
    config.http_port = int(json_dict.get("http_port", config.http_port))
    config.tokenizer = str(json_dict.get("tokenizer", config.tokenizer))
    config.balancer = BalancerConfig()

    obj = json_dict.get("service_level_obj")
    if obj:
        if config.balancer.service_level_obj is None:
            config.balancer.service_level_obj = BalancerConfig.ServiceLevelObj()
        config.balancer.service_level_obj.ttft = \
            float(obj.get("ttft", config.balancer.service_level_obj.ttft))
        config.balancer.service_level_obj.tpot = \
            float(obj.get("tpot", config.balancer.service_level_obj.tpot))
        config.balancer.service_level_obj.cut_point = \
            float(obj.get("cut_point", config.balancer.service_level_obj.cut_point))

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
        config.lmcache.ctl_mgr_port = int(lmcache_obj.get("ctl_mgr_port"))
        config.lmcache.is_p2p_enabled = bool(lmcache_obj.get("is_p2p_enabled"))

    return config


def parse_endpoint_configs(json_list) -> List[VllmEndpointConfig]:
    config_list = []
    for obj in json_list:
        config = VllmEndpointConfig()
        config.endpoint_id = str(obj.get("endpoint_id"))  # i.e. VLLM_INSTANCE_ID
        config.cache_instance_id = str(obj.get("cache_instance_id", config.endpoint_id))
        config.base_url = str(obj.get("base_url"))
        config.kv_event_endpoint = str(obj.get("kv_event_endpoint"))
        stage_str = obj.get("stage")
        if stage_str == "PREFILL/DECODE":
            config.is_dynamic_pd = True
            config.stage = Stage.PREFILL
        elif stage_str == "DECODE/PREFILL":
            config.is_dynamic_pd = True
            config.stage = Stage.DECODE
        else:
            config.is_dynamic_pd = False
            config.stage = Stage[stage_str]
        config.cache_instance_id = \
            str(obj.get("cache_instance_id", config.cache_instance_id))
        config_list.append(config)
    return config_list
