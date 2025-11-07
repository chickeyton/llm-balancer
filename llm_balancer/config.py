from dataclasses import dataclass
from typing import List

from llm_balancer.balancer import BalancerConfig, EndpointConfig, Stage


@dataclass
class AppConfig:
    http_port: int
    tokenizer: str
    balancer: BalancerConfig


@dataclass
class VllmEndpointConfig(EndpointConfig):
    base_url: str = ""
    kv_event_endpoint: str = ""


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
        config.balancer.service_level_obj.quantile = \
            float(obj.get("quantile", config.balancer.service_level_obj.quantile))

    obj = json_dict.get("dynamic_pd")
    if obj:
        if config.balancer.dynamic_pd is None:
            config.balancer.dynamic_pd = BalancerConfig.DynamicPd()
        config.balancer.dynamic_pd.update_on_requests = \
            int(obj.get("update_on_requests", config.balancer.dynamic_pd.update_on_requests))
        config.balancer.dynamic_pd.min_update_time = \
            float(obj.get("min_update_time", config.balancer.dynamic_pd.min_update_time))

    return config


def parse_endpoint_configs(json_list) -> List[VllmEndpointConfig]:
    config_list = []
    for obj in json_list:
        config = VllmEndpointConfig()
        config.endpoint_id = str(obj.get("endpoint_id")) # i.e. VLLM_INSTANCE_ID
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
