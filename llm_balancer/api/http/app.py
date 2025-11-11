import argparse
import json
import uvicorn

from fastapi import FastAPI
from transformers import AutoTokenizer

from llm_balancer.api.http.api import api_router
from llm_balancer.api.http.config import parse_app_config, parse_endpoint_configs
from llm_balancer.api.http.pipeline.p_d import P_D_Pipline
from llm_balancer.balancer import Balancer, StaticEndpointTracker
from llm_balancer.balancer.router import DecodeRouter, PrefillRouter, KvawareRouter, RoundRobinRouter, RandomRouter, \
    QueueLenRouter
from llm_balancer.connectors.lmcache import LMCacheKvConnector
from llm_balancer.connectors.vllm.endpoint import VllmEndpont

app = FastAPI()
app.include_router(api_router)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-h", "--host", argument_default="0.0.0.0")
    parser.add_argument("-p", "--port")
    parser.add_argument("-c", "--config", argument_default="config.json")
    parser.add_argument("-e", "--endpoints", argument_default="endpoints.json")
    return parser.parse_args()


def create_routers(router_configs):
    endpoints = {}
    for stage, config in router_configs.items():
        if config.name == "PREFILL":
            endpoints[stage] = PrefillRouter(config.len_extend_rate)
        elif config.name == "DECODE":
            endpoints[stage] = DecodeRouter(config.len_extend_rate)
        elif config.name == "KVAWARE":
            endpoints[stage] = KvawareRouter()
        elif config.name == "ROUND_ROBIN":
            endpoints[stage] = RoundRobinRouter()
        elif config.name == "RANDOM":
            endpoints[stage] = RandomRouter()
        elif config.name == "QUEUE_LEN":
            endpoints[stage] = QueueLenRouter()
        else:
            raise ValueError(f"Unsupported Router type:{config.type}")
    return endpoints


def main():
    args = parse_args()
    with open(args.config, 'r') as file:
        app_config = parse_app_config(json.load(file))
    with open(args.endpoints, 'r') as file:
        endpoint_configs = parse_endpoint_configs(json.load(file))

    kv_connector = LMCacheKvConnector(app_config.lmcache.ctl_mgr_port,
                                      app_config.lmcache.is_p2p_enabled)
    tracker = StaticEndpointTracker([VllmEndpont(c) for c in endpoint_configs])
    routers = create_routers(app_config.routers)
    balancer = Balancer(app_config.balancer, tracker, routers, kv_connector)
    tracker.start()
    kv_connector.start()

    tokenizer = AutoTokenizer.from_pretrained(app_config.tokenizer)
    app.state.pipeline = P_D_Pipline(tokenizer, balancer)
    uvicorn.run(app, host=args.host, port=int(args.port))
