import argparse
import json
import uvicorn

from fastapi import FastAPI
from transformers import AutoTokenizer

from llm_balancer.api.http.api import api_router
from llm_balancer.api.http.config import parse_app_config, parse_endpoint_configs, detect_pipeline
from llm_balancer.balancer import Balancer, StaticEndpointTracker
from llm_balancer.balancer.logger import StatsLogger
from llm_balancer.balancer.router import DecodeRouter, PrefillRouter, KvawareRouter, RoundRobinRouter, RandomRouter, \
    QueueLenRouter
from llm_balancer.balancer.router.encode import EncodeRouter
# from llm_balancer.connectors.lmcache import LMCacheKvConnector
from llm_balancer.connectors.vllm import VllmEndpoint

app = FastAPI()
app.include_router(api_router)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--host", default="0.0.0.0")
    parser.add_argument("-p", "--port", default="8888")
    parser.add_argument("-c", "--config", default="config.json")
    parser.add_argument("-e", "--endpoints", default="endpoints.json")
    return parser.parse_args()


def create_routers(router_configs):
    routers = {}
    for stage, config in router_configs.items():
        if config.name == "encode":
            routers[stage] = EncodeRouter()
        elif config.name == "prefill":
            routers[stage] = PrefillRouter(config.len_extend_rate)
        elif config.name == "decode":
            routers[stage] = DecodeRouter(config.len_extend_rate)
        elif config.name == "kvaware":
            routers[stage] = KvawareRouter()
        elif config.name == "round_robin":
            routers[stage] = RoundRobinRouter()
        elif config.name == "random":
            routers[stage] = RandomRouter()
        elif config.name == "queue_len":
            routers[stage] = QueueLenRouter()
        else:
            raise ValueError(f"Unsupported Router type:{config.type}")
    return routers


def main():
    args = parse_args()
    with open(args.config, 'r') as file:
        app_config = parse_app_config(json.load(file))
    with open(args.endpoints, 'r') as file:
        endpoint_configs = parse_endpoint_configs(json.load(file))

    #kv_connector = LMCacheKvConnector(app_config.lmcache.controller_pull_port,
    #                                  app_config.lmcache.controller_reply_port,
    #                                  app_config.lmcache.is_cache_shared)
    logger = StatsLogger(service_level_obj=app_config.balancer.service_level_obj)
    tracker = StaticEndpointTracker([VllmEndpoint(c) for c in endpoint_configs])
    routers = create_routers(app_config.routers)
    balancer = Balancer(config=app_config.balancer,
                        tracker=tracker,
                        routers=routers,
                        logger=logger,
                        kv_connector=None)
    #kv_connector.start()

    tokenizer = AutoTokenizer.from_pretrained(app_config.tokenizer)
    if app_config.batch_routing is None:
        app.state.pipeline = \
            detect_pipeline(endpoint_configs, is_batched=False)(tokenizer,
                                                                balancer)
    else:
        app.state.pipeline = \
            detect_pipeline(endpoint_configs, is_batched=True)(tokenizer,
                                                               balancer,
                                                               app_config.batch_routing.max_batch_size,
                                                               app_config.batch_routing.max_batch_time)
    print(f"pipeline: {app.state.pipeline.__class__}")

    uvicorn.run(app, host=args.host, port=int(args.port))

    stats = logger.compute_stats()
    print(f"ttft_mean: {stats.ttft_mean}")
    print(f"ttft_quantile: {stats.ttft_quantile}")
    print(f"tpot_mean: {stats.tpot_mean}")
    print(f"tpot_quantile: {stats.tpot_quantile}")
    print(f"e2e_mean: {stats.e2e_mean}")
    print(f"e2e_quantile: {stats.e2e_quantile}")
    print(f"slo_attainment: {stats.slo_attainment}")
    print(f"num_requests: {stats.num_requests}")


if __name__ == "__main__":
    main()
