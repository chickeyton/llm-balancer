from llm_balancer.balancer.dynamic_pd2 import DynamicPd, SloConfig, StatsConfig, PdEndpointInfo
import numpy as np


dynamic_pd = DynamicPd(slo_config=SloConfig(pass_ttft=1.0,
                                            pass_tpot=0.25,
                                            excel_ttft=-1,
                                            excel_tpot=-1,
                                            p_quantile=0.99,),
                        stats_config=StatsConfig())

new_reqs = 3000
ttfts = np.random.uniform(10.0, 20.0, new_reqs)
tpots = np.random.uniform(0.01, 0.02, new_reqs)
for ttft, tpot in zip(ttfts, tpots):
    dynamic_pd.on_request_finished(ttft, tpot)

endpoints = [
PdEndpointInfo(is_prefill=True,
               is_switchable=True,
               queue_length=10),
PdEndpointInfo(is_prefill=False,
               is_switchable=True,
               queue_length=100),
PdEndpointInfo(is_prefill=False,
               is_switchable=True,
               queue_length=10),
PdEndpointInfo(is_prefill=False,
               is_switchable=True,
               queue_length=100),
]

realloc_advice = dynamic_pd.advise_realloc(endpoints)
elastic_advice = dynamic_pd.advise_elastic(endpoints)

print("==================== Reallocate Advice ====================")
if realloc_advice:

    print(f"switch endpoints: {realloc_advice.switch_endpoints}")
    if endpoints[realloc_advice.switch_endpoints[0]].is_prefill:
        print(f"to DECODE")
    else:
        print(f"to PREFILL")
else:
    print("NO Advice")

print("==================== Elastic Advice ====================")
if elastic_advice:
    print(f"drop prefills: {elastic_advice.drop_prefills}")
    print(f"drop decodes: {elastic_advice.drop_decodes}")
    print(f"add prefills: {elastic_advice.num_add_prefills}")
    print(f"add decodes: {elastic_advice.num_add_decodes}")
    print(f"new P/D : {elastic_advice.new_total_prefills}/{elastic_advice.new_total_decodes}")
else:
    print("NO Advice")
