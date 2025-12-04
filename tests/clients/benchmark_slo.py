import numpy as np
import time
from multiprocessing import Process, Manager, Value
from openai import OpenAI

base_url = "http://localhost:8888/v1"
api = "/chat/completions"
model = "Qwen/Qwen2-7B"

num_requests = 100
num_workers = 20
fixed_prefix_len = 0
subfix_min_len = 20
subfix_max_len = 3900
max_tokens = 3000

slo_ttft = 1
slo_tpot = 0.25


word_pool = ["hi", "hello", "yes", "no", "cat", "dog", "pig", "game", "coffee", "cake", "noodles", "burger", "football", "tennis", "ship", "car", "ship", "boat"]




def gen_prompt(num_words):
    if num_words == 0:
        return ""
    rand_nums = np.random.randint(0, len(word_pool), num_words)
    return " ".join([word_pool[n] for n in rand_nums])


def http_request(prompt):
    json_obj = {}
    json_obj["model"] = model
    json_obj["messages"] = [{"role": "user", "content": prompt}]
    json_obj["max_tokens"] = max_tokens
    # json_obj["extra_body"] = {"return_token_ids": True}
    json_obj["stream"] = True

    client = OpenAI(api_key="", base_url=base_url)
    submit_time = time.time()
    first_token_time = -1
    stream = client.chat.completions.create(**json_obj)
    response_len = 0
    for chunk in stream:
        choice = chunk.choices[0]
        if hasattr(choice, "token_ids"):
            # sometimes choice.token_ids doesn't not exists
            chunk_len = len(choice.token_ids)
        else:
            chunk_len = 0
        if chunk_len > 0:
            if first_token_time == -1:
                first_token_time = time.time()
        response_len += chunk_len
    complete_time = time.time()
    ttft = first_token_time - submit_time
    tpot = (complete_time - first_token_time) / response_len
    return ttft, tpot


fixed_prefix = gen_prompt(fixed_prefix_len)


def request_proc(worker_id, io_remain_requests, o_ttfts, o_tpots, o_slo_passes):
    #np.random.seed(worker_id)
    while True:
        with io_remain_requests.get_lock():
            if io_remain_requests.value <= 0:
                break
            io_remain_requests.value = io_remain_requests.value - 1
        subfix_len = np.random.randint(subfix_min_len, subfix_max_len)
        prompt2 = gen_prompt(subfix_len)
        ttft, tpot = http_request(fixed_prefix + ' ' + prompt2)
        o_ttfts.append(ttft)
        o_tpots.append(tpot)
        if ttft <= slo_ttft and tpot <= slo_tpot:
            o_slo_passes.append(1)
        else:
            o_slo_passes.append(0)


#np.random.seed(123)
remain_requests = Value('i', num_requests)
manager = Manager()
ttfts = manager.list([])
tpots = manager.list([])
slo_passes = manager.list([])

start_time = time.time()
processes = []
for w in range(num_workers):
    p = Process(target=request_proc, args=(w, remain_requests, ttfts, tpots, slo_passes))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
end_time = time.time()

ttfts = list(ttfts)
tpots = list(tpots)
slo_passes = list(slo_passes)

rps = num_requests / (end_time - start_time)

print(f"mean ttfts: {np.mean(ttfts)}")
print(f"p99 ttfts: {np.quantile(ttfts, 0.99)}")
print(f"mean tpots: {np.mean(tpots)}")
print(f"p99 tpots: {np.quantile(tpots, 0.99)}")
print(f"SLO attainment rate: {np.mean(slo_passes)}")
print(f"RPS: {rps}")
