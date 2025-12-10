import asyncio
from threading import Thread

import numpy as np
import time
from multiprocessing import Process, Manager, Value
from openai import OpenAI

base_url = "http://localhost:8888/v1"
api = "/chat/completions"
model = "Qwen/Qwen2-7B"

num_requests = 30
num_workers = 20
fixed_prefix_len = 100
num_fixed_prefixs = 10
subfix_min_len = 20
subfix_max_len = subfix_min_len + fixed_prefix_len
max_tokens = 1
rps = 10

slo_ttft = 1
slo_tpot = 0.25


word_pool = ["hi", "hello", "yes", "no", "cat", "dog", "pig", "game", "coffee", "cake", "noodles", "burger", "football", "tennis", "ship", "car", "ship", "boat"]


def gen_prompt(num_words):
    if num_words == 0:
        return ""
    rand_nums = np.random.randint(0, len(word_pool), num_words)
    return " ".join([word_pool[n] for n in rand_nums])

np.random.seed(996)
fixed_prefixs = [gen_prompt(fixed_prefix_len) for _ in range(num_fixed_prefixs)]


def create_request(prompt):
    json_obj = {}
    json_obj["model"] = model
    json_obj["messages"] = [{"role": "user", "content": prompt}]
    # json_obj["prompt"] = prompt
    json_obj["max_tokens"] = max_tokens
    # json_obj["extra_body"] = {"return_token_ids": True}
    json_obj["stream"] = True

    client = OpenAI(api_key="", base_url=base_url)
    request = client.chat.completions.create(**json_obj)
    return request


sent_requests = 0
gathered_requests = 0
open_requests = []


async def send_requests():
    global num_requests
    global sent_requests
    global open_requests
    while sent_requests < num_requests:
        update_start = time.time()
        num_update_requests = 0
        while num_update_requests < rps:
            prompt = fixed_prefixs[np.random.randint(len(fixed_prefixs))]
            if subfix_min_len > 0:
                subfix_len = np.random.randint(subfix_min_len, subfix_max_len)
                prompt += " " + gen_prompt(subfix_len)
            open_requests.append(create_request(prompt))
            num_update_requests += 1
            sent_requests += 1
        elapsed = time.time() - update_start
        if elapsed < 1.0:
            sleep_time = 1.0 - elapsed
            await asyncio.sleep(sleep_time)


async def gather_requests(loop):
    global num_requests
    global gathered_requests
    global open_requests
    while gathered_requests < num_requests:
        requests = open_requests
        open_requests = []
        if len(requests) == 0:
            await asyncio.sleep(0.1)
            continue
        await asyncio.gather(requests)
        gathered_requests += len(requests)
    loop.stop()


loop = asyncio.new_event_loop()
#thread = Thread(target=loop.run_forever)
#thread.start()
asyncio.run_coroutine_threadsafe(gather_requests(loop), loop)
asyncio.run_coroutine_threadsafe(send_requests(), loop)
loop.run_until_complete()
#thread.join()

"""



def http_request(prompt):
    json_obj = {}
    json_obj["model"] = model
    json_obj["messages"] = [{"role": "user", "content": prompt}]
    #json_obj["prompt"] = prompt
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




def request_proc(worker_id, io_remain_requests, o_ttfts, o_tpots, o_slo_passes):
    np.random.seed(worker_id)
    while True:
        time.sleep(0.5)
        with io_remain_requests.get_lock():
            if io_remain_requests.value <= 0:
                break
            io_remain_requests.value = io_remain_requests.value - 1

        

        msg = fixed_prefixs[np.random.randint(len(fixed_prefixs))]

        if subfix_min_len > 0:
            subfix_len = np.random.randint(subfix_min_len, subfix_max_len)
            msg += " " + gen_prompt(subfix_len)

        ttft, tpot = http_request(msg)
        o_ttfts.append(ttft)
        o_tpots.append(tpot)
        if ttft <= slo_ttft and tpot <= slo_tpot:
            o_slo_passes.append(1)
        else:
            o_slo_passes.append(0)


np.random.seed(996)
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

"""
