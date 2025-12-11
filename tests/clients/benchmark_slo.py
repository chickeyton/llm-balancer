import asyncio
from dataclasses import dataclass
from threading import Thread

import numpy as np
import time
from multiprocessing import Process, Manager, Value
from openai import AsyncOpenAI

base_url = "http://localhost:8888/v1"
api = "/chat/completions"
model = "Qwen/Qwen2-7B"

num_requests = 300
#num_workers = 20
fixed_prefix_len = 10000
num_fixed_prefixs = 10
subfix_min_len = 20
subfix_max_len = subfix_min_len + fixed_prefix_len
max_tokens = 1
target_rps = 10

slo_ttft = 1
slo_tpot = 0.25

start_time = -1
end_time = -1


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

    client = AsyncOpenAI(api_key="", base_url=base_url)
    request = client.chat.completions.create(**json_obj)
    return request


@dataclass
class Request:
    task: asyncio.Task
    submit_time: float
    ended: bool = False

all_requests = []
ttfts = []


def check_rps():
    global all_requests
    cutoff = time.time() - 1.0
    for i in range(len(all_requests) - 1, -1, -1):
        if all_requests[i].submit_time < cutoff:
            return len(all_requests) - 1 - i
    return len(all_requests)


async def send_requests():
    global all_requests
    global start_time
    global end_time
    start_time = time.time()
    while True:
        prompt = fixed_prefixs[np.random.randint(len(fixed_prefixs))]
        if subfix_min_len > 0:
            subfix_len = np.random.randint(subfix_min_len, subfix_max_len)
            prompt += " " + gen_prompt(subfix_len)
        submit_time = time.time()
        request = Request(task=asyncio.create_task(create_request(prompt)),
                          submit_time=submit_time)
        all_requests.append(request)
        if len(all_requests) == num_requests:
            break
        await asyncio.sleep(0)
        #print(f"all_requests : {len(all_requests)}")
        while check_rps() >= target_rps:
            await asyncio.sleep(0.05)
    end_time = time.time()


async def gather_requests():
    global all_requests
    global ttfts
    ended_requests = 0
    #print(f"gather_requests  1")
    while ended_requests < num_requests:
        tasks_to_wait = []
        requests_to_wait = []
        for i, request in enumerate(all_requests):
            if not request.ended:
                tasks_to_wait.append(request.task)
                requests_to_wait.append(request)
        if not requests_to_wait:
            await asyncio.sleep(0)
            continue

        #print(f"gather_requests  2 {len(tasks_to_wait)}")
        dones, _ = await asyncio.wait(tasks_to_wait, return_when=asyncio.FIRST_COMPLETED)
        now = time.time()
        #print(f"gather_requests  3 {len(dones)}")
        for task in dones:
            request = requests_to_wait[tasks_to_wait.index(task)]
            request.ended = True
            ttfts.append(now - request.submit_time)
        ended_requests += len(dones)

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(gather_requests(), send_requests()))
loop.close()

actual_rps = num_requests / (end_time - start_time)
print(f"actual RPS: {actual_rps}")
print(f"mean ttfts: {np.mean(ttfts)}")
print(f"p99 ttfts: {np.quantile(ttfts, 0.99)}")

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
