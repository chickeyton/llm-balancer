import os
import numpy as np
import time
import requests
from multiprocessing import Process, Value, Array, Lock

base_url = "http://localhost:8888/v1"
api = "/chat/completions"
model = "Qwen/Qwen2-7B"

num_requests = 100
num_workers = 50
num_max_active_requests = 5000
fixed_prefix_len = 0
subfix_min_len = 30
subfix_max_len = 3000
max_tokens = 3230


word_pool = ["hi", "hello", "yes", "no", "cat", "dog", "pig", "game", "coffee", "cake", "noodles", "burger", "football", "tennis", "ship", "car", "ship", "boat"]

active_requests = Value('i', 0)
finish_times = Array('d', [0] * num_requests)
lock = Lock()


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

    url = base_url + api

    response = requests.post(url, json=json_obj)
    if(response.status_code != 200):
        raise ValueError(f"status_code:{response.status_code} is not 200")


fixed_prefix = gen_prompt(fixed_prefix_len)


def request_proc(worker_id, num_requests, active_requests, finish_times, lock):
    np.random.seed(worker_id)
    for i in range(num_requests):
        while True:
            time.sleep(0.01)
            with lock:
                if active_requests.value >= num_max_active_requests:
                    continue
                active_requests.value = active_requests.value + 1
                break

        subfix_len = np.random.randint(subfix_min_len, subfix_max_len)
        prompt2 = gen_prompt(subfix_len)
        start_time = time.time()
        http_request(fixed_prefix + ' ' + prompt2)
        elapsed_time = time.time() - start_time
        with lock:
            finish_times[worker_id + i]=elapsed_time
            active_requests.value = active_requests.value - 1


worker_requests = num_requests // num_workers
processes = []
for w in range(num_workers):
    p = Process(target=request_proc, args=(w, worker_requests, active_requests, finish_times, lock))
    p.start()
    processes.append(p)

for p in processes:
    p.join()


print(f"avg finish time: {np.mean(finish_times[:])}s")
