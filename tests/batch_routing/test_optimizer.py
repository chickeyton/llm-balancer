import time

import numpy as np
from llm_balancer.balancer.router import BatchRouteLocalSearch

task_workloads = np.random.uniform(1, 1000, size=(10, 50)).astype(np.float64)
queue_workloads = np.random.uniform(0, 10000, size=(task_workloads.shape[0], )).astype(np.float64)


start = time.time()
optimizer = BatchRouteLocalSearch()
assign, worker_workloads = optimizer.optimize(task_workloads, queue_workloads)
elpased = time.time() - start

original_var = np.var(queue_workloads)
new_var = np.var(worker_workloads)

print(f"compute time: {elpased}")
print(f"original variance:{original_var}")
print(f"new variance:{new_var}")
print(f"diff:{new_var - original_var}")
