import numpy as np
from llm_balancer.balancer.router import BatchRouteLocalSearch

task_workloads = np.random.uniform(1, 100000, size=(10, 50))
queue_workloads = np.random.uniform(0, 100000, size=(task_workloads.shape[0], ))


optimizer = BatchRouteLocalSearch()
assign, worker_workloads = optimizer.optimize(task_workloads, queue_workloads)


original_var = np.var(queue_workloads)
new_var = np.var(worker_workloads)

print(f"original variance:{original_var}")
print(f"new variance:{new_var}")
print(f"diff:{new_var - original_var}")
