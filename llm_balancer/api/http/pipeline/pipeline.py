import asyncio
import time

from fastapi import BackgroundTasks, Request

from llm_balancer.balancer import Balancer


class Pipeline:

    def __init__(self, tokenizer, balancer: Balancer):
        self._tokenizer = tokenizer
        self._balancer = balancer

    async def handle_chat_completions(self, request: Request, background_tasks: BackgroundTasks):
        raise NotImplementedError


class BatchedPipeline(Pipeline):

    class _Batch:
        def __init__(self):
            self.tasks = []
            self.routes = []
            self.num_unpop_routes = 0
            self.first_task_time = -1

        @property
        def size(self):
            return len(self.tasks)

        def add(self, task):
            self.tasks.append(task)
            if len(self.tasks) == 1:
                self.first_task_time = time.time()
            return len(self.routes) + len(self.tasks)

        def on_routed(self, routes):
            self.routes.extend(routes)
            self.num_unpop_routes += len(routes)
            self.tasks.clear()
            self.first_task_time = -1

        def pop_route(self, pop_idx):
            if pop_idx >= len(self.routes):
                return None
            route = self.routes[pop_idx]
            if route is not None:
                self.routes[pop_idx] = None
                self.num_unpop_routes -= 1
                if self.num_unpop_routes == 0:
                    self.routes.clear()
            return route

    def __init__(self, tokenizer, balancer: Balancer, max_batch_size, max_batch_time, is_dynamic_pd):
        super().__init__(tokenizer, balancer)
        self._batches = {}
        self._is_dynamic_pd = is_dynamic_pd
        self._max_batch_size = max_batch_size
        self._max_batch_time = max_batch_time

    async def _batched_route(self, task):
        batch = self._batches.get(task.stage)
        if batch is None:
            batch = self._Batch()
            pop_idx = batch.add(task)
            self._batches[task.stage] = batch
        else:
            pop_idx = batch.add(task)
        while True:
            ret_route = self._fetch_route(batch, pop_idx)
            if ret_route is None:
                await asyncio.sleep(0)
            else:
                break
        return ret_route

    def _fetch_route(self, batch, pop_idx):
        if batch.size >= self._max_batch_size:
            if self._is_dynamic_pd:
                self._balancer.dynamic_pd.update()
            routes = self._balancer.batch_route(batch.tasks)
            batch.on_routed(routes)
        else:
            elapsed = time.time() - batch.first_task_time
            if elapsed >= self._max_batch_time:
                if self._is_dynamic_pd:
                    self._balancer.dynamic_pd.update()
                routes = self._balancer.batch_route(batch.tasks)
                batch.on_routed(routes)
        return batch.pop_route(pop_idx)
