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
            self.routes = {}
            self.first_task_time = -1

        @property
        def size(self):
            return len(self.tasks)

        @property
        def is_empty(self):
            return bool(self.tasks)

        def add(self, task):
            self.tasks.append(task)
            if len(self.tasks) == 1:
                self.first_task_time = time.time()

        def on_routed(self, routes):
            for route in routes:
                self.routes[route.request_id] = route
            self.tasks.clear()
            self.first_task_time = -1

        def pop_route(self, request_id):
            return self.routes.pop(request_id, None)

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
            batch.add(task)
            self._batches[task.stage] = batch
        else:
            batch.add(task)
        while True:
            ret_route = self._fetch_route(batch, task.request_id)
            if ret_route is None:
                await asyncio.sleep(0)
            else:
                break
        assert ret_route.request_id == task.request_id
        return ret_route

    def _fetch_route(self, batch, request_id):
        if batch.size >= self._max_batch_size:
            if self._is_dynamic_pd:
                self._balancer.dynamic_pd.update()
            routes = self._balancer.batch_route(batch.tasks)
            batch.on_routed(routes)
        elif batch.size > 0:
            elapsed = time.time() - batch.first_task_time
            if elapsed >= self._max_batch_time:
                if self._is_dynamic_pd:
                    self._balancer.dynamic_pd.update()
                routes = self._balancer.batch_route(batch.tasks)
                batch.on_routed(routes)
        return batch.pop_route(request_id)
