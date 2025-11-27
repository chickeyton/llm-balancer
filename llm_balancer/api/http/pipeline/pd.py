from .pipeline import Pipeline, BatchedPipeline
from .utils import to_prefill_then_decode_task, async_send_stream_p_then_d, STREAM_DONE
from llm_balancer.balancer import Balancer


class PD_Pipeline(Pipeline):

    def __init__(self, tokenizer, balancer: Balancer):
        super().__init__(tokenizer, balancer)

    async def handle_chat_completions(self, request, _):
        request_json = await request.json()
        task = to_prefill_then_decode_task(self._tokenizer,
                                           request,
                                           request_json,
                                           100)  # TODO: decode length prediction
        handle = self._balancer.route(task).on_submit()
        print(f"Send prefill then decode -> {handle.endpoint.id}")
        async for resp in async_send_stream_p_then_d(request_json, handle, yield_headers=True):
            yield resp
        yield STREAM_DONE


class PD_BatchedPipeline(BatchedPipeline):

    def __init__(self, tokenizer, balancer: Balancer, max_batch_size, max_batch_time):
        super().__init__(tokenizer, balancer, max_batch_size, max_batch_time, is_dynamic_pd=False)

    async def handle_chat_completions(self, request, _):
        request_json = await request.json()
        task = to_prefill_then_decode_task(self._tokenizer,
                                           request,
                                           request_json,
                                           100)  # TODO: decode length prediction
        route = await self._batched_route(task)
        handle = route.on_submit()
        print(f"Send prefill then decode -> {handle.endpoint.id}")
        async for resp in async_send_stream_p_then_d(request_json, handle, yield_headers=True):
            yield resp
        yield STREAM_DONE
