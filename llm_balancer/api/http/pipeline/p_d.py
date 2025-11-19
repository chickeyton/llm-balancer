from .pipeline import Pipeline
from .utils import to_prefill_task, to_decode_task, async_send_prefill, async_send_stream_decode, STREAM_DONE
from llm_balancer.balancer import Balancer


class P_D_Pipeline(Pipeline):

    def __init__(self, tokenizer, balancer: Balancer):
        super().__init__(tokenizer, balancer)

    async def handle_chat_completions(self, request, _):
        request_json = await request.json()
        prefill_task = to_prefill_task(self._tokenizer, request, request_json)
        handle = self._balancer.route(prefill_task).on_submit()
        print(f"Send prefill -> {handle.endpoint.id}")
        await async_send_prefill(request_json, handle)
        decode_task = to_decode_task(handle.route, 100)  # TODO: decode length prediction
        handle = self._balancer.route(decode_task).on_submit()
        print(f"Send decode -> {handle.endpoint.id}")
        async for resp in async_send_stream_decode(request_json, handle, yield_headers=True):
            yield resp
        yield STREAM_DONE
