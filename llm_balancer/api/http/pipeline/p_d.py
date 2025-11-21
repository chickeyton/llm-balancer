from .pipeline import Pipeline
from .utils import to_prefill_task, to_decode_task, async_send_prefill, async_send_stream_decode, STREAM_DONE
from llm_balancer.balancer import Balancer


class P_D_Pipeline(Pipeline):

    def __init__(self, tokenizer, balancer: Balancer):
        super().__init__(tokenizer, balancer)

    async def handle_chat_completions(self, request, _):
        request_json = await request.json()
        prefill_task = to_prefill_task(self._tokenizer, request, request_json)
        advice = self._balancer.dynamic_pd.update(advice_only=False)
        # if need to re-balance P/D ratio by external mechanisms, then set advice_only = True
        # and obtain the suggested endpoint and stage by advice.best_switchable, advice.switchables
        # ,advice.new_stage, advice.new_num_prefills, advice.new_num_decodes, if advice is None
        # means no new action or there is not enough stats or time for advisory yet
        handle = self._balancer.route(prefill_task).on_submit()
        print(f"Send prefill -> {handle.endpoint.id}")
        await async_send_prefill(request_json, handle)
        decode_task = to_decode_task(handle.route, 100)  # TODO: decode length prediction
        handle = self._balancer.route(decode_task).on_submit()
        print(f"Send decode -> {handle.endpoint.id}")
        async for resp in async_send_stream_decode(request_json, handle, yield_headers=True):
            yield resp
        yield STREAM_DONE
