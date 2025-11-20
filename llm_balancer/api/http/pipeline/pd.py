from .pipeline import Pipeline
from .utils import to_prefill_then_decode_task, async_send_stream_p_then_d, STREAM_DONE
from llm_balancer.balancer import Balancer


class PD_Pipeline(Pipeline):

    def __init__(self, tokenizer, balancer: Balancer):
        super().__init__(tokenizer, balancer)

    async def handle_chat_completions(self, request, _):
        request_json = await request.json()
        task = to_prefill_then_decode_task(self._tokenizer, request, request_json)
        advice = self._balancer.dynamic_pd.update(advice_only=False)
        # if need to re-balance P/D ratio by external mechanisms, then set advice_only = True
        # and obtain the suggested endpoint and stage by advice.best_switchable, advice.switchables
        # and advice.new_stage
        handle = self._balancer.route(task).on_submit()
        print(f"Send prefill then decode -> {handle.endpoint.id}")
        async for resp in async_send_stream_p_then_d(request_json, handle, yield_headers=True):
            yield resp
        yield STREAM_DONE
