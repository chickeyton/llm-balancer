from .pipeline import Pipeline
from .utils import to_prefill_then_decode_task, async_send_task
from llm_balancer.balancer import Balancer


class PD_Pipeline(Pipeline):

    def __init__(self, tokenizer, balancer: Balancer):
        super().__init__(tokenizer, balancer)

    async def handle_chat_completions(self, request, _):
        print(f"handle_chat_completions 1")
        request_json = await request.json()
        print(f"handle_chat_completions 2")
        task = to_prefill_then_decode_task(self._tokenizer, request, request_json)
        print(f"handle_chat_completions 3")
        handle = self._balancer.route(task).on_submit()
        print(f"handle_chat_completions 4")
        async for resp in async_send_task(request_json, handle):
            print(f"handle_chat_completions async_send_task yield: {resp}")
            yield resp
