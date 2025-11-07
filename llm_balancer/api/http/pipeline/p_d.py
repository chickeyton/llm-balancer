import uuid

from llm_balancer.balancer import Balancer, PrefillTask, DecodeTask


class P_D_Pipline:

    def __init__(self, tokenizer, balancer: Balancer):
        self._tokenizer = tokenizer
        self._balancer = balancer

    async def handle_completions(self, api, request, _):
        request_json = await request.json()
        prefill_task = self._to_prefill_task(request, request_json)

    def _to_prefill_task(self, request, request_json):
        request_id = request.headhers.get("X-Request-Id") or str(uuid.uuid4())
        prompt_tokens = self._tokenizer.encode(request_json.get("prompt", ""))
        prefill_task = PrefillTask(request_id=request_id,
                                   prompt_tokens=prompt_tokens)
        return prefill_task

    def _to_decode_task(self, prefill_route, predicted_decode_len):
        return DecodeTask(request_id=prefill_route.request_id,
                          prefill_len=prefill_route.num_prompt_tokens + 1,
                          predicted_decode_len=predicted_decode_len)

    def _