import json
import uuid

from llm_balancer.balancer import Balancer, PrefillTask, DecodeTask, Stage


class P_D_Pipline:

    def __init__(self, tokenizer, balancer: Balancer):
        self._tokenizer = tokenizer
        self._balancer = balancer

    async def handle_chat_completions(self, request, _):
        request_json = await request.json()
        prefill_task = self._to_prefill_task(request, request_json)
        handle = self._balancer.route(prefill_task).on_submit()
        async for resp in self._send(request_json, handle):
            yield resp
        if handle.error:
            return
        decode_task = self._to_decode_task(handle.route, -1)
        handle = self._balancer.route(decode_task).on_submit()
        async for resp in self._send(request_json, handle):
            yield resp

    def _to_prefill_task(self, request, request_json):
        request_id = request.headhers.get("X-Request-Id") or str(uuid.uuid4())
        prompt_tokens = self._tokenizer.apply_chat_template(request_json["messages"])
        prefill_task = PrefillTask(request_id=request_id,
                                   prompt_tokens=prompt_tokens)
        return prefill_task

    def _to_decode_task(self, prefill_route, predicted_decode_len):
        return DecodeTask(request_id=prefill_route.request_id,
                          prefill_len=prefill_route.num_prompt_tokens + 1,
                          predicted_decode_len=predicted_decode_len)

    async def _send(self, request_json, handle):
        try:
            response_text = ""
            client = handle.route.endpoint.get_openai_client()
            stream = client.chat.completions.create(
                model=request_json["model"],
                messages=request_json["messages"],
                stream=True,
                logprobs=True,
                max_tokens=1 if handle.stage == Stage.PREFILL else request_json.get("max_tokens")
            )
            for chunk in stream:
                response_text += chunk.choices[0].delta.content
                chunk_len = len(chunk.choices[0].delta.logprobs.contents)
                handle.on_respond(chunk_len)
                stream_data = chunk.model_dump_json()
                yield f"data: {stream_data}\n\n"
            yield "data: [DONE]\n\n"
            handle.on_finished()

            if handle.stage == Stage.PREFILL:
                messages = request_json["message"]
                if messages[-1]["role"] == "assistant":
                    messages[-1]["content"] += " " + response_text
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response_text
                        }
                    )
        except Exception as e:
            handle.on_finished(e)
            error_message = {"error": {"message": str(e), "type": "api_error"}}
            yield f"data: {json.dumps(error_message)}\n\n"
            yield "data: [DONE]\n\n"
