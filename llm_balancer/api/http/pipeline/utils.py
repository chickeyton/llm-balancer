import json
import uuid

from llm_balancer.balancer import PrefillTask, DecodeTask, PrefillThenDecodeTask, Stage


def to_prefill_task(tokenizer, request, request_json):
    request_id = request.headhers.get("X-Request-Id") or str(uuid.uuid4())
    prompt_tokens = tokenizer.apply_chat_template(request_json["messages"])
    task = PrefillTask(request_id=request_id, prompt_tokens=prompt_tokens)
    return task


def to_prefill_then_decode_task(tokenizer, request, request_json):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    prompt_tokens = tokenizer.apply_chat_template(request_json["messages"])
    task = PrefillThenDecodeTask(request_id=request_id,
                                 prompt_tokens=prompt_tokens,
                                 predicted_decode_len=-1)
    return task


def to_decode_task(prefill_route, predicted_decode_len):
    return DecodeTask(request_id=prefill_route.request_id,
                      prefill_len=prefill_route.num_prompt_tokens + 1,
                      predicted_decode_len=predicted_decode_len)


async def async_send_task(request_json, task_handle):
    try:
        response_text = ""
        client = task_handle.route.endpoint.get_openai_client()
        stream = client.chat.completions.create(
            model=request_json["model"],
            messages=request_json["messages"],
            stream=True,
            logprobs=True,
            max_tokens=1 if task_handle.stage == Stage.PREFILL else request_json.get("max_tokens")
        )
        print(f"====== before yield stream.headers:{stream.response.headers}, stream.status_code:{stream.response.status_code}")
        yield stream.response.headers, stream.response.status_code
        print(f"====== after yield stream.headers, stream.status_code")
        for chunk in stream:

            print(f"======= chunk: {chunk}")

            response_text += chunk.choices[0].delta.content
            chunk_len = len(chunk.choices[0].delta.logprobs.contents)
            task_handle.on_respond(chunk_len)
            stream_data = chunk.model_dump_json()
            yield f"data: {stream_data}\n\n"
        task_handle.on_finished()
        yield "data: [DONE]\n\n"

        if task_handle.stage == Stage.PREFILL:
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
        print(f"=========== async_send_task error:{e}")
        task_handle.on_finished(e)
        error_message = {"error": {"message": str(e), "type": "api_error"}}
        yield f"data: {json.dumps(error_message)}\n\n"
        yield "data: [DONE]\n\n"
