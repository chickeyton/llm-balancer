import json
import uuid
from fastapi import HTTPException, status

from llm_balancer.balancer import PrefillTask, DecodeTask, PrefillThenDecodeTask, Stage


def to_prefill_task(tokenizer, request, request_json):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
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


async def async_send_stream_task(request_json, task_handle, yield_headers=True, yield_done=True):
    try:
        response_text = ""
        client = task_handle.route.endpoint.get_openai_client()
        request_json["stream"] = True
        if task_handle.stage == Stage.PREFILL:
            max_tokens_bak = request_json.get("max_tokens")
            request_json["max_tokens"] = 1
        request_json["extra_body"] = {"return_token_ids": True}

        stream = client.chat.completions.create(**request_json)
        # yield the header and status code first
        print(f"========================= yield header")
        if yield_headers:
            yield stream.response.headers, stream.response.status_code
        for chunk in stream:
            choice = chunk.choices[0]
            response_text += choice.delta.content
            if hasattr(choice, "token_ids"):
                # sometimes choice.token_ids doesn't not exists
                chunk_len = len(choice.token_ids)
            else:
                chunk_len = 0
            print(f"========================= chunk_len: {chunk_len}")
            if chunk_len > 0:
                task_handle.on_respond(chunk_len)

            stream_data = chunk.model_dump_json()
            yield f"data: {stream_data}\n\n"
        task_handle.on_finished()

        if yield_done:
            print(f"========================= yield done")
            yield "data: [DONE]\n\n"

        print(f"========================= response_text: [{response_text}]")

        if task_handle.stage == Stage.PREFILL:
            # restore the overwritten settings
            request_json["max_tokens"] = max_tokens_bak
        """
            # append the first token if needed
            if response_text:
                messages = request_json["message"]
                if messages[-1]["role"] == "assistant":
                    messages[-1]["content"] += response_text
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response_text
                        }
                    )
        """
    except Exception as e:
        task_handle.on_error(e)
        print(f"========================= raise error")
        raise HTTPException(status_code=500, detail=str(e))


async def async_send_prefill(request_json, prefill_handle, yield_headers=False):
    try:
        client = prefill_handle.route.endpoint.get_openai_client()
        max_tokens_bak = request_json.get("max_tokens")
        request_json["max_tokens"] = 1

        response = client.chat.completions.with_raw_response.create(**request_json)

        request_json["max_tokens"] = max_tokens_bak

        if yield_headers:
            print(f"========================= yield header")
            yield response.headers, response.status_code
        prefill_handle.on_finished()

    except Exception as e:
        prefill_handle.on_error(e)
        print(f"========================= raise error {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def async_send_stream_decode(request_json, decode_handle, yield_headers=False, yield_done=False):
    try:
        response_text = ""
        client = decode_handle.route.endpoint.get_openai_client()
        request_json["stream"] = True
        request_json["extra_body"] = {"return_token_ids": True}

        stream = client.chat.completions.create(**request_json)
        # yield the header and status code first
        print(f"========================= yield header")
        if yield_headers:
            yield stream.response.headers, stream.response.status_code
        for chunk in stream:
            choice = chunk.choices[0]
            response_text += choice.delta.content
            if hasattr(choice, "token_ids"):
                # sometimes choice.token_ids doesn't not exists
                chunk_len = len(choice.token_ids)
                print(f"========================= chunk_len: {chunk_len}")
                decode_handle.on_respond(chunk_len)
            stream_data = chunk.model_dump_json()
            yield f"data: {stream_data}\n\n"
        decode_handle.on_finished()

        if yield_done:
            print(f"========================= yield done")
            yield "data: [DONE]\n\n"

        print(f"========================= response_text: [{response_text}]")
    except Exception as e:
        decode_handle.on_error(e)
        print(f"========================= raise error")
        raise HTTPException(status_code=500, detail=str(e))


async def async_send_stream_p_then_d(request_json, task_handle, yield_headers=False, yield_done=False):
    try:
        client = task_handle.route.endpoint.get_openai_client()
        request_json["stream"] = True
        request_json["extra_body"] = {"return_token_ids": True}

        stream = client.chat.completions.create(**request_json)
        if yield_headers:
            print(f"========================= yield header")
            yield stream.response.headers, stream.response.status_code

        response_text = ""
        for chunk in stream:
            choice = chunk.choices[0]
            response_text += choice.delta.content
            if hasattr(choice, "token_ids"):
                # sometimes choice.token_ids doesn't not exists
                chunk_len = len(choice.token_ids)
            else:
                chunk_len = 0
            print(f"========================= chunk_len: {chunk_len}")
            if chunk_len > 0:
                task_handle.on_respond(chunk_len)

            stream_data = chunk.model_dump_json()
            yield f"data: {stream_data}\n\n"
        task_handle.on_finished()

        if yield_done:
            print(f"========================= yield done")
            yield "data: [DONE]\n\n"

        print(f"========================= response_text: [{response_text}]")

    except Exception as e:
        task_handle.on_error(e)
        print(f"========================= raise error")
        raise HTTPException(status_code=500, detail=str(e))
