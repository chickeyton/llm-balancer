from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import StreamingResponse

api_router = APIRouter()


@api_router.post("/v1/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    stream = request.app.state.pipeline.handle_chat_completions(request, background_tasks)
    # dump = await
    # headers_dict = dict(headers.items())
    return StreamingResponse(
        anext(stream),
        #status_code=status,
        #headers=headers_dict,
        media_type="text/event-stream")
