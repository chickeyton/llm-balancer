from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import StreamingResponse

api_router = APIRouter()


@api_router.post("/v1/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    return StreamingResponse(
        request.app.state.pipeline.handle_chat_completions(request, background_tasks),
        media_type="text/event-stream")
