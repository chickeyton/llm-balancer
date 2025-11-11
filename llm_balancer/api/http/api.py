from fastapi import (
    APIRouter,
    BackgroundTasks,
    Request,
    StreamingResponse
)


api_router = APIRouter()


@api_router.post("/v1/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    return StreamingResponse(
        request.app.state.pipline.handle_chat_completions(request, background_tasks),
        media_type="text/event-stream")
