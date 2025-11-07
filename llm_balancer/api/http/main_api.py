from fastapi import (
    APIRouter,
    BackgroundTasks,
    Request,
)


main_api_router = APIRouter()


@main_api_router.post("/v1/chat/completions")
async def route_chat_complation(request: Request, background_tasks: BackgroundTasks):
    await request.app.state.pipline.handle_completions("/v1/chat/completions", request, background_tasks)
