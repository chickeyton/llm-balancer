from llm_balancer.balancer import Balancer
from fastapi import BackgroundTasks, Request


class Pipeline:

    def __init__(self, tokenizer, balancer: Balancer):
        self._tokenizer = tokenizer
        self._balancer = balancer

    async def handle_chat_completions(self, request: Request, background_tasks: BackgroundTasks):
        raise NotImplementedError
