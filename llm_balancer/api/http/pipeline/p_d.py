from .pipeline import Pipeline, BatchedPipeline
from .utils import to_prefill_task, to_decode_task, async_send_prefill, async_send_stream_decode, STREAM_DONE
from llm_balancer.balancer import Balancer


class P_D_Pipeline(Pipeline):

    def __init__(self, tokenizer, balancer: Balancer):
        super().__init__(tokenizer, balancer)

    async def handle_chat_completions(self, request, _):
        request_json = await request.json()
        prefill_task = to_prefill_task(self._tokenizer, request, request_json)
        # advice = self._balancer.dynamic_pd.update(advice_only=False)

        if self._balancer.dynamic_pd is not None:
            pd_ep_infos = self._get_pd_ep_infos()
            realloc_advice = self._balancer.dynamic_pd.advise_realloc(pd_ep_infos)
            elastic_advice = self._balancer.dynamic_pd.advise_elastic(pd_ep_infos)
            if realloc_advice:
                print("==================== Reallocate Advice ====================")
                print(f"num endpoints: {len(realloc_advice.switch_endpoints)}")
                if pd_ep_infos[realloc_advice.switch_endpoints[0]].is_prefill:
                    print(f"switch to PREFILL")
                else:
                    print(f"switch to DECODE")

            if elastic_advice:
                print("==================== Elastic Advice ====================")
                print(f"drop prefills: {len(elastic_advice.drop_prefills)}")
                print(f"drop decodes: {len(elastic_advice.drop_decodes)}")
                print(f"add prefills: {elastic_advice.num_add_prefills}")
                print(f"add decodes: {elastic_advice.num_add_decodes}")
                print(f"P/D : {elastic_advice.new_total_prefills}:{elastic_advice.new_total_decodes}")

        # if need to re-balance P/D ratio by external mechanisms, then set advice_only = True
        # and obtain the suggested endpoint and stage by advice.best_switchable, advice.switchables
        # ,advice.new_stage, advice.new_num_prefills, advice.new_num_decodes, if advice is None
        # means no new action or there is not enough stats or time for advisory yet
        handle = self._balancer.route(prefill_task).on_submit()
        print(f"Send prefill -> {handle.endpoint.id}")
        await async_send_prefill(request_json, handle)
        decode_task = to_decode_task(handle.route, 100)  # TODO: decode length prediction
        handle = self._balancer.route(decode_task).on_submit()
        print(f"Send decode -> {handle.endpoint.id}")
        async for resp in async_send_stream_decode(request_json, handle, yield_headers=True):
            yield resp
        yield STREAM_DONE




class P_D_BatchedPipeline(BatchedPipeline):

    def __init__(self, tokenizer, balancer: Balancer, max_batch_size, max_batch_time):
        super().__init__(tokenizer, balancer, max_batch_size, max_batch_time, is_dynamic_pd=True)

    async def handle_chat_completions(self, request, _):
        request_json = await request.json()
        prefill_task = to_prefill_task(self._tokenizer, request, request_json)
        route = await self._batched_route(prefill_task)
        handle = route.on_submit()
        print(f"Send prefill -> {handle.endpoint.id}")
        await async_send_prefill(request_json, handle)
        decode_task = to_decode_task(handle.route, 100)  # TODO: decode length prediction
        route = await self._batched_route(decode_task)
        handle = route.on_submit()
        print(f"Send decode -> {handle.endpoint.id}")
        async for resp in async_send_stream_decode(request_json, handle, yield_headers=True):
            yield resp
        yield STREAM_DONE
