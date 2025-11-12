import os
from vllm.distributed.kv_events import KVEventBatch as KVEventBatchOri


class KVEventBatch(KVEventBatchOri):

    vllm_instance_id: str = ""

    def __post_init__(self):
        self.vllm_instance_id = os.getenv("VLLM_INSTANCE_ID", "")


def monkey_patch():
    if vllm.distributed.kv_events.KVEventBatch is not KVEventBatch:
        vllm.distributed.kv_events.KVEventBatch = KVEventBatch
