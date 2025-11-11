import os


def _kv_event_batch_init(self, *arg, **kwargs):
    self._init_ori(self, *arg, **kwargs)
    self.vllm_instance_id = os.getenv("VLLM_INSTANCE_ID", "")


def monkey_patch():
    KVEventBatch._init_ori = KVEventBatch.__init__
    KVEventBatch.__init__ = _kv_event_batch_init
