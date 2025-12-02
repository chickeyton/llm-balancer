import time
from dataclasses import dataclass
import numpy as np

from llm_balancer.balancer.task_handle import PrefillThenDecodeHandle, DecodeHandle
from llm_balancer.balancer.common import ServiceLevelObj, DEFAULT_P_QUANTILE
from llm_balancer.balancer.utils import CircularList


class Logger:
    def info(self, msg):
        raise NotImplementedError

    def error(self, msg):
        raise NotImplementedError

    def task_ended(self, handle):
        raise NotImplementedError


class NullLogger(Logger):
    def info(self, msg):
        pass

    def error(self, msg):
        pass

    def task_ended(self, handle):
        pass


class PrintLogger(Logger):
    def info(self, msg):
        print(msg)

    def error(self, msg):
        print(msg)

    def task_ended(self, handle):
        print(f"Task ended, request:{handle.request_id} stage:{handle.stage}")


@dataclass
class Stats:
    time: float = -1
    p_quantile: float = -1
    ttft_mean: float = -1
    ttft_quantile: float = -1
    tpot_mean: float = -1
    tpot_quantile: float = -1
    e2e_mean: float = -1
    e2e_quantile: float = -1
    slo_attainment: float = -1
    num_requests: int = -1


class CompositeLogger(Logger):
    def __init__(self, loggers):
        self.loggers = loggers

    def info(self, msg):
        for logger in self.loggers:
            logger.info(msg)

    def error(self, msg):
        for logger in self.loggers:
            logger.error(msg)

    def task_ended(self, handle):
        for logger in self.loggers:
            logger.task_ended(handle)


class StatsLogger(Logger):

    def __init__(self, max_hist_len: int = 1000, service_level_obj: ServiceLevelObj = None):
        self._service_level_obj = service_level_obj
        self._ttft_hist = CircularList(max_hist_len)
        self._tpot_hist = CircularList(max_hist_len)
        self._e2e_hist = CircularList(max_hist_len)
        self._num_requests = 0
        self._num_slo_attained = 0

    @property
    def service_level_obj(self):
        return self._service_level_obj

    def info(self, msg):
        pass

    def error(self, msg):
        pass

    def task_ended(self, handle):
        if isinstance(handle, (DecodeHandle, PrefillThenDecodeHandle)):
            if handle.request_meta.ttft > 0:
                self._ttft_hist.append(handle.request_meta.ttft)
            if handle.request_meta.tpot > 0:
                self._tpot_hist.append(handle.request_meta.tpot)
            if 0 < handle.request_meta.submit_time < handle.end_time:
                self._e2e_hist.append(handle.end_time - handle.request_meta.submit_time)
            self._num_requests += 1
            if self._service_level_obj:
                if 0 < handle.request_meta.ttft <= self._service_level_obj.ttft and \
                        0 < handle.request_meta.tpot <= self._service_level_obj.tpot:
                    self._num_slo_attained += 1

    def reset(self):
        self._ttft_hist.clear()
        self._tpot_hist.clear()
        self._e2e_hist.clear()
        self._num_requests = 0
        self._num_slo_attained = 0

    def compute_stats(self) -> Stats:
        stats = Stats(time=time.time())
        if self._service_level_obj:
            stats.p_quantile = self._service_level_obj.p_quantile
        else:
            stats.p_quantile = DEFAULT_P_QUANTILE

        if self._ttft_hist.length() > 0:
            stats.ttft_mean = np.mean(self._ttft_hist.list)
            stats.ttft_quantile = np.quantile(self._ttft_hist.list, stats.p_quantile)
        if self._tpot_hist.length() > 0:
            stats.tpot_mean = np.mean(self._tpot_hist.list)
            stats.tpot_quantile = np.quantile(self._tpot_hist.list, stats.p_quantile)
        if self._e2e_hist.length() > 0:
            stats.e2e_mean = np.mean(self._e2e_hist.list)
            stats.e2e_quantile = np.quantile(self._e2e_hist.list, stats.p_quantile)

        if self._service_level_obj and self._num_requests > 0:
            stats.slo_attainment = self._num_slo_attained / self._num_requests
        stats.num_requests = self._num_requests
        return stats
