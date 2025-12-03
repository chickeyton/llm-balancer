# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project
from dataclasses import dataclass
from enum import Enum


DEFAULT_P_QUANTILE = 0.99


class Stage(Enum):
    ENCODE = "ENCODE"
    PREFILL = "PREFILL"
    DECODE = "DECODE"
    PREFILL_THEN_DECODE = "PREFILL_THEN_DECODE"


@dataclass
class RequestMeta:
    id: str
    submit_time: float
    ttft: float = -1
    tpot: float = -1


@dataclass
class ServiceLevelObj:
    p_quantile: float = 0.99
    ttft: float = 1
    tpot: float = 0.25


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
