# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

from dataclasses import dataclass
from typing import List

from llm_balancer.balancer.common import Stage


@dataclass
class Task:
    request_id: str

    @property
    def stage(self) -> Stage:
        raise NotImplementedError


@dataclass
class EncodeTask(Task):
    @property
    def stage(self) -> Stage:
        raise Stage.ENCODE


@dataclass
class ViTEncodeTask(EncodeTask):
    num_patches: int


@dataclass
class PrefillTask(Task):
    prompt_tokens: List[int]

    @property
    def stage(self) -> Stage:
        raise Stage.PREFILL


@dataclass
class DecodeTask(Task):
    prefill_len: int
    predicted_decode_len: int
    prefill_route: "PrefillRoute"

    @property
    def stage(self) -> Stage:
        raise Stage.DECODE


@dataclass
class PrefillThenDecodeTask(Task):
    prompt_tokens: List[int]
    predicted_decode_len: int

    @property
    def stage(self) -> Stage:
        raise Stage.PREFILL_THEN_DECODE
