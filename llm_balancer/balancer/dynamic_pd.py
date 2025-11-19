import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import numpy as np

from llm_balancer.balancer.common import Stage
from llm_balancer.balancer.endpoint import Endpoint
from llm_balancer.balancer.task_handle import TaskHandle, PrefillHandle, DecodeHandle


class DynamicPd:

    _EXCEL_SLOT: int = 0
    _GOOD_SLOT: int = 1
    _BAD_SLOT: int = 2
    _NUM_SLOTS: int = 3
    _MAX_Q_LEN_THD_PRE_EP_REQ: int = 5

    class _Action(Enum):
        NONE = 0
        P2D = 1
        D2P = 2
        KEEP_P2D_BY_BAD_TPOT = 3
        KEEP_D2P_BY_BAD_TTFT = 4

    @dataclass
    class _State:
        switchable_prefills: List[Endpoint] = None
        switchable_decodes: List[Endpoint] = None
        num_prefill_only: int = -1
        num_decode_only: int = -1
        ttft_quantile: float = -1
        tpot_quantile: float = -1
        ttft_slot: int = -1
        tpot_slot: int = -1

        @property
        def can_p2d(self):
            return len(self.switchable_prefills) + self.num_prefill_only >= 2 \
                and len(self.switchable_prefills) >= 1

        @property
        def can_d2p(self):
            return len(self.switchable_decodes) + self.num_decode_only >= 2 \
                   and len(self.switchable_decodes) >= 1

    def __init__(self, balancer: "Balancer"):
        self._balancer = balancer
        self._last_update_time = -1
        self._ttft_history = []
        self._tpot_history = []
        self._last_action = self._Action.NONE
        if self._balancer.config.service_level_obj is None:
            # SLO was not set, always optimize RPS
            self._decision_makers = [[self._decide_queue_len_guided] * self._NUM_SLOTS] * self._NUM_SLOTS
        else:
            self._decision_makers = [[None] * self._NUM_SLOTS] * self._NUM_SLOTS
            self._decision_makers[self._EXCEL_SLOT][self._EXCEL_SLOT] = self._decide_excel_or_good_slo
            self._decision_makers[self._EXCEL_SLOT][self._GOOD_SLOT] = self._decide_excel_or_good_slo
            self._decision_makers[self._GOOD_SLOT][self._EXCEL_SLOT] = self._decide_excel_or_good_slo
            self._decision_makers[self._GOOD_SLOT][self._GOOD_SLOT] = self._decide_excel_or_good_slo
            self._decision_makers[self._GOOD_SLOT][self._BAD_SLOT] = self._decide_queue_len_guided
            self._decision_makers[self._BAD_SLOT][self._GOOD_SLOT] = self._decide_queue_len_guided
            self._decision_makers[self._BAD_SLOT][self._BAD_SLOT] = self._decide_queue_len_guided
            self._decision_makers[self._EXCEL_SLOT][self._BAD_SLOT] = self._decide_excel_ttft_bad_tpot
            self._decision_makers[self._BAD_SLOT][self._EXCEL_SLOT] = self._decide_bad_ttft_excel_tpot

    def on_task_ended(self, handle: TaskHandle):
        if isinstance(handle, PrefillHandle):
            if handle.ttft > 0:
                self._ttft_history.append(handle.ttft)
        elif isinstance(handle, DecodeHandle):
            if handle.tpot > 0:
                self._tpot_history.append(handle.tpot)

    def update(self):
        if len(self._ttft_history) < self._balancer.config.dynamic_pd.update_on_requests \
                or len(self._tpot_history) < self._balancer.config.dynamic_pd.update_on_requests:
            return
        if self._last_update_time > 0:
            elapsed = time.time() - self._last_update_time
            if elapsed < self._balancer.config.dynamic_pd.min_update_time:
                return
        self._update()
        self._ttft_history.clear()
        self._tpot_history.clear()
        self._last_update_time = time.time()

    def _update(self):
        state = self._gather_state()
        action = self._decision_makers[state.ttft_slot][state.tpot_slot](state)
        self._take_action(state, action)

    def _take_action(self, state, action):
        if action in (self._Action.P2D, self._Action.KEEP_P2D_BY_BAD_TPOT):
            if not state.can_p2d:
                raise RuntimeError("Cannot P2D")
            best = self._find_best_switchable(state.switchable_prefills)
            best.set_state(Stage.DECODE)
        elif action in (self._Action.D2P, self._Action.KEEP_D2P_BY_BAD_TTFT):
            if not state.can_d2p:
                raise RuntimeError("Cannot D2P")
            best = self._find_best_switchable(state.switchable_decodes)
            best.set_state(Stage.PREFILL)
        self._last_action = action

    @staticmethod
    def _find_best_switchable(switchables):
        best = None
        min_length = -1
        for endpoint in switchables:
            length = endpoint.queue_length()
            if best is None or length < min_length:
                best = endpoint
                min_length = length
        if best is None:
            raise RuntimeError("Cannot find the best switchable")
        return best

    def _gather_state(self):
        switchable_prefills, switchable_decodes, num_prefill_only, num_decode_only = \
            self._gather_endpoints()
        ttft_quantile = np.quantile(self._ttft_history, self._balancer.config.service_level_obj.p_quantile)
        tpot_quantile = np.quantile(self._tpot_history, self._balancer.config.service_level_obj.p_quantile)
        ttft_slot = self._quantize_slo(ttft_quantile, self._balancer.config.service_level_obj.ttft)
        tpot_slot = self._quantize_slo(tpot_quantile, self._balancer.config.service_level_obj.tpot)
        return self._State(
            switchable_prefills=switchable_prefills,
            switchable_decodes=switchable_decodes,
            num_prefill_only=num_prefill_only,
            num_decode_only=num_decode_only,
            ttft_quantile=ttft_quantile,
            tpot_quantile=tpot_quantile,
            ttft_slot=ttft_slot,
            tpot_slot=tpot_slot
        )

    def _gather_endpoints(self):
        switchable_prefills = []
        switchable_decodes = []
        num_prefill_only = 0
        num_decode_only = 0
        for endpoint in self._balancer.get_up_endpoints():
            if endpoint.stage == Stage.PREFILL:
                if endpoint.is_dynamic_pd:
                    switchable_prefills.append(endpoint)
                else:
                    num_prefill_only += 1
            elif endpoint.stage == Stage.DECODE:
                if endpoint.is_dynamic_pd:
                    switchable_decodes.append(endpoint)
                else:
                    num_decode_only += 1
        return switchable_prefills, switchable_decodes, num_prefill_only, num_decode_only

    def _quantize_slo(self, metric, target) -> int:
        if metric > target:
            return self._BAD_SLOT
        excel = self._slo_boundaries(target)[0]
        if metric <= excel:
            return self._EXCEL_SLOT
        return self._GOOD_SLOT

    def _decide_excel_or_good_slo(self, state):
        tpot_mid = self._slo_boundaries(self._balancer.config.service_level_obj.tpot)[1]
        ttft_mid = self._slo_boundaries(self._balancer.config.service_level_obj.ttft)[1]
        if self._last_action == self._Action.KEEP_D2P_BY_BAD_TTFT:
            if state.tpot_quantile < tpot_mid \
                    and state.ttft_quantile > ttft_mid:
                return self._decide_keep_d2p_by_bad_ttft(state)
        elif self._last_action == self._Action.KEEP_P2D_BY_BAD_TPOT:
            if state.ttft_quantile < ttft_mid \
                    and state.tpot_quantile > tpot_mid:
                return self._decide_keep_p2d_by_bad_tpot(state)
        return self._Action.NONE

    def _decide_queue_len_guided(self, state):
        num_prefill_ep = 0
        num_decode_ep = 0
        prefill_queue_len = 0
        decode_queue_len = 0
        for endpoint in self._balancer.endpoints:
            if endpoint.stage == Stage.PREFILL:
                num_prefill_ep += 1
                prefill_queue_len += endpoint.queue_length()
            elif endpoint.stage == Stage.DECODE:
                num_decode_ep += 1
                decode_queue_len += endpoint.queue_length()
        if prefill_queue_len > decode_queue_len:
            max_queue_len = prefill_queue_len
            queue_len_threshold = num_prefill_ep * self._MAX_Q_LEN_THD_PRE_EP_REQ
        else:
            max_queue_len = decode_queue_len
            queue_len_threshold = num_decode_ep * self._MAX_Q_LEN_THD_PRE_EP_REQ

        if max_queue_len <= queue_len_threshold:
            return self._Action.NONE
        switch_threshold = max_queue_len / 2
        if prefill_queue_len < switch_threshold:
            if state.can_p2d:
                return self._Action.P2D
        elif decode_queue_len < switch_threshold:
            if state.can_d2p:
                return self._Action.D2P
        return self._Action.NONE

    def _decide_excel_ttft_bad_tpot(self, state):
        action = self._decide_keep_p2d_by_bad_tpot(state)
        if action == self._Action.NONE:
            return self._decide_queue_len_guided(state)
        return action

    def _decide_bad_ttft_excel_tpot(self, state):
        action = self._decide_keep_d2p_by_bad_ttft(state)
        if action == self._Action.NONE:
            return self._decide_queue_len_guided(state)
        return action

    def _decide_keep_d2p_by_bad_ttft(self, state):
        if state.can_d2p:
            if len(state.switchable_decodes) == 1:
                # the last one to be switched in the prolonged action
                return self._Action.D2P
            return self._Action.KEEP_D2P_BY_BAD_TTFT
        return self._Action.NONE

    def _decide_keep_p2d_by_bad_tpot(self, state):
        if state.can_p2d:
            if len(state.switchable_prefills) == 1:
                # the last one to be switched in the prolonged action
                return self._Action.P2D
            return self._Action.KEEP_P2D_BY_BAD_TPOT
        return self._Action.NONE

    @staticmethod
    def _slo_boundaries(target: float) -> Tuple[float, float, float]:
        excel = target / 2
        mid = (2 * excel * target) / (excel + target)
        return excel, mid, target
