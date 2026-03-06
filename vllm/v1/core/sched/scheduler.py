# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations


import itertools
import time
import datetime
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any, List
import os
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    compute_encoder_budget,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.core.sched.energy_model import EnergySimulator
from vllm.v1.core.sched.dynamo_energy_model import ClockSelector

import pandas as pd
import vllm.envs as envs
import threading
import math
from pynvml import (nvmlInit, nvmlDeviceGetName, nvmlDeviceGetClock, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceResetGpuLockedClocks,
                    nvmlDeviceSetGpuLockedClocks)

from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

nvmlInit()

NUM_CONSECUTIVE_DECODE_ONLY_TO_SET_CLOCK = 5

TPOT_BASED_SCHEDULING_WINDOW_TIME_S = 5

DYNAMOLLM_DVFS_RESCHEDULE_INTERVAL = 5  # seconds

CURRENT_CLOCK = None

def reset_gpu_clock(rank):
    device_handle = nvmlDeviceGetHandleByIndex(rank)
    threading.Thread(target=nvmlDeviceResetGpuLockedClocks,
                    args=(device_handle, )).start()
    
def lock_gpu_clock(rank, new_clock):
    device_handle = nvmlDeviceGetHandleByIndex(rank)
    threading.Thread(target=nvmlDeviceSetGpuLockedClocks,
                    args=(device_handle, new_clock, new_clock)).start()


def apply_dvfs(new_clock: int, num_ranks: int):
    """
    Apply DVFS to the GPU clocks.
    """
    global CURRENT_CLOCK

    # Parse CUDA_VISIBLE_DEVICES to identify which GPUs to control
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        # Handle cases like "0,1"
        target_ranks = [int(x.strip()) for x in cuda_visible_devices.split(',') if x.strip()]
        # We only take the first num_ranks devices
    else:
        target_ranks = list(range(num_ranks))
    
    logger.info(f"APPLY DVFS: {target_ranks} (physical ranks) to {new_clock} MHz")
    
    if CURRENT_CLOCK == new_clock:
        return
    
    if new_clock == None: # Reset
        for rank in target_ranks:
            reset_gpu_clock(rank)
    else:
        for rank in target_ranks:
            lock_gpu_clock(rank, new_clock)
    
    CURRENT_CLOCK = new_clock
    
    
class S1S2DecisionLogger:
    """Logger for S1 and S2 scheduling decisions."""
    
    def __init__(self):
        self.s1_log = []  # List of (timestamp, optimal_chunk_size)
        self.s2_log = []  # List of (timestamp, optimal_num_microbatches)
        self.kv_usage_log = [] # List of (timestamp, kv_usage)
        self.s1_exec_time_log = []  # List of (timestamp, execution_time)
        self.s2_exec_time_log = []  # List of (timestamp, execution_time)
        self.schedule_exec_time_log = []  # List of (timestamp, execution_time)
    
    def log_s1_decision(self, optimal_chunk_size: int):
        """Log S1 scheduling decision with timestamp."""
        timestamp = datetime.datetime.now().timestamp()
        self.s1_log.append((timestamp, optimal_chunk_size))
    
    def log_s2_decision(self, optimal_num_microbatches: int):
        """Log S2 scheduling decision with timestamp."""
        timestamp = datetime.datetime.now().timestamp()
        self.s2_log.append((timestamp, optimal_num_microbatches))

    def log_s1_exec_time(self, exec_time: float):
        """Log S1 algorithm execution time with timestamp."""
        timestamp = datetime.datetime.now().timestamp()
        self.s1_exec_time_log.append((timestamp, exec_time))

    def log_s2_exec_time(self, exec_time: float):
        """Log S2 algorithm execution time with timestamp."""
        timestamp = datetime.datetime.now().timestamp()
        self.s2_exec_time_log.append((timestamp, exec_time))

    def log_schedule_exec_time(self, exec_time: float):
        """Log schedule() function execution time with timestamp."""
        timestamp = datetime.datetime.now().timestamp()
        self.schedule_exec_time_log.append((timestamp, exec_time))

    def log_kv_usage(self, kv_usage: float):
        """Log KV cache usage with timestamp."""
        timestamp = datetime.datetime.now().timestamp()
        self.kv_usage_log.append((timestamp, kv_usage))
    
    def stop_profile(self, output_dir: str):
        """Save S1, S2, and KV usage logs to CSV files."""
        import pandas as pd
        
        # Save S1 log
        if len(self.s1_log) > 0:
            df_s1 = pd.DataFrame(self.s1_log, columns=["timestamp", "optimal_chunk_size"])
            s1_path = f"{output_dir}/s1_log.csv"
            df_s1.to_csv(s1_path, index=False)
            logger.info(f"S1S2DecisionLogger: s1_log saved to {s1_path}")
        else:
            logger.info("S1S2DecisionLogger: s1_log is empty")
        
        # Save S2 log
        if len(self.s2_log) > 0:
            df_s2 = pd.DataFrame(self.s2_log, columns=["timestamp", "optimal_num_microbatches"])
            s2_path = f"{output_dir}/s2_log.csv"
            df_s2.to_csv(s2_path, index=False)
            logger.info(f"S1S2DecisionLogger: s2_log saved to {s2_path}")
        else:
            logger.info("S1S2DecisionLogger: s2_log is empty")

        # Save KV usage log
        if len(self.kv_usage_log) > 0:
            df_kv = pd.DataFrame(self.kv_usage_log, columns=["timestamp", "usage"])
            kv_path = f"{output_dir}/kv_usage_log.csv"
            df_kv.to_csv(kv_path, index=False)
            logger.info(f"S1S2DecisionLogger: kv_usage_log saved to {kv_path}")
        else:
            logger.info("S1S2DecisionLogger: kv_usage_log is empty")

        # Save S1 execution time log
        if len(self.s1_exec_time_log) > 0:
            df_s1_exec = pd.DataFrame(self.s1_exec_time_log, columns=["timestamp", "execution_time"])
            s1_exec_path = f"{output_dir}/s1_exec_time_log.csv"
            df_s1_exec.to_csv(s1_exec_path, index=False)
            logger.info(f"S1S2DecisionLogger: s1_exec_time_log saved to {s1_exec_path}")
        else:
            logger.info("S1S2DecisionLogger: s1_exec_time_log is empty")

        # Save S2 execution time log
        if len(self.s2_exec_time_log) > 0:
            df_s2_exec = pd.DataFrame(self.s2_exec_time_log, columns=["timestamp", "execution_time"])
            s2_exec_path = f"{output_dir}/s2_exec_time_log.csv"
            df_s2_exec.to_csv(s2_exec_path, index=False)
            logger.info(f"S1S2DecisionLogger: s2_exec_time_log saved to {s2_exec_path}")
        else:
            logger.info("S1S2DecisionLogger: s2_exec_time_log is empty")

        # Save schedule execution time log
        if len(self.schedule_exec_time_log) > 0:
            df_schedule_exec = pd.DataFrame(self.schedule_exec_time_log, columns=["timestamp", "execution_time"])
            schedule_exec_path = f"{output_dir}/schedule_exec_time_log.csv"
            df_schedule_exec.to_csv(schedule_exec_path, index=False)
            logger.info(f"S1S2DecisionLogger: schedule_exec_time_log saved to {schedule_exec_path}")
        else:
            logger.info("S1S2DecisionLogger: schedule_exec_time_log is empty")
            
        # reset the logs
        self.s1_log = []
        self.s2_log = []
        self.kv_usage_log = []
        self.s1_exec_time_log = []
        self.s2_exec_time_log = []
        self.schedule_exec_time_log = []

class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
        pp_size: int = 1,
        tp_size: int = 1,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # SLO parameters - can be updated dynamically via API endpoints
        self.tpot_slo = 0.3  # Time per output token SLO in seconds
        self.tbt_slo = 0.45  # Time between tokens (P90) SLO in seconds
        self.ttft_slo = 4.0  # Time to first token (P90) SLO in seconds

        self.pp_size = pp_size
        self.tp_size = tp_size
        self.step_cnt = 0  # (hj) Added for profiling purposes
        self.split_cnt = 0  # (hj) This is used to split large batches into smaller when VEs increase.
        self.split_ratio = 0
        self.num_decode_phases = self.pp_size
        
        self.use_s1 = self.parallel_config.use_s1
        self.use_s1_dvfs_only = self.parallel_config.use_s1_dvfs_only
        self.disagg_mode = self.parallel_config.disagg_mode
        self.use_s2 = self.parallel_config.use_s2
        self.use_window_based_dvfs_only = self.parallel_config.use_window_based_dvfs_only
        self.measure_fidelity = self.parallel_config.measure_fidelity
        
        if self.vllm_config.kv_transfer_config != None:
            self.kv_role = self.vllm_config.kv_transfer_config.kv_role
        else:
            self.kv_role = "kv_both"
        
        logger.info(f"KV_ROLE : {self.kv_role}")
        
        if self.disagg_mode:
            self.is_prefiller = self.kv_role == "kv_producer"
            self.is_decoder = self.kv_role == "kv_consumer"
            if self.is_prefiller:
                assert self.use_s2 == False
            if self.is_decoder:
                assert self.use_s1 == False
        else:
            self.is_prefiller = True
            self.is_decoder = True
        
        # budget_history, which is a deque of (time, budget) tuples
        self.budget_history = deque(maxlen=5000)
        self.num_total_prefill_tokens_in_window = 0
        
        self.last_prefill_timestamp = time.monotonic()
        self.last_prefill_ttft = 0
        
        self.optimal_microbatch_size = pp_size
        
        # HJ : for debugging (fidelity test). This denotes the expected TBT for the current prefill request.
        self.expected_prefill_tbt = 0 # if has_prefill_request is true, expected_tbt is this
        self.expected_decode_tbt = 0 # if has_prefill_request is false, expected_tbt is this
        self.expected_tbt = 0
        
        if self.use_s1 or self.use_s2:
            model_name = vllm_config.model_config.model
            model_name_clean = model_name.replace("/", "_")
            gpu_name = nvmlDeviceGetName(
                nvmlDeviceGetHandleByIndex(0))

            # (hj) Load offline profile data from csv file. In benchmarks/energy/offline_profile_results/
            file_name_working = f"dvfs_profile_{gpu_name}_{model_name_clean}_tp{tp_size}_pp{pp_size}_one.csv"
            dir = "offline_profile_results"  # TODO: Change this to something more sensible
            file_full_path_working = f"{os.getcwd()}/{dir}/{file_name_working}" 

            self.energy_simulator = EnergySimulator(
                profiling_csv_path=file_full_path_working,
                num_pp=self.pp_size,
                num_tp=self.tp_size,
                gpu_name=gpu_name,
                model_name=model_name_clean)
            
            self.prefill_clock = self.energy_simulator.default_clock
            self.decode_clock = self.energy_simulator.default_clock
        else:
            self.energy_simulator = None
            
        # TODO: if use_window_based_dvfs_only, use energy_simulator to schedule window-based DVFS only
        if self.use_window_based_dvfs_only:
            dyn_dir = "dynamollm_profiles"
            model_name = vllm_config.model_config.model
            model_name_clean = model_name.replace("/", "_")
            gpu_name = nvmlDeviceGetName(
                nvmlDeviceGetHandleByIndex(0))

            dynamo_csv_name = f"dynamo_dvfs_profile_{gpu_name}_{model_name_clean}.csv"
            dynamo_csv_path = f"{os.getcwd()}/{dyn_dir}/{dynamo_csv_name}"

            # Store these for ClockSelector re-initialization
            self._dynamo_csv_path = dynamo_csv_path
            if not hasattr(self,
                        '_dynamo_csv_path') or self._dynamo_csv_path is None:
                return

            # Initialize ClockSelector
            self._reinitialize_clock_selector()

        self.num_waiting_prefill_tokens = 0
        self.new_request_arrived = False
        self.num_consecutive_decode_only = 0
        
        self.last_tpot_window_time = time.monotonic()
        
        self.last_dynamo_schedule_time = time.monotonic()
        
        # DynamoLLM-style scheduling tracking variables
        self.tokens_since_last_calculation = 0
        self.current_tps_ema = 0.0
        self.tps_ema_alpha = 0.2  # Weight for new measurements in EMA (0-1)
        
        # Performance targets (SLO parameters) - from scheduler config or defaults

        self.budget_schedule_log = []  # (hj) Log of budget schedule
        self.microbatch_schedule_log = []  # (hj) Log of microbatch schedule
        self.s1_s2_logger = S1S2DecisionLogger()  # (hj) Log of S1/S2 decisions


        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=bool(self.cache_config.enable_prefix_caching),
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        
        self.phase = 0  # (hj) Added for profiling purposes
        self.step_cnt = 0  # (hj) Added for profiling purposes
        self.to_apply_split_count = 0

    def _reinitialize_clock_selector(self):
        """
        Internal method to re-initialize ClockSelector with current SLO settings.
        This should be called whenever TTFT or TBT SLO values are updated.
        """
        if not hasattr(self, '_dynamo_csv_path') or self._dynamo_csv_path is None:
            logger.warning(
                "Scheduler : Cannot re-initialize ClockSelector - dynamo_csv_path not set"
            )
            return

        self.clock_selector = ClockSelector(
            profiling_csv_path=self._dynamo_csv_path,
            ttft_slo_ms=self.ttft_slo * 1000,
            tbt_slo_ms=self.tbt_slo * 1000,
            use_tbt_p90=True,
            use_ttft_p99=False)

        logger.info(
            f"Scheduler : ClockSelector re-initialized with TTFT SLO={self.ttft_slo}s, TBT SLO={self.tbt_slo}s"
        )

    def update_ttft_slo(self, ttft_slo: float):
        """
        Update the TTFT (time to first token) SLO target.
        Re-initializes ClockSelector if window-based DVFS is enabled.
        """
        self.ttft_slo = ttft_slo
        logger.info(f"Scheduler : TTFT SLO updated to {ttft_slo}s")
        
        if self.use_window_based_dvfs_only:
            self._reinitialize_clock_selector()

    def update_tbt_slo(self, tbt_slo: float):
        """
        Update the TBT (time between tokens) SLO target.
        Re-initializes ClockSelector if window-based DVFS is enabled.
        """
        self.tbt_slo = tbt_slo
        logger.info(f"Scheduler : TBT SLO updated to {tbt_slo}s")
        
        if self.use_window_based_dvfs_only:
            self._reinitialize_clock_selector()
        
    def start_profile(self, output_dir: str | None = None):
        self.profiling_start_time = time.monotonic()
        self.output_dir = output_dir

    def stop_profile(self):
        if self.output_dir is None:
            directory = envs.VLLM_TORCH_PROFILER_DIR
        else:
            print("[EnergyModel] Using custom output directory:",
                  self.output_dir)
            directory = self.output_dir

        # Save budget_schedule_log to a csv file
        # use pandas to save the csv file

        # if budget_schedule_log is empty, skip
        # if len(self.budget_schedule_log) == 0:
        #     logger.info(
        #         "Scheduler : stop_profile : budget_schedule_log is empty")
        #     return

        # df = pd.DataFrame(self.budget_schedule_log)

        # df.columns = [
        #     "curr_time", "total_time_taken_prefill", "total_time_taken_decode",
        #     "total_time_taken", "num_running", "num_waiting",
        #     "new_budget_size", "optimal_chunk_size", "optimal_clock",
        #     "slo_used", "slo_budget_size"
        # ]
        # df.to_csv(f"{directory}/budget_schedule_log.csv", index=False)
        # # logger informs save location
        # logger.info(
        #     f"Scheduler : stop_profile : budget_schedule_log saved to {directory}/budget_schedule_log.csv"
        # )

        # # Save microbatch_schedule_log to a csv file
        # if len(self.microbatch_schedule_log) == 0:
        #     logger.info(
        #         "Scheduler : stop_profile : microbatch_schedule_log is empty")
        # else:
        #     df = pd.DataFrame(self.microbatch_schedule_log)
        #     df.columns = ["curr_time", "min_microbatch_size"]
        #     df.to_csv(f"{directory}/microbatch_schedule_log.csv", index=False)

        # df = pd.DataFrame(self.full_scheduling_history)
        # df.columns = [
        #     "total_num_scheduled_tokens", "time", "time_taken",
        #     "has_prefill_requests", "num_waiting_prefill_tokens"
        # ]
        # df.to_csv(f"{directory}/full_scheduling_history.csv", index=False)

        # Save S1/S2 decision logs
        self.s1_s2_logger.stop_profile(directory)

        # Clear budget_schedule_log
        self.budget_schedule_log = []
        self.microbatch_schedule_log = []
        self.full_scheduling_history = []


    def register_microbatch_modification(self, new_num_decode_phases: int):
        if new_num_decode_phases > self.num_decode_phases:
            # If we are expanding decodable phases, we balance the requests out.
            # We run rebalancing mode for self.pp_size + 1 steps.
            self.to_apply_split_count = self.pp_size + 1
        else:
            self.to_apply_split_count = 0
        self.num_decode_phases = new_num_decode_phases
        
    def _previous_prefill_interrupt_moderation(self, ttft: float, per_chunk_latency: float):
        # time_since_last_prefill = time.monotonic() - self.last_prefill_timestamp
        # if time_since_last_prefill < self.last_prefill_ttft:
        #     leftover_time = self.last_prefill_ttft - time_since_last_prefill
        #     overlap_time = max(0, leftover_time - per_chunk_latency * (self.pp_size))
        #     ttft = ttft + overlap_time
        return float(ttft)

    def _previous_prefill_interrupt_moderation_for_decode(self, tbt: float, per_chunk_latency: float):
        # time_since_last_prefill = time.monotonic() - self.last_prefill_timestamp
        # if time_since_last_prefill < self.last_prefill_ttft:
        #     leftover_time = self.last_prefill_ttft - time_since_last_prefill
        #     overlap_time = max(0, leftover_time - per_chunk_latency * (self.pp_size - 1))
        #     tbt = tbt + overlap_time
        return float(tbt)

    def beam_schedule_tbt_based_s1(self, tbt_slo: float, ttft_slo: float, num_waiting_prefill_tokens: int, use_s1_dvfs_only: bool, new_requests: list[Request], disagg_mode = False):
        # Scheduling algorithm that caps TBT.
        
        min_energy_chunk_size = None
        min_energy_clock = None
        min_energy = float('inf')
        min_ttft = float('inf')
        
        has_ttft_adhering_choices = False
        has_tbt_adhering_choices = False
        
        chunk_sizes = self.energy_simulator.available_chunk_sizes
        clocks = self.energy_simulator.available_clocks
        
        # if use_s1_dvfs_only, we fix chunk_size choice to 256
        # When measuring TTFT, we also need to consider queuing time.
        now = time.time()
        # For each new_requests, compute max queuing time.
        max_queuing_time = float('-inf')
        for req in new_requests:
            queuing_time = now - req.arrival_time
            if queuing_time > max_queuing_time:
                max_queuing_time = queuing_time
        max_queuing_time = max(max_queuing_time, 0.0)
        logger.info(f"Max queuing time: {max_queuing_time}")

        time_since_last_prefill = time.monotonic() - self.last_prefill_timestamp
        # If so, limit chunk size choices to 1 above, same, or 1 below.
        # Drastic chunk-size changes create bubbles.
        if time_since_last_prefill < self.last_prefill_ttft:
            # Get current chunk size
            current_chunk_size = self.max_num_scheduled_tokens
            # Find the index of current chunk size in available sizes
            available_chunks = self.energy_simulator.available_chunk_sizes
            if current_chunk_size in available_chunks:
                current_idx = available_chunks.index(current_chunk_size)
                # Limit to adjacent chunk sizes (1 below, same, 1 above)
                start_idx = max(0, current_idx - 1)
                end_idx = min(len(available_chunks), current_idx + 2)
                chunk_sizes = available_chunks[start_idx:end_idx]
                logger.info(f"Prefill in progress: limiting chunk_sizes from {available_chunks} to {chunk_sizes}")

        if use_s1_dvfs_only:
            chunk_sizes = [256]
            

        for chunk_size in chunk_sizes:
            for clock in clocks:
                per_chunk_energy = self.energy_simulator.get_energy(chunk_size, clock)
                per_chunk_latency = self.energy_simulator.get_time_taken(chunk_size, clock)
                avg_decodes_per_chunk = len(self.running) // self.pp_size
                num_chunks_floor = math.floor(num_waiting_prefill_tokens / (chunk_size - avg_decodes_per_chunk)) * self.pp_size
                num_ttft_chunks_floor = math.floor(num_waiting_prefill_tokens / (chunk_size - avg_decodes_per_chunk)) + (self.pp_size - 1)
                
                
                if chunk_size >= num_waiting_prefill_tokens:
                    # Single chunk can handle entire request
                    per_chunk_energy = self.energy_simulator.get_energy(num_waiting_prefill_tokens, clock)
                    per_chunk_latency = self.energy_simulator.get_time_taken(num_waiting_prefill_tokens, clock)
                    
                    tbt = (per_chunk_latency + self.energy_simulator.scheduling_overhead) * self.pp_size
                    ttft = (per_chunk_latency + self.energy_simulator.scheduling_overhead) * self.pp_size
                    ttft = self._previous_prefill_interrupt_moderation(ttft, per_chunk_latency) + max_queuing_time
                    energy = per_chunk_energy * self.pp_size
                    
                    ttft_adhering = ttft <= ttft_slo
                    if disagg_mode: 
                        tbt_adhering = True
                    else:
                        tbt_adhering = tbt <= tbt_slo
                    
                else:
                    # Multiple chunks needed
                    num_leftover_tokens = num_waiting_prefill_tokens % chunk_size
                    leftover_chunk_energy = self.energy_simulator.get_energy(num_leftover_tokens, clock)
                    leftover_chunk_latency = self.energy_simulator.get_time_taken(num_leftover_tokens, clock)
                    
                    tbt = (per_chunk_latency + self.energy_simulator.scheduling_overhead) * self.pp_size
                    ttft = num_ttft_chunks_floor * (per_chunk_latency + self.energy_simulator.scheduling_overhead) + leftover_chunk_latency
                    ttft = self._previous_prefill_interrupt_moderation(ttft, per_chunk_latency) + max_queuing_time
                    energy = per_chunk_energy * num_chunks_floor + leftover_chunk_energy * self.pp_size
                                        
                    ttft_adhering = ttft <= ttft_slo
                    if disagg_mode: 
                        tbt_adhering = True
                    else:
                        tbt_adhering = tbt <= tbt_slo
                
                if ttft_adhering:
                    has_ttft_adhering_choices = True
                if tbt_adhering:
                    has_tbt_adhering_choices = True
                    
                if ttft_adhering and tbt_adhering and energy < min_energy:
                    min_energy = energy
                    min_energy_chunk_size = chunk_size
                    min_energy_clock = clock
                    min_ttft = ttft
                    min_tbt = tbt

        self.last_prefill_timestamp = time.monotonic()
        if min_energy_chunk_size is None or min_energy_clock is None:
            # If no tbt_adhering, use chunk-size 128, if no ttft_adhering, choose 512
            if not has_tbt_adhering_choices:
                logger.info("!!! No tbt_adhering choices, using chunk-size 128")
                chunk_size = 128
            if not has_ttft_adhering_choices:
                logger.info("!!! No ttft_adhering choices, using chunk-size 256")
                chunk_size = 256 # Default size for vLLM. Has precedence.
            
            # Pick default chunk x clock
            num_ttft_chunks = math.ceil(num_waiting_prefill_tokens / chunk_size) + (self.pp_size - 1)
            min_ttft = num_ttft_chunks * (self.energy_simulator.get_time_taken(chunk_size, self.energy_simulator.default_clock_high) + self.energy_simulator.scheduling_overhead)
            self.last_prefill_ttft = min_ttft
            return chunk_size, self.energy_simulator.default_clock_high

        else:
            self.last_prefill_ttft = min_ttft
            return min_energy_chunk_size, min_energy_clock
    
    def beam_schedule_tbt_based_s2(self, tbt_slo: float, num_running_reqs: int):
        # Scheduling algorithm that caps TBT. In decode-only traces.
        min_energy_num_microbatches = None
        min_energy_clock = None
        min_energy = float('inf')

        clocks = self.energy_simulator.available_clocks
        
        # Get total context length of all running requests
        total_ctx_len = 0
        for req in self.running:
            total_ctx_len += req.num_tokens_with_spec
        # Iterate all num_microbatches x clock to find the optimal num_microbatches x clock that can cap TBT.
        for num_microbatches in range(1, self.pp_size + 1):
            for clock in clocks:
                
                ctx_len_per_microbatch = total_ctx_len / num_microbatches
                
                # Get per-microbatch energy and latency
                num_reqs_per_microbatch = math.ceil(num_running_reqs / num_microbatches)
                # TBT is bound to individual microbatch
                tbt = (self.energy_simulator.get_time_taken(num_reqs_per_microbatch, clock, ctx_len_per_microbatch) + self.energy_simulator.scheduling_overhead_decode_only) * self.pp_size
                # Energy is bound to all microbatches
                energy = self.energy_simulator.get_energy(num_reqs_per_microbatch, clock) * num_microbatches
                
                if tbt <= tbt_slo and energy < min_energy:
                    min_energy = energy
                    min_energy_num_microbatches = num_microbatches
                    min_energy_clock = clock

        if min_energy_num_microbatches is None or min_energy_clock is None:
            return self.pp_size, self.energy_simulator.default_clock_high

        return min_energy_num_microbatches, min_energy_clock
    
    def beam_schedule_tpot_based_s1_immediate(self, tpot_slo: float, num_waiting_prefill_tokens: int, ttft_slo: float):
        """
        S1-immediate algorithm. TPOT based
        * Chooses max chunk-size such that every TPOT of each request <= TPOT SLO.
        
        """
        # Choose k* request
        k_star_request = None
        k_star_tbt_list = []
        max_tbt_sum = 0
        for req in self.running:
            req_tbt_list = req._output_token_tbts
            req_tbt_sum = sum(req_tbt_list)
            if req_tbt_sum > max_tbt_sum:
                max_tbt_sum = req_tbt_sum
                k_star_request = req
                k_star_tbt_list = req_tbt_list
        if k_star_request is None:
            return self.energy_simulator.default_chunk_size, self.energy_simulator.default_clock

        # Iterate chunk x clock 
        min_energy_chunk_size = None
        min_energy_clock = None
        min_energy = float('inf')
        min_ttft = float('inf')
        for chunk in self.energy_simulator.available_chunk_sizes:
            for clock in self.energy_simulator.available_clocks:
                # Get per-chunk energy and latency
                per_chunk_energy = self.energy_simulator.get_energy(chunk, clock)
                per_chunk_latency = self.energy_simulator.get_time_taken(chunk, clock)
                # Based on SPP, also consider PP delay
                num_chunks = math.ceil(num_waiting_prefill_tokens / chunk) * self.pp_size
                num_ttft_chunks = math.ceil(num_waiting_prefill_tokens / chunk) + (self.pp_size - 1)
                num_tbt_interrupts = math.ceil(num_ttft_chunks / self.pp_size)
                
                # Given num_ttft_chunks interrupts, compute updated k_star_tbt_list
                updated_k_star_tpot = (max_tbt_sum + num_tbt_interrupts * per_chunk_latency) / (len(k_star_tbt_list) + num_tbt_interrupts - 1)
                
                # Compute the TTFT and TBT for this chunk x clock.
                ttft = num_ttft_chunks * (per_chunk_latency + self.energy_simulator.scheduling_overhead)
                
                # If previous prefill is not finished, leftover time may determine ttft.
                time_since_last_prefill = time.monotonic() - self.last_prefill_timestamp
                if time_since_last_prefill < self.last_prefill_ttft:
                    leftover_time = self.last_prefill_ttft - time_since_last_prefill
                    leftover_plus_chunk = leftover_time + per_chunk_latency 
                    ttft = max(ttft, leftover_plus_chunk)
                
                energy = per_chunk_energy * num_chunks
                
                if updated_k_star_tpot <= tpot_slo and energy < min_energy and ttft <= ttft_slo:
                    min_energy = energy
                    min_energy_chunk_size = chunk
                    min_energy_clock = clock
                    min_ttft = ttft

        self.last_prefill_timestamp = time.monotonic()

        if min_energy_chunk_size is None or min_energy_clock is None:
            num_ttft_chunks = math.ceil(num_waiting_prefill_tokens / self.energy_simulator.default_chunk_size) + (self.pp_size - 1)
            min_ttft = num_ttft_chunks * (self.energy_simulator.get_time_taken(self.energy_simulator.default_chunk_size, self.energy_simulator.default_clock) + self.energy_simulator.scheduling_overhead)
            self.last_prefill_ttft = min_ttft
            return self.energy_simulator.default_chunk_size, self.energy_simulator.default_clock
        else:
            self.last_prefill_ttft = min_ttft
            return min_energy_chunk_size, min_energy_clock


    def energy_schedule_dynamollm_like(self):
        """
        DynamoLLM-style energy scheduling using ClockSelector.
        Called every 5 seconds to dynamically adjust GPU clock frequency
        based on current workload and SLO constraints.
        """
        if not hasattr(self, 'clock_selector') or self.clock_selector is None:
            logger.warning(
                "ClockSelector not initialized for dynamo scheduling")
            return

        # Calculate actual TPS based on tokens processed since last calculation
        current_time = time.monotonic()
        time_elapsed = current_time - self.last_dynamo_schedule_time

        if time_elapsed > 0:
            instantaneous_tps = self.tokens_since_last_calculation / time_elapsed
            # Update exponential moving average
            if self.current_tps_ema == 0.0:
                # First measurement, initialize EMA with instantaneous value
                self.current_tps_ema = instantaneous_tps
            else:
                # EMA formula: new_ema = alpha * current_value + (1 - alpha) * old_ema
                self.current_tps_ema = (
                    self.tps_ema_alpha * instantaneous_tps +
                    (1 - self.tps_ema_alpha) * self.current_tps_ema)
            actual_tps = self.current_tps_ema
        else:
            actual_tps = self.current_tps_ema  # Use previous EMA value if no time elapsed

        # Reset counter for next interval
        self.tokens_since_last_calculation = 0


        # Get optimal clock choice from dynamo energy model
        # Convert actual_tps to int for clock selection
        clock_choice = self.clock_selector.choose(tp_degree=self.tp_size,
                                                  pp_degree=self.pp_size,
                                                  target_tps=int(actual_tps))
        
        logger.info(f"Scheduler: dynamo {clock_choice}")

        # Check if a valid clock choice was found
        if clock_choice is None:
            logger.warning(
                f"Scheduler: No valid clock configuration found for "
                f"tp={self.tp_size}, pp={self.pp_size}, target_tps={actual_tps:.2f}. "
                f"Keeping current clock configuration."
            )
            return

        # Change relevant parameters
        self.prefill_clock = clock_choice.freq_mhz
        self.decode_clock = clock_choice.freq_mhz
        # use apply_dvfs to apply the clock choice
        apply_dvfs(clock_choice.freq_mhz, self.tp_size * self.pp_size)
        
        # Some good looking debugging messages
        logger.info(
            f"Scheduler: dynamo energy model scheduling: "
            f"actual_tps={actual_tps:.2f}, "
            f"prefill_clock={self.prefill_clock}, "
            f"decode_clock={self.decode_clock}"
        )

        # Update timing for next calculation
        self.last_dynamo_schedule_time = current_time

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.
        
        # Track schedule() execution time
        schedule_start_time = time.time()
        
        self.step_cnt += 1  # (hj) Added for profiling purposes
        
        # Log KV cache usage
        self.s1_s2_logger.log_kv_usage(self.kv_cache_manager.usage)

        self.phase = self.step_cnt % self.pp_size
        
        has_prefill_requests = False
        
        if self.to_apply_split_count > 0:
            self.to_apply_split_count -= 1
        
        
        # allow_decode_requests = self.phase < self.num_decode_phases
        if self.use_s2:
            allow_decode_requests = self.phase < self.num_decode_phases
        else:
            allow_decode_requests = True

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()
        
        # HJ : for debugging (fidelity test)
        new_requests = []

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if not allow_decode_requests:
                if request.num_computed_tokens > len(request.prompt_token_ids) + 1:
                    req_index += 1
                    continue
                
            if self.to_apply_split_count > 0:
                # Decode filter
                if request.num_computed_tokens > len(request.prompt_token_ids) + 1:
                    dest_phase = req_index % self.num_decode_phases
                    if dest_phase != self.phase:
                        req_index += 1
                        continue
            
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len or
            # request's max_tokens.
            # This is necessary when using spec decoding and/or async scheduling.
            max_total_tokens = min(
                request.num_prompt_tokens + request.max_tokens, self.max_model_len
            )
            num_new_tokens = min(
                num_new_tokens, max_total_tokens - 1 - request.num_computed_tokens
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens[
                                preempted_req.request_id
                            ]
                            req_to_new_blocks.pop(preempted_req.request_id)
                            num_scheduled_tokens.pop(preempted_req.request_id)
                            scheduled_spec_decode_tokens.pop(
                                preempted_req.request_id, None
                            )
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req.request_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_tokens_to_restore = sum(
                                    preempted_req.get_num_encoder_tokens(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_tokens_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    preempted_req.num_preemptions += 1
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp
                        )

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            
            if num_new_tokens > 1:
                has_prefill_requests = True
                new_requests.append(request)
            
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )
                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule
                )
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            # HJ : restricted wait scheduling (aka prefill) entry to align with decode to prevent unexpected TBT jumps
            # while self.waiting and token_budget > 0 and allow_decode_requests:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        num_external_computed_tokens = ext_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                if self.is_encoder_decoder and request.has_encoder_inputs:
                    # TODO(russellb): For Whisper, we know that the input is
                    # always padded to the maximum length. If we support other
                    # encoder-decoder models, this will need to be updated if we
                    # want to only allocate what is needed.
                    num_encoder_tokens = (
                        self.scheduler_config.max_num_encoder_input_tokens
                    )
                else:
                    num_encoder_tokens = 0

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )
                    self._update_connector_prefix_cache_stats(
                        request, num_external_computed_tokens
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                if num_new_tokens > 1:
                    has_prefill_requests = True
                    new_requests.append(request)

                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        
        # Track tokens for DynamoLLM-style scheduling
        if self.use_window_based_dvfs_only:
            self.tokens_since_last_calculation += total_num_scheduled_tokens
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request = self.running[0]
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(
                        any_request.request_id
                    )
                )

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids()
            )
            for req in scheduled_new_reqs
        ]
        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())
        if self.measure_fidelity:
            if has_prefill_requests:
                chunk_size = total_num_scheduled_tokens
                expected_tbt = (self.energy_simulator.get_time_taken(chunk_size, self.prefill_clock) + self.energy_simulator.scheduling_overhead) * self.pp_size
                self.expected_tbt = float(expected_tbt)
            else:
                total_ctx_len = 0
                for req in self.running:
                    total_ctx_len += req.num_tokens_with_spec
                
                num_running = len(self.running)
                num_microbatches = self.optimal_microbatch_size
                exp_ctx_len = total_ctx_len / num_microbatches
                exp_batch_size_per_microbatch = math.ceil(num_running / num_microbatches)
                
                chunk_time = self.energy_simulator.get_time_taken(exp_batch_size_per_microbatch, self.decode_clock, exp_ctx_len)
                expected_tbt = (chunk_time + self.energy_simulator.scheduling_overhead_decode_only) * self.pp_size
                
                # Get previous chunk-size in case prefill is used
                previous_chunk_size = self.max_num_scheduled_tokens
                previous_chunk_time = self.energy_simulator.get_time_taken(previous_chunk_size, self.prefill_clock)
                expected_tbt = self._previous_prefill_interrupt_moderation_for_decode(expected_tbt, previous_chunk_time)
                self.expected_tbt = float(expected_tbt)

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            expected_tbt=self.expected_tbt,
        )
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            if not self.requests[req_id].prefill_is_finished:
                self.num_waiting_prefill_tokens -= num_scheduled_token
                
        if has_prefill_requests:
            self.num_consecutive_decode_only = 0
        else:
            self.num_consecutive_decode_only += 1


        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta
        if self.use_window_based_dvfs_only and time.monotonic() - self.last_dynamo_schedule_time > DYNAMOLLM_DVFS_RESCHEDULE_INTERVAL:
            self.energy_schedule_dynamollm_like()
        
        if self.new_request_arrived and self.energy_simulator is not None:
            self.new_request_arrived = False

            self.register_microbatch_modification(self.pp_size)
            if self.use_s1:
                s1_start = time.time()
                optimal_chunk_size, optimal_clock = self.beam_schedule_tbt_based_s1(
                    self.tbt_slo, self.ttft_slo, self.num_waiting_prefill_tokens, self.use_s1_dvfs_only, new_requests
                )
                s1_end = time.time()
                self.s1_s2_logger.log_s1_exec_time(s1_end - s1_start)
                # update chunk_size
                self.max_num_scheduled_tokens = optimal_chunk_size
                # update clock
                self.prefill_clock = optimal_clock
                apply_dvfs(optimal_clock, self.tp_size * self.pp_size)
            
                # HJ : for debugging (fidelity test) Register estimated TTFTs for each new request
                chunk_latency = self.energy_simulator.get_time_taken(optimal_chunk_size, optimal_clock)
                
                if self.measure_fidelity:
                    
                    # Get max queuing time (as we do)
                    now = time.time()
                    max_queuing_time = float('-inf')
                    for req in new_requests:
                        queuing_time = now - req.arrival_time
                        if queuing_time > max_queuing_time:
                            max_queuing_time = queuing_time
                    max_queuing_time = max(max_queuing_time, 0.0)
                    
                    for req in new_requests:
                        prefill_size = len(req.prompt_token_ids)
                        avg_decodes_per_chunk = len(self.running) // self.pp_size
                        num_ttft_chunks_ceil = math.ceil(prefill_size / (optimal_chunk_size - avg_decodes_per_chunk)) + (self.pp_size - 1)
                        expected_ttft = num_ttft_chunks_ceil * (chunk_latency + self.energy_simulator.scheduling_overhead)
                        expected_ttft = self._previous_prefill_interrupt_moderation(expected_ttft, chunk_latency) + max_queuing_time
                        req._estimated_ttft = expected_ttft
                        logger.info(f"request {req.request_id} estimated TTFT: {expected_ttft}")
                
                # Log results (more readable, show more metrics)
                logger.info(
                    "\n".join([
                        "Scheduler : beam_schedule_tbt_based_s1",
                        f"  tbt slo = {self.tbt_slo}",
                        f"  optimal_chunk_size = {optimal_chunk_size}",
                        f"  optimal_clock = {optimal_clock}",
                        f"  num_prefill_tokens_to_process = {self.num_waiting_prefill_tokens}",
                    ])
                )
                self.s1_s2_logger.log_s1_decision(optimal_chunk_size)
                # 

            if self.use_s2:
                num_running = len(self.running)
                
                s2_start = time.time()
                optimal_num_microbatches, optimal_clock = self.beam_schedule_tbt_based_s2(
                    self.tbt_slo, num_running
                )
                s2_end = time.time()
                self.s1_s2_logger.log_s2_exec_time(s2_end - s2_start)
                # update microbatch_size
                self.optimal_microbatch_size = optimal_num_microbatches
                self.register_microbatch_modification(self.optimal_microbatch_size)
                # update clock
                self.decode_clock = optimal_clock

                total_ctx_len = 0
                for req in self.running:
                    total_ctx_len += req.num_tokens_with_spec
                
                # Calculate expected decode TBT for estimation
                num_reqs_per_microbatch = math.ceil(num_running / optimal_num_microbatches)
                

                # Log results (more readable, show more metrics)
                logger.info(
                    "\n".join([
                        "Scheduler : beam_schedule_tbt_based_s2",
                        f"  optimal_num_microbatches = {optimal_num_microbatches}",
                        f"  total_ctx_len = {total_ctx_len}",
                        f"  optimal_clock = {optimal_clock}",
                        f"  num_running = {num_running}",
                    ])
                )
                # Log S2 decision
                self.s1_s2_logger.log_s2_decision(optimal_num_microbatches)

        # Decode-only clock setter
        if self.energy_simulator is not None and \
        self.num_consecutive_decode_only == NUM_CONSECUTIVE_DECODE_ONLY_TO_SET_CLOCK:
            logger.info(f"Scheduler: Setting decode clock to {self.decode_clock}")
            # Apply S2-related stuff after decode-only clock setting
            # self.register_microbatch_modification(self.optimal_microbatch_size)
            apply_dvfs(self.decode_clock, self.tp_size * self.pp_size)

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        
        # Log schedule() execution time
        schedule_end_time = time.time()
        self.s1_s2_logger.log_schedule_exec_time(schedule_end_time - schedule_start_time)
        
        return scheduler_output

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = num_scheduled_tokens[req_id] - len(
                spec_decode_tokens.get(req_id, ())
            )
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # Check remote cache first
        if self.ec_connector is not None:
            remote_cache_has_item = self.ec_connector.has_caches(request)
        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_tokens_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if request.mm_features[i].identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_tokens_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            if self.ec_connector is not None and remote_cache_has_item[i]:
                mm_hashes_to_schedule.add(request.mm_features[i].identifier)
                external_load_encoder_input.append(i)
                num_tokens_to_schedule += num_encoder_tokens
                continue

            num_tokens_to_schedule += num_encoder_tokens
            encoder_compute_budget -= num_encoder_tokens
            mm_hashes_to_schedule.add(request.mm_features[i].identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self,
        scheduler_output: SchedulerOutput,
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        # PERF: in case of chunked prefill,
        # request might not include any new tokens.
        # Therefore, we might introduce some additional
        # cycle to fill in the bitmask, which could be a big no-op.
        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id)) and req.use_structured_output
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids
            )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids: list[int] = (
                sampled_token_ids[req_index].tolist() if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len, pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                struct_output_request.grammar.accept_tokens(req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                # Add EngineCoreOutput for this Request.
                # logger.info(f"Estimated TBT: {self.expected_tbt}, Estimated TTFT: {request._estimated_ttft}")
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        estimated_ttft=request._estimated_ttft,
                        estimated_tbt=scheduler_output.expected_tbt,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(spec_decoding_stats, kv_connector_stats)
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)
            request._output_token_estimated_tbts.append(self.expected_tbt)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(
        self,
        draft_token_ids: DraftTokenIds,
    ) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                    spec_token_ids
                )
            else:
                request.spec_token_ids = spec_token_ids

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        
        self.num_waiting_prefill_tokens += request.num_prompt_tokens
        self.new_request_arrived = True
        
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> dict[str, Any] | None:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats = self._make_connector_prefix_cache_stats()
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            kv_connector_stats=kv_connector_stats.data if kv_connector_stats else None,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> SpecDecodingStats | None:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def _update_connector_prefix_cache_stats(
        self, request: Request, num_external_tokens: int
    ) -> None:
        if self.connector_prefix_cache_stats is None:
            return

        self.connector_prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_external_tokens,
            preempted=request.num_preemptions > 0,
        )

    def _make_connector_prefix_cache_stats(self) -> PrefixCacheStats | None:
        if self.connector_prefix_cache_stats is None:
            return None
        stats = self.connector_prefix_cache_stats
        self.connector_prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
            num_computed_tokens = len(block_ids) * self.block_size
            # Handle the case where num request tokens less than one block.
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

            # Update the request state for scheduling.
            request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self, requests: Iterable[Request], invalid_block_ids: set[int]
    ) -> tuple[set[str], int]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests (for observability).
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # Async loading. If num_computed_tokens is set it implies we
                # already processed some block failures for it in a prior step
                req_num_computed_tokens = (
                    request.num_computed_tokens
                    if req_id in self.failed_recving_kv_req_ids
                    else len(req_block_ids) * self.block_size
                )
            else:
                # Sync loading. num_computed_tokens includes new tokens
                req_num_computed_tokens = request.num_cached_tokens

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                total_affected_tokens += (
                    req_num_computed_tokens - request.num_computed_tokens
                )

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - request.num_cached_tokens
                    )
                    request.num_computed_tokens = request.num_cached_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        total_requests_to_reschedule = 0
        total_tokens_to_reschedule = 0

        # --- Handle async KV loads (WAITING_FOR_REMOTE_KVS) ---
        async_load_reqs = (
            req
            for req in self.waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_affected_req_ids, num_tokens_to_reschedule = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs, invalid_block_ids
            )
        )

        total_requests_to_reschedule += len(async_affected_req_ids)
        total_tokens_to_reschedule += num_tokens_to_reschedule

        # Mark requests with async KV load failures; they will be rescheduled
        # once loading completes.
        self.failed_recving_kv_req_ids |= async_affected_req_ids

        # --- Handle sync KV loads (running requests) ---
        sync_affected_req_ids, num_tokens_to_reschedule = (
            self._update_requests_with_invalid_blocks(self.running, invalid_block_ids)
        )

        total_requests_to_reschedule += len(sync_affected_req_ids)
        total_tokens_to_reschedule += num_tokens_to_reschedule

        if total_requests_to_reschedule:
            logger.warning(
                "Recovered from KV load failure: "
                "%d request(s) rescheduled (%d tokens affected).",
                total_requests_to_reschedule,
                total_tokens_to_reschedule,
            )

        # Return the IDs of affected running requests to skip in
        # update_from_output.
        return sync_affected_req_ids
