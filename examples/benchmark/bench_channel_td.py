# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from queue import Empty as QueueEmpty
from typing import Any

import torch
from ray.util.queue import Queue as RayQueue

from rlinf.scheduler import Channel, Cluster, PackedPlacementStrategy, Worker

# Available test names for --tests selection.
AVAILABLE_TESTS = frozenset(
    {
        "channel",  # Channel under pressure (sync + async)
    }
)

# Payload types for --payload-type.
PAYLOAD_TYPES = frozenset(
    {
        "tensor_dict",
    }
)

# Size units (binary: 1024-based)
_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}


def resolve_worker_ranks(placement: str) -> tuple[int, int]:
    if placement == "same":
        return 0, 0
    if placement == "cross":
        return 0, 1
    raise ValueError(f"Unknown placement: {placement}")


def parse_size(s: str | int) -> int:
    """Parse a size string with optional unit (B, KB, MB, GB) into bytes.

    Examples: "1024", "1KB", "64KB", "1MB", "1GB"
    """
    if isinstance(s, int):
        return s
    s = str(s).strip().upper()
    if not s:
        raise ValueError("Empty size string")
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([KMG]?B?)$", s)
    if not m:
        raise ValueError(f"Invalid size format: {s!r}. Use e.g. 1024, 1KB, 1MB, 1GB")
    num_str, unit = m.groups()
    num = float(num_str)
    unit = unit or "B"
    if unit == "K":
        unit = "KB"
    elif unit == "M":
        unit = "MB"
    elif unit == "G":
        unit = "GB"
    if unit not in _SIZE_UNITS:
        raise ValueError(f"Unknown unit: {unit}. Use B, KB, MB, GB")
    return int(num * _SIZE_UNITS[unit])


@dataclass
class BenchmarkConfig:
    num_messages: int = 2000
    num_warmup_messages: int = 2
    payload_size: int = 1024 * 1024  # bytes (derived for tensor_dict payload)
    channel_maxsize: int = 0
    enable_thread_interference: bool = False
    num_noise_threads: int = 2
    payload_type: str = "tensor_dict"
    payload_device: str = "auto"  # "auto" (npu if available else cpu) | "cpu" | "npu"
    batch_size: int = 32
    seq_len: int = 128
    field_num: int = 8
    num_test_iterations: int = 4
    desc: str = "debug"
    placement: str = "same"
    stats_json_path: str | None = None
    enabled_tests: frozenset[str] = field(default_factory=lambda: AVAILABLE_TESTS)
    enable_ray_queue: bool = False  # Run ray.util.queue.Queue comparison for same tests


# Predefined desc -> (batch_size, seq_len, field_num) configurations.
DESC_CONFIGS: dict[str, tuple[int, int, int]] = {
    "debug": (32, 128, 2),
    "tiny": (64, 1024, 4),
    "small": (512, 12800, 4),
    "medium": (1024, 65536, 4),
    "large": (2048, 128000, 5),
    "xlarge": (4096, 128000, 5),
    "huge": (4096, 128000, 10),
}


def _npu_is_available() -> bool:
    """Safely check whether torch.npu is available in this PyTorch build."""
    if not hasattr(torch, "npu"):
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


def _resolve_device(device: str) -> str:
    """Resolve device: 'auto' -> 'npu' if available else 'cpu'; 'cpu'/'npu' as-is."""
    if device == "auto":
        return "npu" if _npu_is_available() else "cpu"
    if device == "npu" and not _npu_is_available():
        raise RuntimeError("Device npu requested but NPU is not available")
    return device


def _create_tensor_payload(
    payload_type: str,
    batch_size: int,
    seq_len: int,
    field_num: int,
    payload_device: str = "auto",
) -> (
    dict[str, torch.Tensor]
):
    """Create tensor_dict payload with fixed structure."""
    if payload_type != "tensor_dict":
        raise ValueError(f"Unknown payload_type: {payload_type}")
    if batch_size <= 0 or seq_len <= 0 or field_num <= 0:
        raise ValueError("batch_size, seq_len and field_num must be positive")
    payload_device = _resolve_device(payload_device)
    torch_device = torch.device(payload_device)
    return {
        f"field_{i}": torch.randn(
            batch_size, seq_len, dtype=torch.float32, device=torch_device
        ).contiguous()
        for i in range(field_num)
    }


class Producer(Worker):
    def __init__(self):
        super().__init__()
        self._noise_started = False

    def _get_payload(self, cfg: BenchmarkConfig) -> Any:
        """Create payload based on config payload_type and device."""
        return _create_tensor_payload(
            cfg.payload_type,
            cfg.batch_size,
            cfg.seq_len,
            cfg.field_num,
            cfg.payload_device,
        )

    def start_cpu_noise(self, cfg: BenchmarkConfig) -> None:
        """Optionally start background CPU-burning threads on this worker."""
        if self._noise_started or not cfg.enable_thread_interference:
            return
        self._noise_started = True

        def _burn():
            while True:
                x = 0.0
                for _ in range(100_000):
                    x += 1.0

        for _ in range(max(1, cfg.num_noise_threads)):
            t = threading.Thread(target=_burn, daemon=True)
            t.start()

    def warmup(self, channel: Channel, cfg: BenchmarkConfig, async_mode: bool) -> None:
        """Un-timed warmup for puts to build up channel state and JITs."""
        device = torch.device(_resolve_device(cfg.payload_device))
        payload = torch.randn(1, dtype=torch.float32, device=device)
        if async_mode:

            async def _warmup() -> None:
                for _ in range(cfg.num_warmup_messages):
                    work = channel.put(payload, async_op=True)
                    await work.async_wait()

            asyncio.run(_warmup())
        else:
            for _ in range(cfg.num_warmup_messages):
                channel.put(payload)

    def run_sync(self, channel: Channel, cfg: BenchmarkConfig) -> float:
        """Synchronous put: each put blocks until finished."""
        payload = self._get_payload(cfg)
        with self.worker_timer("producer_sync"):
            for i in range(cfg.num_messages):
                channel.put(payload)
        return self.pop_execution_time("producer_sync")

    def run_async(self, channel: Channel, cfg: BenchmarkConfig) -> float:
        """Async put using asyncio: await put(..., async_op=True).async_wait()."""
        payload = self._get_payload(cfg)

        async def _run() -> None:
            for i in range(cfg.num_messages):
                work = channel.put(payload, async_op=True)
                await work.async_wait()

        with self.worker_timer("producer_async"):
            asyncio.run(_run())
        return self.pop_execution_time("producer_async")

    def run_sync_ray_queue(self, queue: Any, cfg: BenchmarkConfig) -> float:
        """Synchronous put on ray.util.queue.Queue."""
        if RayQueue is None:
            return 0.0
        payload = self._get_payload(cfg)
        with self.worker_timer("producer_sync_ray_queue"):
            for i in range(cfg.num_messages):
                queue.put(payload)
        return self.pop_execution_time("producer_sync_ray_queue")

    def run_async_ray_queue(self, queue: Any, cfg: BenchmarkConfig) -> float:
        """Async put on ray.util.queue.Queue (put_async returns coroutine)."""
        if RayQueue is None:
            return 0.0
        payload = self._get_payload(cfg)

        async def _run():
            for i in range(cfg.num_messages):
                await queue.put_async(payload)

        with self.worker_timer("producer_async_ray_queue"):
            asyncio.run(_run())
        return self.pop_execution_time("producer_async_ray_queue")


class Consumer(Worker):
    def __init__(self):
        super().__init__()
        self._noise_started = False

    def run_sync(self, channel: Channel, cfg: BenchmarkConfig) -> float:
        with self.worker_timer("consumer_sync"):
            for i in range(cfg.num_messages):
                _ = channel.get()
        return self.pop_execution_time("consumer_sync")

    def run_async(self, channel: Channel, cfg: BenchmarkConfig) -> float:
        """Async get using asyncio: await get(async_op=True).async_wait()."""

        async def _run() -> None:
            for i in range(cfg.num_messages):
                work = channel.get(async_op=True)
                await work.async_wait()

        with self.worker_timer("consumer_async"):
            asyncio.run(_run())
        return self.pop_execution_time("consumer_async")

    def warmup(self, channel: Channel, cfg: BenchmarkConfig, async_mode: bool) -> None:
        """Un-timed warmup for gets."""
        if async_mode:

            async def _warmup() -> None:
                for _ in range(cfg.num_warmup_messages):
                    work = channel.get(async_op=True)
                    await work.async_wait()

            asyncio.run(_warmup())
        else:
            for _ in range(cfg.num_warmup_messages):
                _ = channel.get()

    def run_sync_ray_queue(self, queue: Any, cfg: BenchmarkConfig) -> float:
        """Synchronous get on ray.util.queue.Queue."""
        if RayQueue is None:
            return 0.0
        with self.worker_timer("consumer_sync_ray_queue"):
            for i in range(cfg.num_messages):
                _ = queue.get()
        return self.pop_execution_time("consumer_sync_ray_queue")

    def run_async_ray_queue(self, queue: Any, cfg: BenchmarkConfig) -> float:
        """Async get on ray.util.queue.Queue (get_async returns coroutine)."""
        if RayQueue is None:
            return 0.0

        async def _run():
            for i in range(cfg.num_messages):
                _ = await queue.get_async()

        with self.worker_timer("consumer_async_ray_queue"):
            asyncio.run(_run())
        return self.pop_execution_time("consumer_async_ray_queue")

    def start_cpu_noise(self, cfg: BenchmarkConfig) -> None:
        """Optionally start background CPU-burning threads on this worker."""
        if self._noise_started or not cfg.enable_thread_interference:
            return
        self._noise_started = True

        def _burn():
            while True:
                x = 0.0
                for _ in range(100_000):
                    time.sleep(0.001)
                    x += 1.0

        for _ in range(max(1, cfg.num_noise_threads)):
            t = threading.Thread(target=_burn, daemon=True)
            t.start()


def run_benchmark(cfg: BenchmarkConfig) -> None:
    # Initialize a single-node cluster first so Channel/Workers share the same Ray cluster.
    cluster = Cluster(num_nodes=1)

    producer_rank, consumer_rank = resolve_worker_ranks(cfg.placement)
    placement = PackedPlacementStrategy(start_hardware_rank=producer_rank, end_hardware_rank=producer_rank)
    producer_group = Producer.create_group().launch(
        cluster=cluster, placement_strategy=placement, name="channel_perf_producer"
    )
    placement = PackedPlacementStrategy(start_hardware_rank=consumer_rank, end_hardware_rank=consumer_rank)
    consumer_group = Consumer.create_group().launch(
        cluster=cluster, placement_strategy=placement, name="channel_perf_consumer"
    )

    if cfg.enable_thread_interference:
        print(
            f"\n[Info] Enabling thread interference with "
            f"{cfg.num_noise_threads} background threads per worker."
        )
        producer_group.start_cpu_noise(cfg).wait()
        consumer_group.start_cpu_noise(cfg).wait()

    # Single shared channel for all tests.
    channel = Channel.create(
        name="pressure_demo_channel_shared",
        maxsize=cfg.channel_maxsize,
        distributed=False,
    )

    def reset_channel() -> None:
        """Drain all remaining messages from the shared channel."""
        from asyncio import QueueEmpty

        while True:
            try:
                channel.get_nowait()
            except QueueEmpty:
                break

    # Ray util queue for comparison (ray.util.queue.Queue).
    ray_queue = RayQueue(
        maxsize=cfg.channel_maxsize or 0,
        actor_options={
            "runtime_env": {
                "env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
            }
        },
    )

    def reset_ray_queue() -> None:
        """Drain ray.util.queue.Queue."""
        if ray_queue is None:
            return
        while True:
            try:
                ray_queue.get_nowait()
            except QueueEmpty:
                break

    def ray_queue_round(async_mode: bool) -> dict[str, float] | None:
        """Mixed put+get on ray.util.queue.Queue (same pattern as one_round)."""
        if ray_queue is None:
            return None
        reset_ray_queue()
        if async_mode:
            prod_res = producer_group.run_async_ray_queue(ray_queue, cfg)
        else:
            prod_res = producer_group.run_sync_ray_queue(ray_queue, cfg)
        put_time = prod_res.wait()[0]

        time.sleep(2)

        if async_mode:
            cons_res = consumer_group.run_async_ray_queue(ray_queue, cfg)
        else:
            cons_res = consumer_group.run_sync_ray_queue(ray_queue, cfg)
        get_time = cons_res.wait()[0]
        payload_size_gb = cfg.payload_size / float(_SIZE_UNITS["GB"])
        total_data_size_gb = payload_size_gb * float(cfg.num_messages)
        put_gbit_per_sec = (
            (total_data_size_gb * 8) / put_time if put_time > 0 else float("inf")
        )
        get_gbit_per_sec = (
            (total_data_size_gb * 8) / get_time if get_time > 0 else float("inf")
        )
        total_gbit_per_sec = (
            (total_data_size_gb * 16) / (put_time + get_time)
            if (put_time + get_time) > 0
            else float("inf")
        )
        return {
            "payload_size_gb": payload_size_gb,
            "total_data_size_gb": total_data_size_gb,
            "put_time": put_time,
            "get_time": get_time,
            "put_gbit_per_sec": put_gbit_per_sec,
            "get_gbit_per_sec": get_gbit_per_sec,
            "total_gbit_per_sec": total_gbit_per_sec,
        }

    def one_round(async_mode: bool) -> dict[str, float]:
        reset_channel()

        # Warmup both producer and consumer.
        producer_group.warmup(channel, cfg, async_mode).wait()
        consumer_group.warmup(channel, cfg, async_mode).wait()

        if async_mode:
            prod_res = producer_group.run_async(channel, cfg)
        else:
            prod_res = producer_group.run_sync(channel, cfg)
        put_time = prod_res.wait()[0]

        time.sleep(2)

        if async_mode:
            cons_res = consumer_group.run_async(channel, cfg)
        else:
            cons_res = consumer_group.run_sync(channel, cfg)
        get_time = cons_res.wait()[0]

        payload_size_gb = cfg.payload_size / float(_SIZE_UNITS["GB"])
        total_data_size_gb = payload_size_gb * float(cfg.num_messages)
        put_gbit_per_sec = (
            (total_data_size_gb * 8) / put_time if put_time > 0 else float("inf")
        )
        get_gbit_per_sec = (
            (total_data_size_gb * 8) / get_time if get_time > 0 else float("inf")
        )
        total_gbit_per_sec = (
            (total_data_size_gb * 16) / (put_time + get_time)
            if (put_time + get_time) > 0
            else float("inf")
        )
        return {
            "payload_size_gb": payload_size_gb,
            "total_data_size_gb": total_data_size_gb,
            "put_time": put_time,
            "get_time": get_time,
            "put_gbit_per_sec": put_gbit_per_sec,
            "get_gbit_per_sec": get_gbit_per_sec,
            "total_gbit_per_sec": total_gbit_per_sec,
        }

    def compute_stats(round_values: list[float]) -> dict[str, float]:
        vals = sorted(round_values)
        n = len(vals)
        if n == 0:
            return {"avg": 0.0, "max": 0.0, "min": 0.0, "p99": 0.0}
        p99_idx = max(0, min(n - 1, math.ceil(0.99 * n) - 1))
        return {
            "avg": sum(vals) / n,
            "max": vals[-1],
            "min": vals[0],
            "p99": vals[p99_idx],
        }

    def aggregate_rounds(round_results: list[dict[str, float]]) -> dict[str, Any]:
        if not round_results:
            return {"rounds": [], "stats": {}}
        metric_names = list(round_results[0].keys())
        stats: dict[str, Any] = {}
        for name in metric_names:
            if name in ("payload_size_gb", "total_data_size_gb"):
                stats[name] = round_results[0][name]
            else:
                stats[name] = compute_stats([r[name] for r in round_results])
        return {"rounds": round_results, "stats": stats}

    def run_iterations(
        title: str, fn: callable, async_mode: bool
    ) -> dict[str, Any] | None:
        rounds: list[dict[str, float]] = []
        for i in range(cfg.num_test_iterations):
            print(
                f"[Start] {title} ({'async' if async_mode else 'sync'}) "
                f"iteration {i + 1}/{cfg.num_test_iterations}"
            )
            result = fn(async_mode=async_mode)
            if result is None:
                return None
            rounds.append(result)
        return aggregate_rounds(rounds)

    enabled = cfg.enabled_tests

    # Results storage
    sync_stats = async_stats = None
    sync_rayq_stats = async_rayq_stats = None

    print(f"Running channel pressure benchmark with config: {cfg}")
    payload_info = f"payload_type: {cfg.payload_type}"
    if cfg.payload_type == "tensor_dict":
        payload_info += f", device: {cfg.payload_device}"
        payload_info += (
            f", tensor_dict(batch_size={cfg.batch_size}, "
            f"seq_len={cfg.seq_len}, field_num={cfg.field_num})"
        )
    payload_info += f", iterations: {cfg.num_test_iterations}"
    print(
        f"Enabled tests: {sorted(enabled)}, {payload_info}"
        + (
            ", ray_queue comparison enabled"
            if cfg.enable_ray_queue and ray_queue
            else ""
        )
    )

    # Ray util.queue.Queue comparison runs for same tests when enabled
    run_ray_queue = cfg.enable_ray_queue and ray_queue is not None

    # Channel under pressure
    if "channel" in enabled:
        sync_stats = run_iterations("Channel under pressure", one_round, async_mode=False)
        async_stats = run_iterations("Channel under pressure", one_round, async_mode=True)

    # Ray util.queue.Queue comparison for same tests when enabled.
    if run_ray_queue and "channel" in enabled:
        sync_rayq_stats = run_iterations(
            "Ray util.queue.Queue under pressure", ray_queue_round, async_mode=False
        )
        async_rayq_stats = run_iterations(
            "Ray util.queue.Queue under pressure", ray_queue_round, async_mode=True
        )

    def fmt(s: dict[str, float]) -> str:
        return (
            f"payload={s['payload_size_gb']:.6f} GB, "
            f"total_data={s['total_data_size_gb']:.6f} GB, "
            f"put_time={s['put_time']:.4f}s, "
            f"get_time={s['get_time']:.4f}s, "
            f"put={s['put_gbit_per_sec']:.3f} Gbit/s, "
            f"get={s['get_gbit_per_sec']:.3f} Gbit/s, "
            f"total={s['total_gbit_per_sec']:.3f} Gbit/s"
        )

    def print_round_details(result: dict[str, Any]) -> None:
        rounds = result["rounds"]
        for i, item in enumerate(rounds, start=1):
            print(f"  - round {i}: {fmt(item)}")

    def print_summary_stats(result: dict[str, Any]) -> None:
        stats = result["stats"]
        metric_order = [
            "payload_size_gb",
            "total_data_size_gb",
            "put_time",
            "get_time",
            "put_gbit_per_sec",
            "get_gbit_per_sec",
            "total_gbit_per_sec",
        ]
        print("  Summary:")
        for metric in metric_order:
            if metric not in stats:
                continue
            m = stats[metric]
            if metric in ("payload_size_gb", "total_data_size_gb"):
                print(f"    {metric}: {m:.6f}")
                continue
            if isinstance(m, dict) and {"avg", "max", "min", "p99"} <= m.keys():
                if metric.endswith("_gbit_per_sec"):
                    fmt_str = "{:.3f}"
                elif metric.endswith("_time"):
                    fmt_str = "{:.6f}"
                else:
                    fmt_str = "{:.6f}"
                print(
                    f"    {metric}: avg={fmt_str.format(m['avg'])}, "
                    f"max={fmt_str.format(m['max'])}, "
                    f"min={fmt_str.format(m['min'])}, "
                    f"p99={fmt_str.format(m['p99'])}"
                )

    def _append_stats_record(
        kind: str, result: dict[str, Any] | None, *, cfg: BenchmarkConfig
    ) -> None:
        """Append a single stats record to JSON file if configured."""
        if cfg.stats_json_path is None or result is None:
            return
        record = {
            "desc": cfg.desc,
            "placement": cfg.placement,
            "device": cfg.payload_device,
            "kind": kind,
            "num_messages": cfg.num_messages,
            "stats": result["stats"],
        }
        os.makedirs(os.path.dirname(cfg.stats_json_path), exist_ok=True)
        data: list[dict[str, Any]] = []
        if os.path.exists(cfg.stats_json_path):
            try:
                with open(cfg.stats_json_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        data = loaded
            except Exception:
                # If file is invalid, start fresh.
                data = []
        data.append(record)
        with open(cfg.stats_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    if sync_stats is not None:
        print("\n=== Channel under pressure (sync) ===")
        print_round_details(sync_stats)
        print_summary_stats(sync_stats)
        _append_stats_record("channel_sync", sync_stats, cfg=cfg)
    if async_stats is not None:
        print("\n=== Channel under pressure (async) ===")
        print_round_details(async_stats)
        print_summary_stats(async_stats)
        _append_stats_record("channel_async", async_stats, cfg=cfg)

    # Ray util.queue.Queue comparison results.
    if sync_rayq_stats is not None:
        print("\n=== Ray util.queue.Queue under pressure (sync) ===")
        print_round_details(sync_rayq_stats)
        print_summary_stats(sync_rayq_stats)
        _append_stats_record("ray_queue_sync", sync_rayq_stats, cfg=cfg)
    if async_rayq_stats is not None:
        print("\n=== Ray util.queue.Queue under pressure (async) ===")
        print_round_details(async_rayq_stats)
        print_summary_stats(async_rayq_stats)
        _append_stats_record("ray_queue_async", async_rayq_stats, cfg=cfg)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for benchmark configuration and test selection."""
    parser = argparse.ArgumentParser(
        description="Channel pressure benchmark: measure put/get throughput for bytes and tensors."
    )
    parser.add_argument(
        "--tests",
        "-t",
        type=str,
        default=None,
        help=(
            "Comma-separated list of tests to run. "
            f"Available: {', '.join(sorted(AVAILABLE_TESTS))}. "
            "Default: run all tests."
        ),
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="Print available test names and exit.",
    )
    parser.add_argument(
        "--num-messages",
        type=int,
        default=1,
        help="Number of messages (default: 1).",
    )
    parser.add_argument(
        "--desc",
        type=str,
        default="debug",
        choices=sorted(DESC_CONFIGS.keys()),
        help=(
            "Preset payload configuration name. One of: "
            f"{', '.join(sorted(DESC_CONFIGS.keys()))} (default: debug)."
        ),
    )
    parser.add_argument(
        "--placement",
        type=str,
        default="same",
        choices=("same", "cross"),
        help=(
            "Worker placement strategy: same -> producer@0, consumer@0; "
            "cross -> producer@0, consumer@1 (default: same)."
        ),
    )
    parser.add_argument(
        "--payload-type",
        type=str,
        default="tensor_dict",
        choices=sorted(PAYLOAD_TYPES),
        help=f"Payload type: {', '.join(sorted(PAYLOAD_TYPES))}.",
    )
    parser.add_argument(
        "--payload-device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "npu"),
        help=(
            "Device for tensor_dict payload: auto (npu if available else cpu), "
            "cpu, or npu (default: auto)."
        ),
    )
    parser.add_argument(
        "--num-test-iterations",
        type=int,
        default=4,
        help="Number of repeated benchmark rounds for each test (default: 4).",
    )
    parser.add_argument(
        "--channel-maxsize",
        type=int,
        default=0,
        help="Channel max size, 0 = unbounded (default: 0).",
    )
    parser.add_argument(
        "--enable-thread-interference",
        action="store_true",
        help="Enable background CPU-burning threads to stress test under interference.",
    )
    parser.add_argument(
        "--num-noise-threads",
        type=int,
        default=2,
        help="Number of noise threads per worker (default: 2).",
    )
    parser.add_argument(
        "--ray-queue",
        action="store_true",
        help="Run ray.util.queue.Queue comparison for `channel`.",
    )
    parser.add_argument(
        "--stats-json-path",
        type=str,
        default=None,
        help=(
            "If set, append benchmark statistics to the given JSON file. "
            "JSON entries will include desc/payload_device and summary stats."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the channel benchmark."""
    args = parse_args()

    if args.list_tests:
        print("Available tests:")
        for name in sorted(AVAILABLE_TESTS):
            print(f"  {name}")
        return

    enabled_tests = AVAILABLE_TESTS
    if args.tests is not None:
        requested = {s.strip() for s in args.tests.split(",") if s.strip()}
        invalid = requested - AVAILABLE_TESTS
        if invalid:
            raise SystemExit(
                f"Unknown test(s): {invalid}. Available: {', '.join(sorted(AVAILABLE_TESTS))}"
            )
        enabled_tests = requested

    if args.num_test_iterations <= 0:
        raise SystemExit("num_test_iterations must be positive.")
    if args.payload_device == "npu" and not _npu_is_available():
        raise SystemExit("Device npu requested but NPU is not available.")

    if args.desc not in DESC_CONFIGS:
        raise SystemExit(
            f"Unknown desc {args.desc!r}. Available: {', '.join(sorted(DESC_CONFIGS))}"
        )
    batch_size, seq_len, field_num = DESC_CONFIGS[args.desc]
    if batch_size <= 0 or seq_len <= 0 or field_num <= 0:
        raise SystemExit("batch_size, seq_len and field_num must be positive.")

    payload_size = batch_size * seq_len * field_num * 4

    cfg = BenchmarkConfig(
        num_messages=args.num_messages,
        num_warmup_messages=2,
        payload_size=payload_size,
        channel_maxsize=args.channel_maxsize,
        enable_thread_interference=args.enable_thread_interference,
        num_noise_threads=args.num_noise_threads,
        payload_type=args.payload_type,
        payload_device=args.payload_device,
        batch_size=batch_size,
        seq_len=seq_len,
        field_num=field_num,
        num_test_iterations=args.num_test_iterations,
        desc=args.desc,
        placement=args.placement,
        stats_json_path=args.stats_json_path,
        enabled_tests=enabled_tests,
        enable_ray_queue=args.ray_queue,
    )
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
