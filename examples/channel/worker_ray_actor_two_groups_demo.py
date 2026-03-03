import os
import time

import ray
import torch

from rlinf.scheduler.cluster import Cluster
from rlinf.scheduler.hardware import AcceleratorType
from rlinf.scheduler.worker import Worker


class ActorWithWorker:
    def __init__(
        self,
        group_name: str,
        rank: int,
        world_size: int,
        cluster_node_rank: int,
        node_group_label: str,
        accelerator_type: str,
        isolate_accelerator: bool = False,
    ):
        accel_type = AcceleratorType(accelerator_type)
        local_acc_rank = 0 if accel_type != AcceleratorType.NO_ACCEL else -1
        self._worker = Worker.attach_to_current_ray_actor(
            group_name=group_name,
            rank=rank,
            world_size=world_size,
            cluster_node_rank=cluster_node_rank,
            node_group_label=node_group_label,
            accelerator_type=accel_type,
            accelerator_model="",
            local_accelerator_rank=local_acc_rank,
            node_local_rank=0,
            node_local_world_size=1 if isolate_accelerator else world_size,
            local_hardware_ranks="",
            isolate_accelerator=isolate_accelerator,
            catch_system_failure=False,
        )

    def _get_device(self, device: str) -> torch.device:
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available in this actor.")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        raise ValueError(f"Unsupported device: {device}")

    def _make_payload(self, kind: str, device: torch.device):
        if kind == "tensor":
            return torch.arange(6, dtype=torch.int64, device=device).reshape(2, 3)
        if kind == "tensor_list":
            return [
                torch.full((2,), float(i), dtype=torch.float32, device=device)
                for i in range(3)
            ]
        if kind == "tensor_dict":
            return {
                "a": torch.arange(4, dtype=torch.int64, device=device).reshape(2, 2),
                "b": torch.ones((3,), dtype=torch.float32, device=device) * 3.0,
            }
        raise ValueError(f"Unsupported kind: {kind}")

    def send_payload(
        self, dst_group_name: str, dst_rank: int, *, device: str, kind: str
    ):
        dev = self._get_device(device)
        payload = self._make_payload(kind, dev)
        self._worker.send(payload, dst_group_name=dst_group_name, dst_rank=dst_rank)

    def recv_and_check(
        self, src_group_name: str, src_rank: int, *, device: str, kind: str
    ) -> str:
        dev = self._get_device(device)
        got = self._worker.recv(src_group_name=src_group_name, src_rank=src_rank)
        expected = self._make_payload(kind, dev)

        def _assert_tensor_equal(x: torch.Tensor, y: torch.Tensor):
            assert isinstance(x, torch.Tensor)
            assert x.device.type == y.device.type
            assert x.shape == y.shape
            assert x.dtype == y.dtype
            assert torch.equal(x.cpu(), y.cpu())

        if kind == "tensor":
            _assert_tensor_equal(got, expected)
        elif kind == "tensor_list":
            assert isinstance(got, list) and isinstance(expected, list)
            assert len(got) == len(expected)
            for x, y in zip(got, expected, strict=True):
                _assert_tensor_equal(x, y)
        elif kind == "tensor_dict":
            assert isinstance(got, dict) and isinstance(expected, dict)
            assert set(got.keys()) == set(expected.keys())
            for k in expected.keys():
                _assert_tensor_equal(got[k], expected[k])
        else:
            raise ValueError(f"Unsupported kind: {kind}")

        return f"ok(kind={kind}, device={device})"


ActorWithWorkerRemote = ray.remote(ActorWithWorker)


def main():
    # 1. 初始化单节点 Cluster，启动 Worker 所需的全局管理器。
    cluster = Cluster(num_nodes=1)

    # 2. 基本放置信息（单节点）。
    node_rank = 0
    node_group = cluster.get_node_group()
    assert node_group is not None, "Default node group not found in Cluster."
    node_group_label = node_group.label

    def run_roundtrip_two_groups(
        base_group_name_a: str,
        base_group_name_b: str,
        *,
        accelerator_type: str,
        use_gpu: bool,
    ):
        # 每个 Ray Actor 所在 group 的 world_size 都是 1，rank 也是 0。
        world_size = 1
        isolate = use_gpu

        group_name_a = f"{base_group_name_a}_{os.getpid()}_{time.time_ns()}"
        group_name_b = f"{base_group_name_b}_{os.getpid()}_{time.time_ns()}"

        actor_a = ActorWithWorkerRemote.options(
            name=f"{group_name_a}:0", **({} if not use_gpu else {"num_gpus": 1})
        ).remote(
            group_name=group_name_a,
            rank=0,
            world_size=world_size,
            cluster_node_rank=node_rank,
            node_group_label=node_group_label,
            accelerator_type=accelerator_type,
            isolate_accelerator=isolate,
        )

        actor_b = ActorWithWorkerRemote.options(
            name=f"{group_name_b}:0", **({} if not use_gpu else {"num_gpus": 1})
        ).remote(
            group_name=group_name_b,
            rank=0,
            world_size=world_size,
            cluster_node_rank=node_rank,
            node_group_label=node_group_label,
            accelerator_type=accelerator_type,
            isolate_accelerator=isolate,
        )

        device = "cuda" if use_gpu else "cpu"
        for kind in ["tensor", "tensor_list", "tensor_dict"]:
            # group A(rank 0) -> group B(rank 0)
            r = actor_b.recv_and_check.remote(
                src_group_name=group_name_a, src_rank=0, device=device, kind=kind
            )
            actor_a.send_payload.remote(
                dst_group_name=group_name_b, dst_rank=0, device=device, kind=kind
            )
            print("[A->B]", ray.get(r))

            # group B(rank 0) -> group A(rank 0)
            r = actor_a.recv_and_check.remote(
                src_group_name=group_name_b, src_rank=0, device=device, kind=kind
            )
            actor_b.send_payload.remote(
                dst_group_name=group_name_a, dst_rank=0, device=device, kind=kind
            )
            print("[B->A]", ray.get(r))

    # CPU 测试（必跑）
    run_roundtrip_two_groups(
        "demo_two_groups_cpu_A",
        "demo_two_groups_cpu_B",
        accelerator_type=AcceleratorType.NO_ACCEL.value,
        use_gpu=False,
    )

    # GPU 测试（可选）
    node0 = cluster.get_node_info(0)
    if (
        node0.num_accelerators >= 2
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
    ):
        run_roundtrip_two_groups(
            "demo_two_groups_gpu_A",
            "demo_two_groups_gpu_B",
            accelerator_type=node0.accelerator_type,
            use_gpu=True,
        )
    else:
        print(
            "Skip GPU tests: need >=2 GPUs visible (cluster + torch) to run NCCL-style send/recv."
        )


if __name__ == "__main__":
    main()

