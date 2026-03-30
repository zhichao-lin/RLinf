import os
import time

import ray
import torch

from rlinf.scheduler.cluster import Cluster
from rlinf.scheduler.hardware import AcceleratorType, AcceleratorUtil
from rlinf.scheduler.worker import Worker


def _actor_resource_options(
    accelerator_type: AcceleratorType, request_one: bool
) -> dict:
    """Ray scheduling: GPU uses num_gpus; Ascend NPU uses the registered NPU resource."""
    if not request_one:
        return {}
    if accelerator_type == AcceleratorType.NV_GPU:
        return {"num_gpus": 1}
    if accelerator_type == AcceleratorType.NPU:
        return {"resources": {"NPU": 1.0}}
    return {}


def _accel_device_token(accelerator_type: AcceleratorType) -> str:
    if accelerator_type == AcceleratorType.NV_GPU:
        return "cuda"
    if accelerator_type == AcceleratorType.NPU:
        return "npu"
    raise ValueError(f"No device token for accelerator type {accelerator_type!r}")


def _two_accel_tests_runnable(
    node0_num_accelerators: int, accelerator_type: AcceleratorType
) -> bool:
    if node0_num_accelerators < 2:
        return False
    if accelerator_type == AcceleratorType.NV_GPU:
        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    if accelerator_type == AcceleratorType.NPU:
        return (
            hasattr(torch, "npu")
            and torch.npu.is_available()
            and torch.npu.device_count() >= 2
        )
    return False


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
        # Attach a Worker instance to this Ray actor process.
        accel_type = AcceleratorType(accelerator_type)
        local_acc_rank = 0 if accel_type != AcceleratorType.NO_ACCEL else -1
        self._worker = Worker.attach_to_current_ray_actor(
            group_name=group_name,
            rank=rank,
            world_size=world_size,
            cluster_node_rank=cluster_node_rank,
            node_group_label=node_group_label,
            accelerator_type=accelerator_type,
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
            # When isolate_accelerator=1, LOCAL_RANK is always 0.
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        if device == "npu":
            if not hasattr(torch, "npu") or not torch.npu.is_available():
                raise RuntimeError("NPU (torch.npu) is not available in this actor.")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.npu.set_device(local_rank)
            return torch.device("npu", local_rank)
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

        logic_id = None
        if dev != torch.device('cpu'):
            visible_devices = AcceleratorUtil.get_visible_devices(self._worker.accelerator_type)
            logic_id = visible_devices[dev.index]
        return f"send(kind={kind}, device={device}, logic_id={logic_id})"

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

        logic_id = None
        if dev != torch.device('cpu'):
            visible_devices = AcceleratorUtil.get_visible_devices(self._worker.accelerator_type)
            logic_id = visible_devices[dev.index]
        return f"ok(kind={kind}, device={device}, logic_id={logic_id})"


ActorWithWorkerRemote = ray.remote(ActorWithWorker)


def main():
    # 1. Initialize a single-node Cluster, which in turn initializes Ray and
    #    launches all the global managers required by Worker.
    cluster = Cluster(num_nodes=1)

    # 2. Basic placement info for our demo group of two workers.
    node_rank = 0
    node_group = cluster.get_node_group()
    assert node_group is not None, "Default node group not found in Cluster."
    node_group_label = node_group.label

    def run_roundtrip(base_group_name: str, *, accelerator_type: str, use_accelerator: bool):
        world_size = 2
        isolate = use_accelerator  # Ray isolates one accelerator per actor when True.
        accel_enum = AcceleratorType(accelerator_type)
        res_opts = _actor_resource_options(accel_enum, use_accelerator)

        group_name = f"{base_group_name}_{os.getpid()}_{time.time_ns()}"

        actor0 = ActorWithWorkerRemote.options(
            name=f"{group_name}:0", **res_opts
        ).remote(
            group_name=group_name,
            rank=0,
            world_size=world_size,
            cluster_node_rank=node_rank,
            node_group_label=node_group_label,
            accelerator_type=accelerator_type,
            isolate_accelerator=isolate,
        )
        actor1 = ActorWithWorkerRemote.options(
            name=f"{group_name}:1", **res_opts
        ).remote(
            group_name=group_name,
            rank=1,
            world_size=world_size,
            cluster_node_rank=node_rank,
            node_group_label=node_group_label,
            accelerator_type=accelerator_type,
            isolate_accelerator=isolate,
        )

        device = _accel_device_token(accel_enum) if use_accelerator else "cpu"
        for kind in ["tensor", "tensor_list", "tensor_dict"]:
            # rank 0 -> rank 1
            r = actor1.recv_and_check.remote(
                src_group_name=group_name, src_rank=0, device=device, kind=kind
            )
            s = actor0.send_payload.remote(
                dst_group_name=group_name, dst_rank=1, device=device, kind=kind
            )
            print(ray.get(s))
            print(ray.get(r))

            # rank 1 -> rank 0
            r = actor0.recv_and_check.remote(
                src_group_name=group_name, src_rank=1, device=device, kind=kind
            )
            s = actor1.send_payload.remote(
                dst_group_name=group_name, dst_rank=0, device=device, kind=kind
            )
            print(ray.get(s))
            print(ray.get(r))

    # CPU tests (always run)
    run_roundtrip(
        "demo_worker_group_cpu",
        accelerator_type=AcceleratorType.NO_ACCEL.value,
        use_accelerator=False,
    )

    # GPU / NPU tests (optional): two actors, each needs one device.
    node0 = cluster.get_node_info(0)
    accel_t = AcceleratorType(node0.accelerator_type)
    if _two_accel_tests_runnable(node0.num_accelerators, accel_t):
        run_roundtrip(
            "demo_worker_group_accel",
            accelerator_type=node0.accelerator_type,
            use_accelerator=True,
        )
    else:
        print(
            "Skip accelerator tests: need >=2 GPUs or >=2 NPUs "
            "(cluster detection + torch.cuda / torch.npu) for HCCL/NCCL-style send/recv."
        )


if __name__ == "__main__":
    main()

