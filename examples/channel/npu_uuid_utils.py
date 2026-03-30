import torch
import subprocess
import re
import os
import time

_COMPUTE_CHIPS_CACHE = None
_SERIAL_CACHE = {}

def _get_compute_chips():
    global _COMPUTE_CHIPS_CACHE
    if _COMPUTE_CHIPS_CACHE is not None:
        return _COMPUTE_CHIPS_CACHE

    result = subprocess.run("npu-smi info -m", shell=True, capture_output=True, text=True, encoding="utf-8")
    compute_chips = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "NPU ID" in line or "Chip" in line:
            continue

        parts = list(filter(None, re.split(r'\s+', line)))
        if len(parts) < 4:
            continue
        npu_id, chip_id, chip_logic_id = parts[0], parts[1], parts[2]

        if chip_logic_id != "-":
            compute_chips.append((int(chip_logic_id), int(npu_id), int(chip_id)))
    
    _COMPUTE_CHIPS_CACHE = compute_chips
    return compute_chips

def _get_visible_logic_ids():
    visible_env = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    if visible_env:
        return [int(x) for x in visible_env.split(",") if x.strip().isdigit()]
    
    num_npu = torch.npu.device_count()
    return list(range(num_npu))

def get_device_uuid(device: torch.device) -> str:
    if device.type != "npu":
        raise ValueError(f"only support NPU device, current device type: {device.type}")
    torch_dev_idx = device.index

    compute_chips = _get_compute_chips()
    visible_logic_ids = _get_visible_logic_ids()
    target_logic_id = visible_logic_ids[torch_dev_idx]

    chip_map = {logic: (npu, cid) for logic, npu, cid in compute_chips}
    npu_id, chip_id = chip_map[target_logic_id]

    if npu_id not in _SERIAL_CACHE:
        res_serial = subprocess.run(
            f"npu-smi info -t board -i {npu_id}",
            shell=True, capture_output=True, text=True, encoding="utf-8"
        )
        serial = re.search(r"Serial Number\s+:\s*(\S+)", res_serial.stdout).group(1)
        _SERIAL_CACHE[npu_id] = serial
    
    return f"{_SERIAL_CACHE[npu_id]}:{chip_id}"

def get_logic_id(device: torch.device):
    if device.type != "npu":
        raise ValueError(f"only support NPU device, current device type: {device.type}")
    torch_dev_idx = device.index
    visible_logic_ids = _get_visible_logic_ids()
    return visible_logic_ids[torch_dev_idx]

if __name__ == "__main__":
    num_npu = torch.npu.device_count()
    print(f"{num_npu=}")

    for i in range(num_npu):
        start = time.time()
        a = torch.arange(6, device=i)
        t_c = time.time() - start

        start = time.time()
        uuid = get_device_uuid(a.device)
        t_u = time.time() - start

        start = time.time()
        logical_id = get_logic_id(a.device)
        t_l = time.time() - start

        start = time.time()
        device = torch.npu.get_device_properties(a.device)
        t_d = time.time() - start

        print(f"{t_c=:.4f}, {a.device=}, {t_u=:.4f}, {uuid=}, {t_l=:.4f}, {logical_id=}, {t_d=:.4f}, {device=}")
