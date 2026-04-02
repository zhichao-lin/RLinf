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

"""Visualize one bench_channel_td JSON file as grouped bar charts.

Pass a single JSON path (list of records). Requires matplotlib: ``pip install matplotlib``.

Writes two PNG files: ``<json_stem>_sync.png`` and ``<json_stem>_async.png``, each with
two subplots (put on top, get on bottom).

Per desc category: sub-bars cpu, same-npu, cross-npu using ``stats.<metric>.max``.
For cpu, uses max of placement=same vs placement=cross when both exist.

Use ``PUT_SUBPLOT_LOG_Y`` / ``GET_SUBPLOT_LOG_Y`` to choose log y-axis per subplot independently;
bar labels remain linear Gbps.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

# Must match bench_channel_td.DESC_CONFIGS order for the x-axis.
DESC_ORDER = ("debug", "tiny", "small", "medium", "large", "xlarge", "huge")

KIND_SYNC = "channel_sync"
KIND_ASYNC = "channel_async"

# Grouped bar colors: cpu | same-npu | cross-npu (left to right)
COLOR_CPU = "#7f8c8d"
COLOR_SAME_NPU = "#2ecc71"
COLOR_CROSS_NPU = "#3498db"

# If True, use a logarithmic y-axis on that subplot; bar labels still show linear Gbps.
PUT_SUBPLOT_LOG_Y = True
GET_SUBPLOT_LOG_Y = True


def _metric_max(stats: dict[str, Any], metric: str) -> float | None:
    m = stats.get(metric)
    if not isinstance(m, dict):
        return None
    v = m.get("max")
    if v is None:
        return None
    if isinstance(v, (int, float)) and (math.isinf(v) or math.isnan(v)):
        return None
    return float(v)


def _load_json_file(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON root must be a list, got {type(data).__name__}")
    return data


def _index_records(records: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Key: (desc, placement, device, kind) -> record; later records overwrite earlier."""
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for rec in records:
        try:
            key = (rec["desc"], rec["placement"], rec["device"], rec["kind"])
        except KeyError:
            continue
        out[key] = rec
    return out


def _cpu_max(
    by_key: dict[tuple[str, str, str, str], dict[str, Any]],
    desc: str,
    kind: str,
    metric: str,
) -> float | None:
    same = by_key.get((desc, "same", "cpu", kind))
    cross = by_key.get((desc, "cross", "cpu", kind))
    vals: list[float] = []
    if same is not None:
        v = _metric_max(same["stats"], metric)
        if v is not None:
            vals.append(v)
    if cross is not None:
        v = _metric_max(cross["stats"], metric)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return max(vals)


def _single_placement(
    by_key: dict[tuple[str, str, str, str], dict[str, Any]],
    desc: str,
    placement: str,
    device: str,
    kind: str,
    metric: str,
) -> float | None:
    rec = by_key.get((desc, placement, device, kind))
    if rec is None:
        return None
    return _metric_max(rec["stats"], metric)


def _plot_one(
    ax: Any,
    desc_labels: tuple[str, ...],
    cpu_vals: list[float | None],
    same_npu_vals: list[float | None],
    cross_npu_vals: list[float | None],
    title: str,
    ylabel: str,
    *,
    show_legend: bool = True,
    y_axis_scale: str = "linear",
) -> None:
    import numpy as np

    x = np.arange(len(desc_labels))
    width = 0.25

    def _series(vals: list[float | None]) -> list[float]:
        return [0.0 if v is None else float(v) for v in vals]

    s_cpu = _series(cpu_vals)
    s_same = _series(same_npu_vals)
    s_cross = _series(cross_npu_vals)

    use_log = y_axis_scale == "log"
    if use_log:
        max_lin = max(s_cpu + s_same + s_cross, default=0.0)
        if max_lin <= 0.0:
            use_log = False

    if use_log:

        def _heights_for_log(seq: list[float]) -> list[float]:
            out: list[float] = []
            for v in seq:
                if v > 0.0:
                    out.append(float(v))
                else:
                    out.append(float("nan"))
            return out

        p_cpu = _heights_for_log(s_cpu)
        p_same = _heights_for_log(s_same)
        p_cross = _heights_for_log(s_cross)
    else:
        p_cpu, p_same, p_cross = s_cpu, s_same, s_cross

    # Matplotlib bar() defaults to align="center": x is the bar center, not the left edge.
    bar_kw = {"width": width, "align": "center", "edgecolor": "white", "linewidth": 0.5}
    bars_cpu = ax.bar(
        x - width, p_cpu, label="cpu", color=COLOR_CPU, **bar_kw
    )
    bars_same = ax.bar(
        x,
        p_same,
        label="same-npu",
        color=COLOR_SAME_NPU,
        **bar_kw,
    )
    bars_cross = ax.bar(
        x + width,
        p_cross,
        label="cross-npu",
        color=COLOR_CROSS_NPU,
        **bar_kw,
    )

    centers_cpu = x - width
    centers_same = x
    centers_cross = x + width

    def _annotate_linear_labels() -> None:
        for container in (bars_cpu, bars_same, bars_cross):
            for rect in container:
                h = float(rect.get_height())
                ax.annotate(
                    f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2.0, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#2c3e50",
                )

    def _annotate_log_labels() -> None:
        zero_bottom: list[tuple[float, str]] = []

        def one_group(
            centers: Any, plot_h: list[float], label_h: list[float]
        ) -> None:
            for xc, ph, lh in zip(centers, plot_h, label_h):
                txt = f"{lh:.2f}"
                if lh <= 0.0 or (isinstance(ph, float) and math.isnan(ph)):
                    zero_bottom.append((float(xc), txt))
                    continue
                ax.annotate(
                    txt,
                    xy=(xc, lh),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#2c3e50",
                )

        one_group(centers_cpu, p_cpu, s_cpu)
        one_group(centers_same, p_same, s_same)
        one_group(centers_cross, p_cross, s_cross)

        ax.set_yscale("log", base=10)
        y1, y2 = ax.get_ylim()
        if y2 > y1 > 0:
            ax.set_ylim(y1, y2 * 1.15)
        ymin = ax.get_ylim()[0]
        for xc, txt in zero_bottom:
            ax.annotate(
                txt,
                xy=(xc, ymin),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#2c3e50",
            )

    if use_log:
        _annotate_log_labels()
    else:
        _annotate_linear_labels()

    ax.set_xticks(x)
    ax.set_xticklabels(desc_labels)
    ax.tick_params(axis="both", labelsize=10)
    if use_log:
        ax.grid(True, which="major", axis="y", alpha=0.3)
        ax.grid(True, which="minor", axis="y", alpha=0.15)
    else:
        ax.grid(True, alpha=0.3, axis="y")
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.1)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    if show_legend:
        ax.legend(fontsize=10)


def _collect_bars(
    by_key: dict[tuple[str, str, str, str], dict[str, Any]],
    kind: str,
    metric: str,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    cpu_vals: list[float | None] = []
    same_vals: list[float | None] = []
    cross_vals: list[float | None] = []
    for desc in DESC_ORDER:
        cpu_vals.append(_cpu_max(by_key, desc, kind, metric))
        same_vals.append(_single_placement(by_key, desc, "same", "npu", kind, metric))
        cross_vals.append(_single_placement(by_key, desc, "cross", "npu", kind, metric))
    return cpu_vals, same_vals, cross_vals


def _warn_missing(
    subtitle: str,
    cpu_vals: list[float | None],
    same_vals: list[float | None],
    cross_vals: list[float | None],
) -> None:
    labels = ("cpu", "same-npu", "cross-npu")
    series_list = (cpu_vals, same_vals, cross_vals)
    for lab, series in zip(labels, series_list):
        for desc, v in zip(DESC_ORDER, series):
            if v is None:
                print(
                    f"  [missing] {subtitle}: desc={desc} series={lab} "
                    f"(no matching JSON record; bar height 0)"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one bench_channel_td JSON file (put/get sync | async)."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to a bench_channel_td stats JSON file (list of records).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output PNGs (default: same directory as the JSON file).",
    )
    args = parser.parse_args()

    json_path = args.json_path.resolve()
    if not json_path.is_file():
        raise SystemExit(f"JSON file not found: {json_path}")

    output_dir = args.output_dir or json_path.parent
    out_prefix = json_path.stem

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    try:
        records = _load_json_file(json_path)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        raise SystemExit(f"Failed to load JSON: {e}") from e

    if not records:
        raise SystemExit(f"No records in JSON: {json_path}")

    by_key = _index_records(records)
    os.makedirs(output_dir, exist_ok=True)

    y_label = "Throughput (Gbps)"

    for kind, suffix in ((KIND_SYNC, "sync"), (KIND_ASYNC, "async")):
        put_cpu, put_same, put_cross = _collect_bars(by_key, kind, "put_gbit_per_sec")
        get_cpu, get_same, get_cross = _collect_bars(by_key, kind, "get_gbit_per_sec")

        _warn_missing(f"put ({suffix})", put_cpu, put_same, put_cross)
        _warn_missing(f"get ({suffix})", get_cpu, get_same, get_cross)

        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=True,
            constrained_layout=True,
        )
        _plot_one(
            ax_top,
            DESC_ORDER,
            put_cpu,
            put_same,
            put_cross,
            f"Throughput — put ({suffix})",
            y_label,
            show_legend=True,
            y_axis_scale="log" if PUT_SUBPLOT_LOG_Y else "linear",
        )
        _plot_one(
            ax_bottom,
            DESC_ORDER,
            get_cpu,
            get_same,
            get_cross,
            f"Throughput — get ({suffix})",
            y_label,
            show_legend=False,
            y_axis_scale="log" if GET_SUBPLOT_LOG_Y else "linear",
        )
        ax_bottom.set_xlabel("Config", fontsize=12)
        out_path = output_dir / f"{out_prefix}_{suffix}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
