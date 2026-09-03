# -*- coding: utf-8 -*-
"""
精度验证：外圈点云(红点) 与 轮廓线 的贴合程度
指标：每个外圈点 -> 轮廓线最近距离 d，误差取 d²。
统计：轮廓线按弧长每 5m 一段；每段统计 点数/平均d²/最大d²。
颜色：平均误差小=绿(<=GREEN_D)、中=橙(<=ORANGE_D)、大=红(>ORANGE_D)。
输出：`*_精度验证.png`（按段着色的轮廓线）+ `*_精度验证.json`。
用法：python validate.py <源点云.ply/.las> 输出/<NAME>_正交轮廓.json
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import outer as outer_mod

BIN_LEN = 5.0        # 每段弧长(m)
GREEN_D = 0.5        # 平均误差 <=0.5m -> 绿
ORANGE_D = 1.5       # 平均误差 <=1.5m -> 橙, 否则红
SAMPLE_STEP = 0.25   # 轮廓线采样步长(m)

COL = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728", 3: "#999999"}   # 绿/橙/红/灰(无点)


def densify_ring(verts):
    """把开口顶点环闭合, 按 SAMPLE_STEP 重采样; 返回 (样本点(M,2), 每样本弧长s)。"""
    closed = np.vstack([verts, verts[:1]])
    seg = np.diff(closed, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    pts = []
    s = 0.0
    arc = []
    for i in range(len(seg)):
        L = seglen[i]
        nseg = max(int(L / SAMPLE_STEP), 1)
        for k in range(nseg):
            pts.append(closed[i] + seg[i] * (k / nseg))
            arc.append(s + L * (k / nseg))
        s += L
    return np.array(pts), np.array(arc)


def classify_bin(dists, s_vals, total_len, bin_len=BIN_LEN):
    """dist: 每样本对应距离? 不, 这里直接对点: 返回每个点(on 样本)的bin索引."""
    return np.floor(s_vals / bin_len).astype(int)


def process(source, outline_json, out_dir):
    base = os.path.splitext(os.path.basename(outline_json))[0].replace("_正交轮廓", "")
    print("========== 精度验证: %s ==========" % base)
    # 1) 轮廓线：读 json 顶点, 闭合后采样
    info = json.load(open(outline_json, "r", encoding="utf-8"))
    verts = np.array(info["vertex_xy"])          # 开口顶点
    samples, s_vals = densify_ring(verts)
    total_len = float(s_vals[-1] + np.linalg.norm(verts[-1] - verts[0]))
    # 2) 外圈点云(红点)
    raw = outer_mod.read_xyz(source)
    red, orange, _i = outer_mod.classify(raw)
    red_xy = raw[red][:, :2]
    print("外圈点=%d, 轮廓采样=%d, 轮廓总长=%.1fm" % (len(red_xy), len(samples), total_len))
    # 3) 每个外圈点 -> 轮廓最近样本 -> 距离d, d², 所在5m段
    tree = cKDTree(samples)
    d, idx = tree.query(red_xy, k=1)
    d2 = d ** 2
    bins = (s_vals[idx] // BIN_LEN).astype(int)
    nb = int(total_len // BIN_LEN) + 2
    stats = []
    for b in range(nb):
        sel = bins == b
        cnt = int(sel.sum())
        mean_d = float(d[sel].mean()) if cnt else 0.0
        max_d = float(d[sel].max()) if cnt else 0.0
        # 无外圈点段 -> 灰色(无数据, 仍绘制保证闭合连续)
        lvl = 3 if cnt == 0 else (0 if mean_d <= GREEN_D else (1 if mean_d <= ORANGE_D else 2))
        stats.append({"bin": b, "s_start": b * BIN_LEN, "pts": cnt,
                      "mean_d": round(mean_d, 3), "max_d": round(max_d, 3), "level": lvl})
    # 4) 画图：轮廓线按 5m 段等级着色, 外圈点云淡蓝衬底
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(red_xy[:, 0], red_xy[:, 1], s=0.5, c="#c6dbef", marker=".", alpha=0.6,
               label="外圈点云")
    for st in stats:
        m = (s_vals // BIN_LEN).astype(int) == st["bin"]
        ax.plot(samples[m, 0], samples[m, 1], color=COL[st["level"]], lw=2.5)
    ax.plot([], [], color=COL[0], lw=3, label="绿: 平均误差<=%.1fm" % GREEN_D)
    ax.plot([], [], color=COL[1], lw=3, label="橙: 平均误差<=%.1fm" % ORANGE_D)
    ax.plot([], [], color=COL[2], lw=3, label="红: 平均误差>%.1fm" % ORANGE_D)
    ax.plot([], [], color=COL[3], lw=3, label="灰: 该段无外圈点")
    ax.set_aspect("equal")
    ax.set_title("%s 精度验证(外圈点云->轮廓线距离, 5m/段)" % base)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "%s_精度验证.png" % base), dpi=150)
    plt.close(fig)
    # 5) 汇总
    mean_d2_all = float(d2.mean())
    mean_d_all = float(d.mean())
    frac_g = sum(1 for s_ in stats if s_["level"] == 0) / max(len(stats), 1)
    summary = {"outer_errors": {"mean_d": round(mean_d_all, 3), "max_d": round(float(d.max()), 3),
                                "mean_d2": round(mean_d2_all, 3)},
               "bin_len_m": BIN_LEN, "bins": stats,
               "levels_frac": {"green": round(frac_g, 3)}}
    with open(os.path.join(out_dir, "%s_精度验证.json" % base), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("全局: 平均d=%.3fm 最大d=%.3fm 平均d²=%.3f 段数=%d" % (
        summary["outer_errors"]["mean_d"], summary["outer_errors"]["max_d"],
        summary["outer_errors"]["mean_d2"], len(stats)))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "输出")
    os.makedirs(out_dir, exist_ok=True)
    process(sys.argv[1], sys.argv[2], out_dir)


if __name__ == "__main__":
    main()
