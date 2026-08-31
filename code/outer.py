# -*- coding: utf-8 -*-
"""
地下车库 外圈高密度点云高亮（阶段二·可视化，v0.7 密度用原始点）
输入：原始点云 `*.ply`（CLEAN_UNDER_GROUND.ply / RESULT_B1.ply）。
目标：把**外圈高密度墙体点**用红色标出（不描轮廓）。
密度定义：**竖直堆叠密度** —— 同一 0.3m XY 小柱内的点数。墙体/柱在竖直方向叠满、密度高；
         地面/天花板只在顶底各一层、密度低。故"密度前5%"能聚到墙/柱。
判定 = 两者结合：竖直密度 前5%  AND  距外边界 ≤ BAND。
输出：`*_外圈点云.png`（红=外圈高密度墙，橙=高密度但不在外圈，浅蓝=其余）。
用法：
    python outer.py <原始.ply ...>
例如：
    python outer.py ../CLEAN_UNDER_GROUND.ply ../RESULT_B1.ply
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import open3d as o3d

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

COL_GRID = 0.3   # 竖直密度小柱边(m)
GRID = 0.5       # 外圈位置栅格边(m)
CLOSE_ITER = 2
PCT = 95         # 密度百分位：前5%判为高密度
BAND = 1.5       # 外圈带宽(m)


def read_xyz(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("空点云: %s" % ply_path)
    return pts


def footprint_mask(xy, grid=GRID, close_iter=CLOSE_ITER):
    x0, y0 = xy.min(axis=0)
    g = np.floor((xy - [x0, y0]) / grid).astype(np.int64)
    nx, ny = int(g[:, 0].max()) + 1, int(g[:, 1].max()) + 1
    mask = np.zeros((ny, nx), dtype=bool)
    mask[g[:, 1], g[:, 0]] = True
    if close_iter:
        mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=close_iter)
    mask = binary_fill_holes(mask)
    return mask, g, (float(x0), float(y0)), grid


def classify(raw, pct=PCT, band=BAND):
    """返回红/橙布尔掩码（与 raw 对齐）。"""
    xy = raw[:, :2]
    # 1) 竖直堆叠密度：同 0.3m XY 柱内点数
    cell = np.floor(xy / COL_GRID).astype(np.int64)
    key = cell[:, 0] * 100000 + cell[:, 1]
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    density = counts[inv].astype(np.float64)
    thresh = np.percentile(density, pct)
    high = density >= thresh
    # 2) 外圈位置：到外边界距离
    mask, g, (x0, y0), grid = footprint_mask(xy)
    dist_cells = distance_transform_edt(mask)
    band_cells = band / grid
    inb = ((g[:, 0] >= 0) & (g[:, 0] < mask.shape[1]) &
           (g[:, 1] >= 0) & (g[:, 1] < mask.shape[0]))
    d = np.zeros(len(xy))
    d[inb] = dist_cells[g[inb, 1], g[inb, 0]]
    outer = inb & (d <= band_cells)
    red = high & outer
    orange = high & ~outer
    return red, orange, int(thresh)


def plot(raw, red, orange, path, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(raw[~red, 0], raw[~red, 1], s=0.3, c="#c6dbef", marker=".", alpha=0.5, label="其余点")
    ax.scatter(raw[orange, 0], raw[orange, 1], s=0.9, c="#ff7f0e", marker=".", alpha=0.85,
               label="高密度但不在外圈")
    ax.scatter(raw[red, 0], raw[red, 1], s=1.2, c="#d62728", marker=".", alpha=0.95,
               label="外圈高密度点(外墙)")
    ax.legend(loc="best", markerscale=6)
    x0, x1 = raw[:, 0].min(), raw[:, 0].max()
    y0, y1 = raw[:, 1].min(), raw[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * 1.05
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def process(base, ply, out_dir):
    print("\n========== 外圈高亮(原始点竖直密度+位置): %s ==========" % ply)
    raw = read_xyz(ply)
    red, orange, thresh = classify(raw)
    plot(raw, red, orange, "%s/%s_外圈点云.png" % (out_dir, base), "%s 外圈高密度点云(红)" % base)
    n = len(raw)
    print("竖直密度阈值=%d 点/%s m柱; 高密度前%.0f%%=%d点; 红(外圈高密度墙)=%d (%.1f%%); 橙(密但不在外圈)=%d" % (
        thresh, COL_GRID, PCT, int((red | orange).sum()), int(red.sum()), 100 * red.sum() / n, int(orange.sum())))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "输出")
    os.makedirs(out_dir, exist_ok=True)
    for ply in sys.argv[1:]:
        base = os.path.splitext(os.path.basename(ply))[0]
        process(base, ply, out_dir)
    print("\n完成，输出目录: %s" % out_dir)


if __name__ == "__main__":
    main()
