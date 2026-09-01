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
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt, label

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import open3d as o3d

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

COL_GRID = 0.3    # 竖直密度小柱边(m)
GRID = 0.5        # 外圈位置栅格边(m)
CLOSE_ITER = 2
GLOBAL_PCT = 90   # 全局密度百位分：≥90% 为"全局前10%"
LOCAL_PCT = 90    # 局域密度百位分：≥该局部区域90% 为"局域前10%"
LOCAL_CELL = 12.0 # 局域比较窗口边长(m)
BAND = 1.0        # 外圈带宽(m)
KEEP_ONLY_BIGGEST = True  # 只保留最大连通块，剔除脱离主体的噪声点/块


def read_xyz(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("空点云: %s" % ply_path)
    return pts


def footprint_mask(xy, grid=GRID, close_iter=CLOSE_ITER):
    """生成主footprint栅格；只保留最大连通块，剔除脱离主体的噪声点/块。"""
    x0, y0 = xy.min(axis=0)
    g = np.floor((xy - [x0, y0]) / grid).astype(np.int64)
    nx, ny = int(g[:, 0].max()) + 1, int(g[:, 1].max()) + 1
    mask = np.zeros((ny, nx), dtype=bool)
    mask[g[:, 1], g[:, 0]] = True
    if close_iter:
        mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=close_iter)
    mask = binary_fill_holes(mask)
    if KEEP_ONLY_BIGGEST:
        lab, nlab = label(mask)
        if nlab > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            mask = lab == int(sizes.argmax())
    return mask, g, (float(x0), float(y0)), grid


def classify(raw, global_pct=GLOBAL_PCT, local_pct=LOCAL_PCT,
             local_cell=LOCAL_CELL, band=BAND):
    """标记条件 = 局域前local_pct% AND 全局前global_pct% AND 外圈位置。
    返回 (red, orange, info)。"""
    xy = raw[:, :2]
    # 1) 竖直堆叠密度：同 0.3m XY 柱内点数
    cell = np.floor(xy / COL_GRID).astype(np.int64)
    key = cell[:, 0] * 100000 + cell[:, 1]
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    density = counts[inv].astype(np.float64)
    # 全局前5%
    global_thr = np.percentile(density, global_pct)
    global_top = density >= global_thr
    # 局域前10%：以 12m 网格为"局部区域"，点密度≥该区域90分位
    lc = np.floor(xy / local_cell).astype(np.int64)
    lkey = lc[:, 0] * 1000000 + lc[:, 1]
    _, linv = np.unique(lkey, return_inverse=True)
    local_top = np.zeros(len(xy), dtype=bool)
    for ci in np.unique(linv):
        idx = np.where(linv == ci)[0]
        local_top[idx] = density[idx] >= np.percentile(density[idx], local_pct)
    dual = global_top & local_top       # 全局前10% 且 局域前10%
    # 2) 主 footprint + 外圈位置
    mask, g, (x0, y0), grid = footprint_mask(xy)
    dist_cells = distance_transform_edt(mask)          # 主块内的点到其外边界距离
    in_keep = mask[g[:, 1], g[:, 0]]                   # 是否落在"主 footprint 内"
    d = np.full(len(xy), 1e9, dtype=np.float64)        # 主块外=无穷大
    d[in_keep] = dist_cells[g[in_keep, 1], g[in_keep, 0]]
    outer = in_keep & (d <= band / grid)
    red = dual & outer                  # 双密度门槛 且 在主外圈
    orange = dual & in_keep & ~outer    # 双密度门槛 但主内但非外圈(内部柱)
    noise = ~in_keep                    # 落在被剔除噪声块的点 -> 已剔除噪声
    info = {"global_thr": int(global_thr), "global_top": int(global_top.sum()),
            "dual": int(dual.sum()), "red": int(red.sum()), "orange": int(orange.sum()),
            "noise": int(noise.sum())}
    return red, orange, noise, info


def plot(raw, red, orange, noise, path, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    other = ~(red | orange | noise)
    ax.scatter(raw[other, 0], raw[other, 1], s=0.3, c="#c6dbef", marker=".", alpha=0.5, label="其余点")
    ax.scatter(raw[noise, 0], raw[noise, 1], s=0.7, c="#9467bd", marker=".", alpha=0.8,
               label="墙外噪声(已剔除)")
    ax.scatter(raw[orange, 0], raw[orange, 1], s=0.9, c="#ff7f0e", marker=".", alpha=0.85,
               label="高密度但不在外圈(内部柱)")
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
    red, orange, noise, info = classify(raw)
    plot(raw, red, orange, noise, "%s/%s_外圈点云.png" % (out_dir, base), "%s 外圈高密度点云(红)" % base)
    n = len(raw)
    print("全局密度阈值=%d点/柱; 双重(全局前%.0f%%且局域前%.0f%%)=%d点; 红(外墙)=%d (%.1f%%); "
          "橙(内部柱)=%d; 墙外噪声(已剔除)=%d" % (
              info["global_thr"], GLOBAL_PCT, LOCAL_PCT, info["dual"],
              info["red"], 100 * info["red"] / n, info["orange"], info["noise"]))


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
