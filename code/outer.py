# -*- coding: utf-8 -*-
"""
地下车库 外圈高密度点云高亮（阶段二·可视化）
输入：阶段一输出的 `*_wall.ply`（投影后 Z=0 的外墙点云）。
目标：不描轮廓线，把**外侧一圈（最外层、密集成环）的高密度墙体点**用红色标出，其余点保持浅色。
方法：栅格化 -> 闭运算连缝 -> 填内部空腔 -> 距离变换求"到外边界距离" -> 阈值内判为外圈点。
输出：`*_外圈点云.png`
用法：
    python outer.py <wall.ply ...>
例如：
    python outer.py ../输出/CLEAN_UNDER_GROUND_wall.ply ../输出/RESULT_B1_wall.ply
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

GRID = 0.5        # 栅格边长(m)
CLOSE_ITER = 2    # 连缝次数
BAND = 2.0        # 外圈带宽(m)，距外边界小于该值为"外圈"


def read_wall_xy(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("空点云: %s" % ply_path)
    return pts[:, :2]


def build_mask(pts, grid=GRID, close_iter=CLOSE_ITER):
    x0, y0 = pts.min(axis=0)
    g = np.floor((pts - [x0, y0]) / grid).astype(np.int64)
    nx, ny = int(g[:, 0].max()) + 1, int(g[:, 1].max()) + 1
    mask = np.zeros((ny, nx), dtype=bool)
    mask[g[:, 1], g[:, 0]] = True
    if close_iter:
        mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=close_iter)
    mask = binary_fill_holes(mask)
    return mask, g, (float(x0), float(y0))


def highlight_outer(pts, grid=GRID, band=BAND):
    """返回布尔掩码：True=外圈高密度点。"""
    mask, g, (x0, y0) = build_mask(pts, grid)
    dist_cells = distance_transform_edt(mask)          # 内部点到最近外界的栅格距离
    band_cells = band / grid
    # 每个墙点对应的栅格距离；越界点(稀少)强制为外圈(视为噪声剔除)
    inb = (g[:, 0] >= 0) & (g[:, 0] < mask.shape[1]) & (g[:, 1] >= 0) & (g[:, 1] < mask.shape[0])
    d = np.zeros(len(pts))
    d[inb] = dist_cells[g[inb, 1], g[inb, 0]]
    outer = inb & (d <= band_cells)
    return outer


def plot(pts, outer, path, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    # 背景浅色点
    ax.scatter(pts[~outer, 0], pts[~outer, 1], s=0.5, c="#c6dbef", marker=".", alpha=0.7,
               label="内部/其余点")
    # 外圈高密度点 -> 红色
    ax.scatter(pts[outer, 0], pts[outer, 1], s=1.2, c="#d62728", marker=".", alpha=0.9,
               label="外圈高密度点(外墙)")
    ax.legend(loc="best", markerscale=6)
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
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


def process(base, wall_ply, out_dir):
    print("\n========== 外圈高亮: %s ==========" % wall_ply)
    pts = read_wall_xy(wall_ply)
    outer = highlight_outer(pts)
    plot(pts, outer, "%s/%s_外圈点云.png" % (out_dir, base), "%s 外圈高密度点云(红)" % base)
    print("外圈点=%d / 总点=%d (占比 %.1f%%)" % (int(outer.sum()), len(pts), 100 * outer.sum() / len(pts)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "输出")
    os.makedirs(out_dir, exist_ok=True)
    for ply in sys.argv[1:]:
        base = os.path.splitext(os.path.basename(ply))[0]
        base = base.replace("_wall", "") if base.endswith("_wall") else base
        process(base, ply, out_dir)
    print("\n完成，输出目录: %s" % out_dir)


if __name__ == "__main__":
    main()
