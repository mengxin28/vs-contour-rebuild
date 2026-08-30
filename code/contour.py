# -*- coding: utf-8 -*-
"""
地下车库 贴合外墙轮廓线提取（阶段二）
输入：阶段一输出的 `*_wall.ply`（投影后 Z=0 的外墙点云）。
目标：输出一条**贴合外沿墙体走向的轮廓线**（沿真实墙体外沿走线，保留真实转角与内凹，
      不在凹口/空腔处搭斜线、不用凹包/凸包包合成闭合区域）。
方法：墙点栅格化 -> 形态学闭运算(连缝) -> 填充内部空腔 -> 提取外沿轮廓(marching squares)
      -> shapely 简化成直线段 -> 输出折线与统计。
输出：
  - `*_外轮廓.png` —— 墙点散点上叠加「贴合外沿的轮廓线」
  - `*_外轮廓.json` —— 轮廓线顶点、面积、周长
用法：
    python contour.py <wall.ply ...>
例如：
    python contour.py ../输出/CLEAN_UNDER_GROUND_wall.ply ../输出/RESULT_B1_wall.ply
"""
import os
import sys
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening
from shapely.geometry import Polygon

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import open3d as o3d

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RASTER_GRID = 0.5   # 栅格边长(m)，越小越精细
CLOSE_ITER = 2      # 闭运算次数：把外墙缝连起来（0=不连缝）
OPEN_KERNEL = 2     # 开运算核边长：去掉一格外伸的细刺(0=不去刺)
SIMPLIFY_TOL = 0.8  # 贴合轮廓线简化容差(m)
REG_TOL = 4.0       # 规则化：Douglas-Peucker 简化容差(m)，越大越规则(顶点越少)


def read_wall_xy(ply_path):
    """读取 Z=0 的外墙点云，返回 (N,2) XY 点阵。"""
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("空点云: %s" % ply_path)
    return pts[:, :2]


def raster_mask(pts, grid=RASTER_GRID):
    """墙点 -> 占用栅格布尔掩码 + 原点与尺寸。"""
    x0, y0 = pts.min(axis=0)
    g = np.floor((pts - [x0, y0]) / grid).astype(np.int64)
    nx, ny = int(g[:, 0].max()) + 1, int(g[:, 1].max()) + 1
    mask = np.zeros((ny, nx), dtype=bool)
    mask[g[:, 1], g[:, 0]] = True
    return mask, (float(x0), float(y0)), grid


def trace_outline(pts, grid=RASTER_GRID, close_iter=CLOSE_ITER,
                  open_kernel=OPEN_KERNEL, tol=SIMPLIFY_TOL):
    """栅格化 -> 连缝 -> 填内部空腔 -> 去外伸细刺 -> 提取外沿轮廓 -> 简化。返回闭合 Polygon。"""
    mask, (x0, y0), g = raster_mask(pts, grid)
    if close_iter:
        mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=close_iter)
    mask = binary_fill_holes(mask)
    if open_kernel:
        mask = binary_opening(mask,
                              structure=np.ones((open_kernel, open_kernel), dtype=bool))
    # 用 matplotlib 的 marching squares 在该级(0.5)提轮廓；外沿为最长的一条
    cs = plt.contour(mask.astype(np.float64), levels=[0.5])
    segs = cs.allsegs[0]
    if not segs:
        plt.close(cs.axes.figure)
        raise ValueError("未提取到轮廓线")
    path = max(segs, key=lambda a: len(a))       # 索引坐标 (col, row)
    plt.close(cs.axes.figure)
    xy = np.column_stack([x0 + path[:, 0] * g, y0 + path[:, 1] * g])
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    poly = poly.simplify(tol, preserve_topology=True)
    return poly


def _rotate(pts, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def regularize(poly, tol=REG_TOL):
    """把贴合轮廓线规则化：转到主方向后做 Douglas-Peucker，使每段墙成为一条直边。
    DP 在欧氏距离下与方向无关，故转正主要作为后续"吸附主导方向"的锚点；此处转正后简化再转回，
    保证输出外形不受因旋转引入的坐标误差影响。"""
    if poly.geom_type != "Polygon" or poly.exterior is None:
        return poly
    rc = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
    d = rc[1] - rc[0]
    ang = math.atan2(d[1], d[0]) % (math.pi / 2)
    if ang > math.pi / 4:
        ang -= math.pi / 2
    coords = np.asarray(poly.exterior.coords)[:-1]
    rot = _rotate(coords, -ang)          # 转正：主方向墙对齐坐标轴
    simp = poly.simplify(tol, preserve_topology=True)
    if simp.geom_type != "Polygon" or simp.exterior is None:
        return poly
    s = np.asarray(simp.exterior.coords)[:-1]
    back = _rotate(s, ang)               # 转回原方向
    newpoly = Polygon(np.vstack([back, back[:1]]))
    if not newpoly.is_valid:
        newpoly = newpoly.buffer(0)
    if newpoly.geom_type != "Polygon":
        return poly
    return newpoly


def plot_result(pts, poly, hug_poly, path, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c="#1f77b4", marker=".", alpha=0.5)
    if hug_poly is not None:
        hx, hy = np.asarray(hug_poly.exterior.coords).T
        ax.plot(hx, hy, "-", color="#7f7f7f", lw=1, alpha=0.7, label="贴合外沿(原始)")
    px, py = np.asarray(poly.exterior.coords).T
    ax.plot(px, py, "-", color="#d62728", lw=2.2, label="规则化外轮廓")
    ax.legend(loc="best")
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
    fig.savefig(path, dpi=135)
    plt.close(fig)


def process(base, wall_ply, out_dir):
    print("\n========== 规则外轮廓: %s ==========" % wall_ply)
    pts = read_wall_xy(wall_ply)
    hug = trace_outline(pts)            # 贴合外沿的线
    poly = regularize(hug)              # 规则化（直角规整）
    coords = np.asarray(poly.exterior.coords)
    info = {
        "wall_points": int(len(pts)),
        "outline_vertices": int(len(coords) - 1),
        "outline_area_m2": round(float(poly.area), 2),
        "outline_perimeter_m": round(float(poly.length), 2),
        "raster_grid_m": RASTER_GRID,
        "reg_tol_m": REG_TOL,
        "outline_vertices_xy": [[round(float(x), 3), round(float(y), 3)] for x, y in coords[:-1]],
    }
    plot_result(pts, poly, hug, "%s/%s_外轮廓.png" % (out_dir, base), "%s 规则化外轮廓" % base)
    with open("%s/%s_外轮廓.json" % (out_dir, base), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("规则化后顶点=%d, 面积=%.1f m², 周长=%.1f m" %
          (info["outline_vertices"], info["outline_area_m2"], info["outline_perimeter_m"]))
    return info


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "输出")
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for ply in sys.argv[1:]:
        base = os.path.splitext(os.path.basename(ply))[0]
        base = base.replace("_wall", "") if base.endswith("_wall") else base
        summary[base] = process(base, ply, out_dir)
    with open(os.path.join(out_dir, "外轮廓_汇总.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n完成，输出目录: %s" % out_dir)


if __name__ == "__main__":
    main()
