# -*- coding: utf-8 -*-
"""
地下车库 规则外轮廓提取（阶段二）
输入：阶段一输出的 `*_wall.ply`（投影后 Z=0 的外墙点云）。
输出：
  - `*_外轮廓.png` —— 墙点散点上叠加 外轮廓多边形 + 最小外接旋转矩形
  - `*_外轮廓.json` —— 外轮廓多边形顶点、面积、周长、最小外接矩形信息
方法：凹包(concave hull) -> 取最大多边形 -> Douglas-Peucker 简化成直线段多边形。

用法：
    python contour.py <wall.ply ...>
例如：
    python contour.py ../输出/CLEAN_UNDER_GROUND_wall.ply ../输出/RESULT_B1_wall.ply
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Polygon
from shapely import concave_hull

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import open3d as o3d

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CONCAVE_RATIO = 0.15   # 凹包参数: 越小越贴凹边, 1=凸包
SIMPLIFY_TOL = 0.8     # 欧几里得简化容差(m)


def read_wall_xy(ply_path):
    """读取 Z=0 的外墙点云，返回 (N,2) XY 点阵。"""
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("空点云: %s" % ply_path)
    return pts[:, :2]


def largest_polygon(geom):
    """若结果是多边形集合，取面积最大的一个外环。"""
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    raise ValueError("凹包结果不是多边形: %s" % geom.geom_type)


def extract_contour(pts, ratio=CONCAVE_RATIO, tol=SIMPLIFY_TOL):
    """返回 (简化外轮廓 Polygon, 最小外接旋转矩形 Polygon)。"""
    mp = MultiPoint(pts)
    hull = concave_hull(mp, ratio=ratio, allow_holes=False)
    poly = largest_polygon(hull)
    poly = poly.simplify(tol, preserve_topology=True)
    rect = poly.minimum_rotated_rectangle
    return poly, rect


def plot_result(pts, poly, rect, path, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c="#1f77b4", marker=".", alpha=0.6)
    px, py = np.asarray(poly.exterior.coords).T
    ax.plot(px, py, "-", color="#d62728", lw=2, label="外轮廓(简化)")
    rx, ry = np.asarray(rect.exterior.coords).T
    ax.plot(rx, ry, "--", color="#2ca02c", lw=1.5, label="最小外接矩形")
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
    print("\n========== 外轮廓: %s ==========" % wall_ply)
    pts = read_wall_xy(wall_ply)
    poly, rect = extract_contour(pts)
    coords = np.asarray(poly.exterior.coords)
    rcoords = np.asarray(rect.exterior.coords)
    info = {
        "wall_points": int(len(pts)),
        "outline_vertices": int(len(coords) - 1),
        "outline_area_m2": round(float(poly.area), 2),
        "outline_perimeter_m": round(float(poly.length), 2),
        "outline_vertices_xy": [[round(float(x), 3), round(float(y), 3)] for x, y in coords[:-1]],
        "rotated_rect_area_m2": round(float(rect.area), 2),
        "rotated_rect_vertices_xy": [[round(float(x), 3), round(float(y), 3)] for x, y in rcoords[:-1]],
    }
    plot_result(pts, poly, rect, "%s/%s_外轮廓.png" % (out_dir, base),
                "%s 规则外轮廓" % base)
    with open("%s/%s_外轮廓.json" % (out_dir, base), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("外轮廓顶点=%d, 面积=%.1f m², 周长=%.1f m, 顶点/矩形见 json" %
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
        # base 形如 "CLEAN_UNDER_GROUND_wall"，去掉 _wall 后缀
        base = base.replace("_wall", "") if base.endswith("_wall") else base
        summary[base] = process(base, ply, out_dir)
    with open(os.path.join(out_dir, "外轮廓_汇总.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n完成，输出目录: %s" % out_dir)


if __name__ == "__main__":
    main()
