# -*- coding: utf-8 -*-
"""
地下车库 正交轮廓（阶段三·方案A：wall.ply 连线 + snap_1d 直角正交化）
输入：preprocess 得到的 `*_wall.ply`（投影后 Z=0 的整面墙点）。
流程：contour.trace_outline 连成闭合轮廓 -> 在建筑主方向坐标系内 snap_1d 一维聚类拉平 -> 去共线。
      在"主方向坐标系"内正交化，因此旋转类车库(如雅德)也能得到带直角的规整矩形。
输出：`*_正交轮廓.png`（灰=墙点投影，红=正交轮廓）+ `*_正交轮廓.json`。
用法：
    python outline.py 输出/粟塘B1_wall.ply 输出/雅德B1_wall.ply
"""
import os
import sys
import math
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import open3d as o3d
import contour as contour_mod       # 复用 trace_outline（栅格+闭运算+填孔+外沿轮廓）
import outer as outer_mod           # 复用 read_xyz / classify（获得红点=外墙高密度点）

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 用户版"连线规则"参数
GRID_SIZE = 0.3      # 网格
BRIDGE_DIST = 1.5    # 搭桥缝合距离
FLATTEN_TOL = 3.0    # 一维聚类拉平容差(线必须水平/竖直)
MORPH = 0.0          # 形态学滤毛刺半径(用户版 buffer(-r).buffer(r))；红点是细墙环，设0避免被削没
ORIENT_THRESH = 8.0  # 主方向角度阈值(°)：小 => 世界系水平/竖直；大 => 主方向系正交(斜着但线线90°)

ORTHO_TOL = 3.0   # 一维坐标聚类容差(m)：把落点吸附到同一直线 -> 直角
SIMPLIFY_TOL = 6.0  # 连线后 Douglas-Peucker 简化容差(m)：先把微台阶并成直段再直角化


def snap_1d_coordinates(values, tol=ORTHO_TOL):
    """一维坐标聚类：相近坐标吸附到均值，得到严格直角（来自用户版）。"""
    if len(values) == 0:
        return values
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    snapped = np.zeros_like(values)
    clusters = []
    curr = [0]
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] - sorted_vals[curr[0]] <= tol:
            curr.append(i)
        else:
            clusters.append(curr)
            curr = [i]
    clusters.append(curr)
    for c in clusters:
        avg = np.mean(sorted_vals[c])
        for idx in c:
            snapped[sorted_idx[idx]] = avg
    return snapped


def remove_collinear_points(coords):
    """剔除共线冗余点，仅保留转折点（来自用户版）。"""
    if len(coords) < 3:
        return np.array(coords)
    cleaned = [coords[0]]
    for i in range(1, len(coords) - 1):
        p1, p2, p3 = cleaned[-1], coords[i], coords[i + 1]
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        if abs(cross) < 1e-5:
            continue
        cleaned.append(p2)
    cleaned.append(coords[-1])
    return np.array(cleaned)


def _rotate(pts, angle, c):
    """绕中心 c 旋转 angle 弧度。返回 (N,2)。"""
    ca, sa = math.cos(angle), math.sin(angle)
    R = np.array([[ca, -sa], [sa, ca]])
    return (pts - c) @ R.T + c


def orthogonalize(poly, tol=ORTHO_TOL):
    """在"主方向(min外接矩形)坐标系"内对轮廓做 snap_1d 直角拉平，返回正交 Polygon。"""
    if poly.geom_type != "Polygon" or poly.exterior is None:
        return poly
    rc = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
    d = rc[1] - rc[0]
    ang = math.atan2(d[1], d[0]) % (math.pi / 2)
    if ang > math.pi / 4:
        ang -= math.pi / 2
    coords = np.asarray(poly.exterior.coords)[:-1]
    c = coords.mean(axis=0)
    rot = _rotate(coords, -ang, c)             # 转到主方向坐标系（墙对齐坐标轴）
    sx = snap_1d_coordinates(rot[:, 0], tol)
    sy = snap_1d_coordinates(rot[:, 1], tol)
    s = np.column_stack([sx, sy])
    back = _rotate(s, ang, c)                  # 转回原方向
    if len(s) >= 2 and np.allclose(back[0], back[-1]):
        back = back[:-1]
    newpoly = Polygon(np.vstack([back, back[:1]]))
    if not newpoly.is_valid:
        newpoly = newpoly.buffer(0)
    if newpoly.geom_type != "Polygon":
        return poly
    return newpoly


def orthogonal_connect(pts, grid_size=GRID_SIZE, bridge_dist=BRIDGE_DIST,
                       flatten_tol=FLATTEN_TOL, morph=MORPH):
    """用户版连线规则：网格->buffer融合->搭桥->形态学滤毛刺->snap_1d拉平->去共线。
    返回 最终坐标（线为水平/竖直）。pts 为 (N,2) 红点。"""
    if len(pts) < 3:
        raise ValueError("红点过少")
    grid_coords = np.floor(pts / grid_size).astype(np.int32)
    unique_cells = np.unique(grid_coords, axis=0)
    centers = unique_cells * grid_size + grid_size / 2.0
    base = MultiPoint(centers).buffer(grid_size / 2.0 * 1.05, cap_style=3)
    fused = base.buffer(bridge_dist, join_style=2).buffer(-bridge_dist, join_style=2)
    if morph:
        fused = fused.buffer(-morph, join_style=2).buffer(morph, join_style=2)
    if hasattr(fused, "geoms"):
        fused = max(fused.geoms, key=lambda p: p.area)
    raw_coords = np.array(fused.exterior.coords)[:-1]
    # 自动抉择：主方向角度小(轴对齐) => 世界系水平/竖直；角度大(整体倾斜) => 主方向系正交(线线90°)
    rc = np.asarray(fused.minimum_rotated_rectangle.exterior.coords)
    d = rc[1] - rc[0]
    ang = math.atan2(d[1], d[0]) % (math.pi / 2)
    if ang > math.pi / 4:
        ang -= math.pi / 2
    c = raw_coords.mean(axis=0)
    if abs(ang) <= math.radians(ORIENT_THRESH):
        ang = 0.0                                       # 轴对齐建筑：世界系水平/竖直
    if ang == 0.0:
        sx = snap_1d_coordinates(raw_coords[:, 0], tol=flatten_tol)
        sy = snap_1d_coordinates(raw_coords[:, 1], tol=flatten_tol)
        back = np.column_stack([sx, sy])
    else:
        rot = _rotate(raw_coords, -ang, c)              # 转到主方向(墙对齐轴)
        sx = snap_1d_coordinates(rot[:, 0], tol=flatten_tol)
        sy = snap_1d_coordinates(rot[:, 1], tol=flatten_tol)
        back = _rotate(np.column_stack([sx, sy]), ang, c)  # 转回：斜着但线线正交
    try:
        clean_poly = Polygon(np.vstack([back, back[:1]])).buffer(0)
        if clean_poly.geom_type == "MultiPolygon":
            clean_poly = max(clean_poly.geoms, key=lambda p: p.area)
        final = np.array(clean_poly.exterior.coords)
    except Exception:
        final = np.vstack([back, back[:1]])
    return remove_collinear_points(final)


def read_wall_xy(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("空点云: %s" % ply_path)
    return pts[:, :2]


def plot(wall_xy, coords, path, title):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(wall_xy[:, 0], wall_xy[:, 1], s=0.1, c="#c6dbef", alpha=0.2, label="墙点投影")
    ax.plot(coords[:, 0], coords[:, 1], color="red", lw=3.0, label="正交轮廓")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process(base, wall_ply, out_dir):
    print("\n========== 正交轮廓(用户连线规则): %s ==========" % wall_ply)
    wall_xy = read_wall_xy(wall_ply)                   # 连续可闭合的墙点作为连线输入
    if len(wall_xy) < 3:
        print("墙点过少，跳过")
        return
    coords = orthogonal_connect(wall_xy)               # 用户版连线规则(线必须水平/竖直)
    poly = Polygon(coords)
    info = {
        "file": wall_ply,
        "vertices": int(len(coords) - 1),
        "area_m2": round(float(poly.area), 2),
        "perimeter_m": round(float(poly.length), 2),
        "vertex_xy": [[round(float(x), 3), round(float(y), 3)] for x, y in coords[:-1]],
    }
    plot(wall_xy, coords, "%s/%s_正交轮廓.png" % (out_dir, base), "%s 正交轮廓" % base)
    with open("%s/%s_正交轮廓.json" % (out_dir, base), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("正交轮廓: 顶点=%d, 面积=%.1f m², 周长=%.1f m" % (info["vertices"], info["area_m2"], info["perimeter_m"]))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "输出")
    os.makedirs(out_dir, exist_ok=True)
    for wall_ply in sys.argv[1:]:
        base = os.path.splitext(os.path.basename(wall_ply))[0]
        base = base.replace("_wall", "") if base.endswith("_wall") else base
        process(base, wall_ply, out_dir)
    print("\n完成，输出目录: %s" % out_dir)


if __name__ == "__main__":
    main()
