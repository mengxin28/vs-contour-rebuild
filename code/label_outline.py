# -*- coding: utf-8 -*-
"""
红+紫点云的最贴合简洁外包多边形（v0.30）
输入：源点云（CLEAN_UNDER_GROUND.ply / RESULT_B1.ply）+ 阶段一的 `*_wall.ply`。
流程：
  1. classify(return_ramp=True) 得 红点(外墙高密度) ∪ 紫点(坡道补充)——正是
     "坡道优化包"两张图里的红/紫；
  2. orthogonal_connect(wall_xy, guide=红∪紫) —— 复用 v0.27 管线:
     连线+snap正交+hug_outer_edge 贴外墙皮(引导=红∪紫, 坡道被自然包入);
  3. 后处理简洁化: remove_collinear_points + Douglas-Peucker(simplify 0.8m,
     面积变化>2%则回退), 保证正交长边、顶点少、闭合有效;
  4. 输出 `*_贴合轮廓.png`(红/紫点 + 黑粗轮廓线) + `*_贴合轮廓.json`。
用法：
    python code/label_outline.py CLEAN_UNDER_GROUND.ply 输出/CLEAN_UNDER_GROUND_wall.ply
    python code/label_outline.py RESULT_B1.ply 输出/RESULT_B1_wall.ply
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import open3d as o3d

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import outer as outer_mod      # read_xyz / classify(return_ramp=True)
import outline as ol           # orthogonal_connect(用户连线规则,含hug贴墙) / remove_collinear_points

SIMPLIFY_TOL = 0.8   # Douglas-Peucker 简化容差(m)(正交长边已拉平, DP只去微台阶)
AREA_GUARD = 0.02    # 简化后面积偏差 >2% 则回退未简化版本


def read_wall_xy(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("空点云: %s" % ply_path)
    return pts[:, :2]


def attach_ramp_protrusion(coords, ramp_xy, max_move=10.0, tol=0.8):
    """紫点带外伸贴附(坡道段): 对轮廓每条边, 收集其外法向带的紫点;
    若外侧紫点带距边 ≥ ramp_min_gap(1.2m) 且集中, 则在该边的沿边区间
    [a0,a1] 插入"贴外皮台阶"——外移到紫点带的最外皮位置(正交台阶)。
    与 v0.28 bump_notch_local 同构, 但引导=紫点(真实坡道点位,非推测矩形)。
    返回修正点列(闭合性由调用方校验); 完全在轮廓内的紫点(如 CLEAN 中部
    短条)自动无影响(距边<阈值)。
    """
    n = len(coords)
    if n < 4 or len(ramp_xy) < 10:
        return coords
    # CCW 定向(右侧=外侧)
    area = 0.5 * float(np.sum(coords[:, 0] * np.roll(coords[:, 1], -1) -
                              np.roll(coords[:, 0], -1) * coords[:, 1]))
    ring = coords.copy()
    if area < 0:
        ring = ring[::-1]
    from scipy.spatial import cKDTree as _T3
    closed = np.vstack([ring, ring[:1]])
    seg = np.diff(closed, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    # 每边采样(0.4m)用于邻近判定
    samp, eids = [], []
    for i in range(len(seg)):
        L = seglen[i]; ns = max(int(L / 0.4), 1)
        for k in range(ns):
            samp.append(closed[i] + seg[i] * (k / ns))
            eids.append(i)
    samp = np.array(samp); eids = np.array(eids)
    tree_s = _T3(samp)
    per_edge = {}
    for p in ramp_xy:
        _d, si = tree_s.query(p, k=1)
        eid = int(eids[si])
        per_edge.setdefault(eid, []).append(p)
    out = []
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        L = float(np.linalg.norm(b - a))
        out.append(a)
        arr = per_edge.get(i, [])
        if L < 1e-9 or not arr or len(arr) < 8:
            continue
        t = (b - a) / L
        nout = np.array([t[1], -t[0]])                      # 右侧=外侧
        pts = np.array(arr)
        along = (pts - a) @ t
        lat = (pts - a) @ nout
        # 只处理"紫点明显凸出边外侧"的情况(距边>=1.2m, 最外皮<=max_move)
        if np.median(lat) < 1.2 or lat.max() > max_move:
            continue
        a0 = float(np.clip(along.min() - 0.5, 0.0, L))
        a1 = float(np.clip(along.max() + 0.5, 0.0, L))
        if a1 - a0 < 2.0:
            continue
        lat_new = float(np.percentile(lat, 97.5))            # 紫点带最外皮
        # 正交台阶: a → a0 → (a0+lat_new) → (a1+lat_new) → a1 → b
        out.append(a + t * a0)
        out.append(a + t * a0 + nout * lat_new)
        out.append(a + t * a1 + nout * lat_new)
        out.append(a + t * a1)
    out.append(ring[-1])
    return np.array(out)


def plot_fit(raw, red, ramp, coords, path, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    rest = np.where(~(red | ramp))[0]
    if len(rest) > 2_000_000:
        rest = rest[::2 * len(rest) // 2_000_000]
    ax.scatter(raw[rest, 0], raw[rest, 1], s=0.25, c="#c6dbef", marker=".",
               alpha=0.45, label="其余点")
    ax.scatter(raw[ramp, 0], raw[ramp, 1], s=1.4, c="#9467bd", marker=".",
               alpha=0.95, label="坡道补充点(紫)")
    ax.scatter(raw[red & ~ramp, 0], raw[red & ~ramp, 1], s=1.1, c="#d62728",
               marker=".", alpha=0.9, label="外圈高密度点(红)")
    ax.plot(coords[:, 0], coords[:, 1], color="black", lw=3.0, label="贴合轮廓(闭合)")
    ax.plot(coords[:, 0], coords[:, 1], "o", color="black", ms=3.0, alpha=0.85)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", markerscale=6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process(source_ply, wall_ply, out_dir):
    base = os.path.splitext(os.path.basename(source_ply))[0]
    print("\n========== 贴合简洁外包多边形: %s ==========" % base)
    raw = outer_mod.read_xyz(source_ply)
    red, orange, info, ramp = outer_mod.classify(raw, return_ramp=True)
    guide = raw[red | ramp][:, :2]                     # 红∪紫引导贴墙/包坡道
    print("红点=%d 紫点=%d -> 引导=%d" % (int((red & ~ramp).sum()), int(ramp.sum()), len(guide)))
    wall_xy = read_wall_xy(wall_ply)
    coords = ol.orthogonal_connect(wall_xy, guide=guide)
    coords = ol.remove_collinear_points(coords)
    # 坡道外皮贴附: 紫点带凸出轮廓的边 -> 正交外伸台阶
    ramp_xy = raw[ramp][:, :2]
    fixed = attach_ramp_protrusion(coords, ramp_xy)
    p_f = Polygon(np.vstack([fixed, fixed[:1]]))
    if not p_f.is_valid:
        p_f = p_f.buffer(0)
    if p_f.geom_type == "Polygon":
        coords = ol.remove_collinear_points(np.asarray(p_f.exterior.coords)[:-1])
    # 简洁化: DP 简化(面积变化>2% 回退)
    poly = Polygon(np.vstack([coords, coords[:1]]))
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda gp: gp.area)
    simp = poly.simplify(SIMPLIFY_TOL, preserve_topology=True)
    if simp.geom_type == "Polygon" and simp.area >= poly.area * (1.0 - AREA_GUARD):
        poly = simp
    coords = np.asarray(poly.exterior.coords)[:-1]
    n = len(coords)
    print("轮廓: 顶点=%d, 面积=%.2f m², 周长=%.2f m" % (
        n, poly.area, poly.length))
    info_out = {
        "file": os.path.basename(source_ply),
        "wall_ply": os.path.basename(wall_ply),
        "vertices": n,
        "area_m2": round(float(poly.area), 2),
        "perimeter_m": round(float(poly.length), 2),
        "points": {"red": int((red & ~ramp).sum()), "ramp_purple": int(ramp.sum()),
                   "guide": int(len(guide))},
        "ramp_info": info.get("ramp", {}),
        "vertex_xy": [[round(float(x), 3), round(float(y), 3)] for x, y in coords],
    }
    plot_fit(raw, red, ramp, np.vstack([coords, coords[:1]]),
             "%s/%s_贴合轮廓.png" % (out_dir, base), "%s 贴合简洁外包多边形" % base)
    with open("%s/%s_贴合轮廓.json" % (out_dir, base), "w", encoding="utf-8") as f:
        json.dump(info_out, f, ensure_ascii=False, indent=2)
    print("输出: %s/%s_贴合轮廓.png / .json" % (out_dir, base))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "输出")
    os.makedirs(out_dir, exist_ok=True)
    process(sys.argv[1], sys.argv[2], out_dir)


if __name__ == "__main__":
    main()
