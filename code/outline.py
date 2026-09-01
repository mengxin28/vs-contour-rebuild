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
from scipy.ndimage import uniform_filter1d

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
FLATTEN_TOL = 5.0    # 一维聚类拉平容差(线必须水平/竖直)(调大并干净台阶)
MORPH = 0.0          # 形态学滤毛刺半径(用户版 buffer(-r).buffer(r))；红点是细墙环，设0避免被削没
ORIENT_THRESH = 8.0  # 主方向角度阈值(°)：小 => 世界系水平/竖直；大 => 主方向系正交(斜着但线线90°)

# v0.22 分段拟合参数（保留斜线+圆弧）
TURN_EPS = 2.0       # 直段判据：平滑后转弯率|度| ≤ 此值
CORNER_EPS = 25.0    # 转角判据：原始转弯率|度| > 此值
SMOOTH_W = 9         # 转弯率平滑窗宽
SLANT_TOL = 2.0      # 斜线阈：偏离轴线(度)>=此值且长度>=SLANT_MIN_LEN 保留为斜线
SLANT_MIN_LEN = 2.0  # 斜线最短长度(m)
SLANT_RES_MAX = 0.8  # 斜线拟合最大残差(m)
ARC_R_MIN = 5.0      # 圆弧半径下限(m)
ARC_R_MAX = 200.0    # 圆弧半径上限(m)
ARC_MIN_SPAN = 25.0  # 圆弧最小跨度(度)
ARC_RESIDUAL_MAX = 0.8  # 圆弧拟合最大残差(m)
ARC_SAMPLES = 24     # 圆弧采样点数
STAIR_LEN = 4.0      # 基底"台阶段"判定：短边长度阈值(m)
PATCH_NEAR = 8.0     # 补丁拟合时取原始边界附近点的半径(m)(需覆盖真圆弧约等于R)

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


def _turns(coords):
    """每顶点转弯角(度, 环形平滑)。"""
    n = len(coords)
    e = coords - np.roll(coords, 1, axis=0)
    ang = np.arctan2(e[:, 1], e[:, 0])
    t = np.diff(np.append(ang, ang[:1]))
    return uniform_filter1d(np.degrees(t), size=SMOOTH_W, mode="wrap")


def segmentize(coords):
    """按转弯率把环形边界分成 (start,length,label) 各段；label 1=圆/斜段, 0=直段。
    转角顶点(|t|>CORNER_EPS)作为段与段的分界，不归属任何段。"""
    n = len(coords)
    t = _turns(coords)
    raw_t = t * 1.0  # 平滑后
    corner = np.abs(raw_t) > CORNER_EPS
    if corner.sum() == 0:
        lab = 1 if np.mean(np.abs(raw_t)) > TURN_EPS else 0
        return [(0, n, lab)]
    runs = []
    ci = np.flatnonzero(corner)
    for k in range(len(ci)):
        a = ci[k]
        b = ci[(k + 1) % len(ci)]
        length = (b - a - 1) % n
        if length < 3:                      # 过短段并入相邻，不入特征
            continue
        start = (a + 1) % n
        idx = (start + np.arange(length)) % n
        avg = np.mean(np.abs(raw_t[idx]))
        runs.append((start, length, 1 if avg > TURN_EPS else 0))
    return runs


def fit_line(pts):
    """最小二乘直线：返回 (点p0, 单位方向d)。"""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    d = vt[0]
    return c, d / np.linalg.norm(d)


def _line_orient(d):
    ang = math.degrees(math.atan2(d[1], d[0])) % 180.0
    dev = min(ang % 90.0, 90.0 - ang % 90.0)
    axis = 0 if (ang % 90.0) < 45.0 else 1   # 0=水平,1=竖直
    return ang, dev, axis


def fit_circle(pts):
    """Kasa 圆拟合，返回 (cx, cy, R, maxresid)。"""
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = pts[:, 0] ** 2 + pts[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, k = sol
    R = np.sqrt(k + cx * cx + cy * cy)
    resid = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - R)
    return cx, cy, R, float(resid.max())


def fit_circle_trimmed(sel):
    """带内点剔除的圆拟合：迭代去残差点，抗直墙点污染。"""
    cx, cy, R, res = fit_circle(sel)
    for _ in range(3):
        d = np.abs(np.hypot(sel[:, 0] - cx, sel[:, 1] - cy) - R)
        tol = max(res * 0.8, 0.6)
        sel2 = sel[d <= tol]
        if len(sel2) < 10:
            break
        cx, cy, R, res = fit_circle(sel2)
    return cx, cy, R, float(res)


def _corner_point(f1, f2, near):
    """相邻两特征求交点；失败返回 None。"""
    if f1[0] == "line" and f2[0] == "line":
        p1, d1 = f1[1][:2]
        p2, d2 = f2[1][:2]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-9:
            return None
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
        return p1 + t * d1
    # 线-圆 / 圆-线 / 圆-圆统一用"找特征2上离near最近且满足特征1"的近似：
    # 简化：用特征1与特征2在 near 附近各自投影的最小间距点——直接用 near 的投影
    if f1[0] == "line" and f2[0] == "arc":
        p, d = f1[1][:2]
        cx, cy, R = f2[1][:3]
        return _line_circle_pt(p, d, (cx, cy, R), near)
    if f1[0] == "arc" and f2[0] == "line":
        p, d = f2[1][:2]
        cx, cy, R = f1[1][:3]
        return _line_circle_pt(p, d, (cx, cy, R), near)
    return near                                # 圆-圆等罕见情况：直接用分界点


def _line_circle_pt(p, d, circle, near):
    cx, cy, R = circle
    pc = p - np.array([cx, cy])
    A = float(d @ d)
    B = 2.0 * float(d @ pc)
    C = float(pc @ pc) - R * R
    disc = max(B * B - 4 * A * C, 0.0)   # 相切时浮点误差可为负，截断为0取切点
    sq = math.sqrt(disc)
    cands = [p + t_ * d for t_ in [(-B + sq) / (2 * A), (-B - sq) / (2 * A)]]
    return min(cands, key=lambda q: float(np.linalg.norm(q - near)))


def _smooth_ring(coords, iters=3, size=5):
    """绕环均匀滤波，抹掉 buffer 边界锯齿，让转弯率局部化。"""
    c = np.asarray(coords, dtype=np.float64)
    for _ in range(iters):
        c = np.column_stack([uniform_filter1d(c[:, 0], size=size, mode="wrap"),
                             uniform_filter1d(c[:, 1], size=size, mode="wrap")])
    return c


def enhance_slants(base, raw_b, min_edge_len=5.0):
    """斜线检补（主攻斜交墙面）：遍历基底长边(≥5m)，取附近融合边界点拟合直线；
    若该处边界确为斜线(偏离轴线≥SLANT_TOL、长度≥SLANT_MIN_LEN、残差≤0.6m)，
    则用拟合斜线替换该边(端点投影到斜线上)。失败则原样保留。"""
    n = len(base)
    if n < 4:
        return base
    out = []
    for i in range(n):
        a, b_pt = base[i], base[(i + 1) % n]
        e = b_pt - a
        L = float(np.linalg.norm(e))
        if L < min_edge_len:
            out.append(a)
            continue
        d = e / L
        # 取"该边直线附近(≤2.5m 走廊, 投影 t∈[-2,L+2])"的融合边界点
        v = raw_b - a
        t = v @ d
        lat = np.abs(v[:, 0] * (-d[1]) + v[:, 1] * d[0])
        sel = raw_b[(np.abs(lat) <= 2.5) & (t >= -2.0) & (t <= L + 2.0)]
        out.append(a)
        if len(sel) < 8:
            continue
        p0, d2 = fit_line(sel)
        ang, dev, axis = _line_orient(d2)
        v = sel - p0
        tv = v @ d2
        chain_len = float(tv.max() - tv.min())
        res = float(np.abs((v @ d2) * 0 + np.linalg.norm(v - np.outer(tv, d2), axis=1)).max())
        if dev >= SLANT_TOL and chain_len >= SLANT_MIN_LEN and res <= SLANT_RES_MAX:
            a2 = p0 + float((a - p0) @ d2) * d2          # 端点投到斜线
            b2 = p0 + float((b_pt - p0) @ d2) * d2
            out.append(a2)
            out.append(b2)
    out = np.array(out)
    return out if len(out) >= 4 else base


def enhance_arcs_slants(base, raw, min_corner=40.0, max_corner=140.0):
    """[已停用-暂不主攻圆弧] 角上圆角(fillet)：遍历基底每个角，若角附近原始墙边能拟合出真实圆弧(R∈[5,200]、残差小、跨度足够)，
    则将角替换为"两切点之间"的圆弧采样；否则角度若为斜线(弦偏离轴线≥SLANT_TOL且长≥SLANT_MIN_LEN)则保留弦。
    失败/无特征则原样保留。raw: 融合边界(活动坐标系)。"""
    n = len(base)
    if n < 6:
        return base
    from scipy.spatial import cKDTree
    tree = cKDTree(raw)
    out = []
    for i in range(n):
        v_prev, v, v_next = base[i - 1], base[i], base[(i + 1) % n]
        e1 = v - v_prev
        e2 = v_next - v
        l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
        if l1 < 1e-9 or l2 < 1e-9:
            out.append(v)
            continue
        d1, d2 = e1 / l1, e2 / l2
        turn = math.degrees(math.atan2(d2[1], d2[0]) - math.atan2(d1[1], d1[0]))
        turn = (turn + 180.0) % 360.0 - 180.0
        should_fillet = (abs(turn) >= min_corner and abs(turn) <= max_corner
                         and l1 >= 6.0 and l2 >= 6.0)          # 只处理两邻边够长的真角
        is_arc = False
        if should_fillet:
            sel = raw[tree.query_ball_point(v, r=PATCH_NEAR)]
            if len(sel) >= 8:
                cx, cy, R, res = fit_circle_trimmed(sel)
                if ARC_R_MIN <= R <= ARC_R_MAX and res <= ARC_RESIDUAL_MAX:
                    j1 = _line_circle_pt_along(v, -d1, (cx, cy, R), v)
                    j2 = _line_circle_pt_along(v, d2, (cx, cy, R), v)
                    t1 = float(np.linalg.norm(j1 - v))
                    t2 = float(np.linalg.norm(j2 - v))
                    if 0.5 <= t1 <= l1 and 0.5 <= t2 <= l2:     # 切点须落在两邻边内
                        a0 = math.atan2(j1[1] - cy, j1[0] - cx)
                        a1 = math.atan2(j2[1] - cy, j2[0] - cx)
                        da = a1 - a0
                        da = (da + math.pi) % (2 * math.pi) - math.pi   # 短弧
                        if ARC_MIN_SPAN <= abs(math.degrees(da)) <= 170.0:
                            samples = [np.array([cx + R * math.cos(a0 + s_), cy + R * math.sin(a0 + s_)])
                                       for s_ in np.linspace(0.0, da, ARC_SAMPLES)[1:-1]]
                            out.append(np.array(j1))
                            out.extend(samples)
                            out.append(np.array(j2))
                            is_arc = True
        if not is_arc:
            out.append(v)                                  # 非弧：原样保留角
    out = np.array(out)
    return out if len(out) >= 4 else base


def _line_circle_pt_along(p, d, circle, near):
    """过 p 沿方向 d 的直线与圆(circle)的最近交点(near 附近)。"""
    cx, cy, R = circle
    pc = p - np.array([cx, cy])
    A = float(d @ d)
    B = 2.0 * float(d @ pc)
    C = float(pc @ pc) - R * R
    disc = max(B * B - 4 * A * C, 0.0)   # 相切时浮点误差可为负，截断为0取切点
    sq = math.sqrt(disc)
    cands = [p + t_ * d for t_ in [(-B + sq) / (2 * A), (-B - sq) / (2 * A)]]
    return min(cands, key=lambda q: float(np.linalg.norm(q - near)))


def orthogonal_connect(pts, grid_size=GRID_SIZE, bridge_dist=BRIDGE_DIST,
                       flatten_tol=FLATTEN_TOL, morph=MORPH):
    """用户版连线规则：网格->buffer融合->搭桥->形态学滤毛刺(可选)。
    接着 分段识别+拟合+重建：直线(含水平/竖直/小斜线) + 圆弧；失败回退 snap_1d 正交版。"""
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
    # 自动抉择：主方向角度小(轴对齐) => 世界系；角度大(整体倾斜) => 主方向系
    rc = np.asarray(fused.minimum_rotated_rectangle.exterior.coords)
    d = rc[1] - rc[0]
    ang = math.atan2(d[1], d[0]) % (math.pi / 2)
    if ang > math.pi / 4:
        ang -= math.pi / 2
    c = raw_coords.mean(axis=0)
    if abs(ang) <= math.radians(ORIENT_THRESH):
        ang = 0.0
    frame = raw_coords if ang == 0.0 else _rotate(raw_coords, -ang, c)
    # 1) 基底：v0.21 snap_1d 正交拉平（稳定）
    sx = snap_1d_coordinates(frame[:, 0], tol=flatten_tol)
    sy = snap_1d_coordinates(frame[:, 1], tol=flatten_tol)
    back = np.column_stack([sx, sy])
    try:
        cp = Polygon(np.vstack([back, back[:1]])).buffer(0)
        if cp.geom_type == "MultiPolygon":
            cp = max(cp.geoms, key=lambda p: p.area)
        base = np.array(cp.exterior.coords)[:-1]
    except Exception:
        base = back
    base = remove_collinear_points(base)
    # 2) 补丁：斜线检补（斜交墙面保留为直线；失败/无效则原样保留正交基底）
    cp_base_area = float(Polygon(np.vstack([base, base[:1]])).area)
    try:
        enhanced = enhance_slants(base, frame)
        p_enh = Polygon(np.vstack([enhanced, enhanced[:1]]))
        if not p_enh.is_valid:
            p_enh = p_enh.buffer(0)
        if p_enh.geom_type == "Polygon" and 0.6 * cp_base_area <= p_enh.area <= 1.4 * cp_base_area:
            base = enhanced
    except Exception:
        pass
    back = base if ang == 0.0 else _rotate(base, ang, c)
    return back


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
