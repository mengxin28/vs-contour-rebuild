# -*- coding: utf-8 -*-
"""
地下车库 外圈高密度点云高亮（阶段二·可视化，v0.7 密度用原始点）
输入：原始点云 `*.las` / `*.ply`。
目标：把**外圈高密度墙体点**用红色标出（不描轮廓）。
密度定义：**竖直堆叠密度** —— 同一 0.3m XY 小柱内的点数。墙体/柱在竖直方向叠满、密度高；
         地面/天花板只在顶底各一层、密度低。故"密度前5%"能聚到墙/柱。
判定 = 两者结合：竖直密度 前5%  AND  距外边界 ≤ BAND。
输出：`*_外圈点云.png`（红=外圈高密度墙，橙=高密度但不在外圈，浅蓝=其余）。
用法：
    python outer.py <原始.las/.ply ...>
例如：
    python outer.py ../粟塘B1.las ../雅德B1.las
"""
import os
import sys
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_closing, binary_fill_holes, distance_transform_edt,
                           gaussian_filter, sobel, label, binary_dilation)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import open3d as o3d

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

COL_GRID = 0.1    # 竖直密度小柱边(m)（基本单元 0.1m×0.1m）
GRID = 0.5        # 外圈位置栅格边(m)
CLOSE_ITER = 2
GLOBAL_PCT = 90   # 全局密度百位分：≥90% 为"全局前10%"
LOCAL_PCT = 90    # 局域密度百位分：≥该局部区域90% 为"局域前10%"
LOCAL_CELL = 12.0 # 局域比较窗口边长(m)
BAND = 1.0        # 外圈带宽(m)

# ------------------- v0.29 全局梯度法找墙/柱（绕车） -------------------
WALL_COL_GRID = 0.1  # 梯度密度场单元(m)(用户指定 0.1m)
WALL_SIGMA = 2.0     # 密度场高斯平滑 sigma(格=0.2m): 标定最优(墙/车分离度~1.5×)
WALL_G_THR = 0.25    # 梯度阈值(标定: σ=2 墙P50=0.19/车P50=0.13 ⇒ 0.25 居中)
WALL_D_THR = 0.30    # 平滑密度阈值(点/柱): 压掉车内低密度区的高梯度毛刺
WALL_CONN = 3        # 8连通连缝半径(格=0.3m)补点线断裂
CAR_MIN_MAJOR = 3.0  # 车体长轴剔除阈值(m): 大小在 3~8m 且长宽比<1.4 → 车
CAR_MAJOR_MAX = 8.0
CAR_ASPECT = 1.4
CAR_NEAR_WALL_M = 1.5  # 距墙带域 ≤1.5m 可视为墙侧延伸保留
COL_BAND_W = 5.0     # 墙带域最大宽度(m): 墙为细长带(≤烟囱/楼梯间宽)


def read_las_xyz(path, chunk=2_000_000):
    """自写 LAS 读取（未压缩 LAS 1.x），返回 (N,3) float64 米制坐标。"""
    with open(path, "rb") as fh:
        d = fh.read(400)
    assert d[0:4] == b"LASF", "不是有效 LAS: %s" % path
    off = struct.unpack("<I", d[96:100])[0]
    pt_len = d[105]
    sx, sy, sz = struct.unpack("<d", d[131:139])[0], struct.unpack("<d", d[139:147])[0], struct.unpack("<d", d[147:155])[0]
    ox, oy, oz = struct.unpack("<d", d[155:163])[0], struct.unpack("<d", d[163:171])[0], struct.unpack("<d", d[171:179])[0]
    rec = np.dtype({"names": ["X", "Y", "Z"], "formats": ["<i4", "<i4", "<i4"],
                    "offsets": [0, 4, 8], "itemsize": pt_len})
    total = (os.path.getsize(path) - off) // pt_len
    out, pos = [], 0
    while pos < total:
        n = min(chunk, total - pos)
        r = np.fromfile(path, dtype=rec, count=n, offset=off + pos * pt_len)
        out.append(np.column_stack([r["X"] * sx + ox, r["Y"] * sy + oy,
                                    r["Z"] * sz + oz]).astype(np.float64))
        pos += n
    return np.vstack(out) if out else np.empty((0, 3))


def read_xyz(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".las":
        return read_las_xyz(path)
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("空点云: %s" % path)
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


def detect_wall_columns(raw, g_thr=WALL_G_THR, d_thr=WALL_D_THR,
                        grid=WALL_COL_GRID, sigma=WALL_SIGMA,
                        conn=WALL_CONN):
    """v0.29 全局梯度法找墙面/柱子（绕车）。
    三联判据(用户方案 + 实测改进)：
      1) 0.1m XY 柱竖直堆叠密度(墙/柱叠满层高, 车只叠到车高) -> 高斯平滑 σ=0.3m;
      2) sobel 梯度幅值 G：墙边缘平台密度高(4~12点/柱), 车低(1~3点/柱) =>
         墙带梯度均值≈0.24 vs 车区≈0.17, 重叠大 ⇒ 必须叠加密度门槛;
         候选 = G≥G_THR 且 平滑密度 D≥D_THR (压掉车内低密度区的高梯度毛刺);
      3) 形态分型(单连通域归一): 细长带=墙 / 小团=柱 / 中椭圆团=车→剔除;
         车体各域间, 距墙带域≤CAR_NEAR_WALL_M 的视为墙侧延伸保留。
    返回 (wall_mask, column_mask, info)：与 raw 同长 bool 掩码。
    info: {g_thr, d_thr, grad_p90, den_p90, wall_pts, col_pts, car_cells}。
    旧 classify()(双密度+外圈) 保留用于回归对比/历史路径。"""
    xy = raw[:, :2]
    origin = xy.min(axis=0)
    g = np.floor((xy - origin) / grid).astype(np.int64)
    nx, ny = int(g[:, 0].max()) + 1, int(g[:, 1].max()) + 1
    lin = g[:, 1] * nx + g[:, 0]
    size = int(nx * ny)
    count = np.bincount(lin, minlength=size).reshape(ny, nx).astype(np.float64)
    # 1) 平滑密度场
    dens = gaussian_filter(count, sigma)
    # 2) sobel 梯度幅值 (保持 scipy 原刻度: 与诊断标定的 G_THR≈0.45 一致)
    gx = sobel(dens, axis=1)
    gy = sobel(dens, axis=0)
    grad = np.hypot(gx, gy)
    cand = (grad >= g_thr) & (dens >= d_thr)
    # 3) 连缝补线: 闭运算(膨胀→腐蚀)连接点线断口, 不改变候选本体
    if conn > 1:
        cand = binary_closing(cand, structure=np.ones((3, 3), dtype=bool),
                              iterations=conn - 1)
    # 4) 连通域, 逐域形状分型: 车(中椭圆团,w小) / 墙(细长带或宽≤COL_BAND_W) / 柱(小团)
    lab, ncomp = label(cand, structure=np.ones((3, 3), dtype=bool))
    comp_of_pt = lab.ravel()[lin]
    wall_ids, col_ids, car_ids = set(), set(), set()
    car_info = []       # (domain_id, center_xy, major, minor)
    for ci in range(1, ncomp + 1):
        if int((lab == ci).sum()) == 0:
            continue
        cells = lab == ci
        ys, xs = np.nonzero(cells)
        cxy = np.column_stack([origin[0] + (xs + 0.5) * grid,
                               origin[1] + (ys + 0.5) * grid])
        center = cxy.mean(axis=0)
        q = cxy - center
        d0 = np.array([1.0, 0.0])
        if len(cxy) >= 3:
            _, _, vt = np.linalg.svd(q, full_matrices=False)
            d0 = vt[0] / np.linalg.norm(vt[0])
        d1 = np.array([-d0[1], d0[0]])
        p0 = q @ d0
        p1 = q @ d1
        major = float(np.percentile(p0, 99) - np.percentile(p0, 1))
        minor = float(np.percentile(p1, 97.5) - np.percentile(p1, 2.5))
        aspect = major / max(minor, 1e-6)
        if CAR_MIN_MAJOR <= major <= CAR_MAJOR_MAX and aspect < CAR_ASPECT:
            car_ids.add(ci)                 # 车: 中椭圆团(4~6m×2m)
            car_info.append((ci, center, major, minor))
        elif major >= CAR_MIN_MAJOR or aspect >= CAR_ASPECT or major >= 2.0:
            wall_ids.add(ci)                # 墙: 细长带 / 大块墙体
        else:
            col_ids.add(ci)                 # 柱: 小团(≲2m)
    # 5) 车域靠墙近的并入墙侧(墙侧延伸, 例: 过道旁长条设备带)
    if car_info:
        wall_centers = np.array([np.nonzero(lab == ci) and
                                 np.column_stack([origin[0] + (np.nonzero(lab == ci)[1] + 0.5) * grid,
                                                  origin[1] + (np.nonzero(lab == ci)[0] + 0.5) * grid]).mean(axis=0)
                                 for ci in wall_ids]) if wall_ids else np.empty((0, 2))
        if len(wall_centers):
            from scipy.spatial import cKDTree as _T_outer
            tree_w = _T_outer(wall_centers)
            for ci, center, major, minor in car_info:
                dist, _ = tree_w.query(center)
                if float(dist) <= CAR_NEAR_WALL_M:
                    car_ids.discard(ci)
                    wall_ids.add(ci)
    wall_mask = np.isin(comp_of_pt, list(wall_ids))
    col_mask = np.isin(comp_of_pt, list(col_ids))
    info = {"g_thr": g_thr, "d_thr": d_thr,
            "grad_p90": round(float(np.percentile(grad, 90)), 3),
            "den_p90": round(float(np.percentile(dens, 90)), 3),
            "candidate_cells": int(cand.sum()),
            "components": int(ncomp),
            "wall_domains": len(wall_ids),
            "col_domains": len(col_ids),
            "car_domains": len(car_ids),
            "wall_pts": int(wall_mask.sum()), "col_pts": int(col_mask.sum())}
    return wall_mask, col_mask, info


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
    # 全局前10%
    global_thr = np.percentile(density, global_pct)
    global_top = density >= global_thr
    # 局域前10%：以 12m 网格为"局部区域"，点密度≥该区域90分位
    # (排序分组再求分位，避免逐区域 np.where 的 O(N×区域) 慢循环)
    lc = np.floor(xy / local_cell).astype(np.int64)
    lkey = lc[:, 0] * 1000000 + lc[:, 1]
    order = np.argsort(lkey, kind="stable")
    start = np.unique(lkey[order], return_index=True)[1]
    N = len(xy)
    local_top = np.zeros(N, dtype=bool)
    for k, s in enumerate(start):
        e = start[k + 1] if k + 1 < len(start) else N
        ids = order[s:e]
        local_top[ids] = density[ids] >= np.percentile(density[ids], local_pct)
    dual = global_top & local_top       # 全局前10% 且 局域前10%
    # 2) 外圈位置：到外边界距离
    mask, g, (x0, y0), grid = footprint_mask(xy)
    dist_cells = distance_transform_edt(mask)
    band_cells = band / grid
    inb = ((g[:, 0] >= 0) & (g[:, 0] < mask.shape[1]) &
           (g[:, 1] >= 0) & (g[:, 1] < mask.shape[0]))
    d = np.zeros(len(xy))
    d[inb] = dist_cells[g[inb, 1], g[inb, 0]]
    outer = inb & (d <= band_cells)
    red = dual & outer                  # 双密度门槛 且 在外圈
    orange = dual & ~outer              # 双密度门槛 但不在外圈
    info = {"global_thr": int(global_thr), "global_top": int(global_top.sum()),
            "dual": int((dual).sum()), "red": int(red.sum()), "orange": int(orange.sum())}
    return red, orange, info


def plot(raw, red, orange, path, title, max_pts=2_000_000):
    """点染；大点云对"其余点"抽样以提速。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    rest = np.where(~red)[0]
    if len(rest) > max_pts:                     # 抽样背景点，红/橙全保留
        rest = rest[::2 * len(rest) // max_pts]
    ax.scatter(raw[rest, 0], raw[rest, 1], s=0.3, c="#c6dbef", marker=".", alpha=0.5, label="其余点")
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


def plot_gradient(raw, wall, col, path, title, max_pts=2_000_000):
    """v0.29 梯度法结果: 红=墙面, 橙=柱, 灰=其余(车/地面被剔除)。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    rest = np.where(~(wall | col))[0]
    if len(rest) > max_pts:
        rest = rest[::2 * len(rest) // max_pts]
    ax.scatter(raw[rest, 0], raw[rest, 1], s=0.3, c="#cccccc", marker=".", alpha=0.4,
               label="其余(车/地面, 梯度不高)")
    ax.scatter(raw[col, 0], raw[col, 1], s=0.9, c="#ff7f0e", marker=".", alpha=0.85,
               label="柱子(小团)")
    ax.scatter(raw[wall, 0], raw[wall, 1], s=1.0, c="#d62728", marker=".", alpha=0.9,
               label="墙面(细长高梯度带)")
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


def process(base, ply, out_dir, gradient=False):
    print("\n========== 外圈高亮(原始点竖直密度+位置): %s ==========" % ply)
    raw = read_xyz(ply)
    if gradient:
        wall, col, info = detect_wall_columns(raw)
        plot_gradient(raw, wall, col, "%s/%s_梯度墙柱.png" % (out_dir, base),
                      "%s 梯度法墙/柱(0.1m单元)  G_THR=%.2f D_THR=%.1f" % (base, info["g_thr"], info["d_thr"]))
        n = len(raw)
        print("梯度法: 候选格=%d (梯度P90=%.3f/密度P90=%.1f); 墙域=%d 柱域=%d 车域=%d; "
              "墙点=%d (%.1f%%) 柱点=%d" % (
                  info["candidate_cells"], info["grad_p90"], info["den_p90"],
                  info["wall_domains"], info["col_domains"], info["car_domains"],
                  info["wall_pts"], 100 * info["wall_pts"] / n, info["col_pts"]))
        return
    red, orange, info = classify(raw)
    plot(raw, red, orange, "%s/%s_外圈点云.png" % (out_dir, base), "%s 外圈高密度点云(红)" % base)
    n = len(raw)
    print("全局密度阈值=%d点/柱; 全局前%.0f%%=%d点; 双重(全局前%.0f%%且局域前%.0f%%)=%d点; "
          "红(外圈墙)=%d (%.1f%%); 橙(双密度但不在外圈)=%d" % (
              info["global_thr"], GLOBAL_PCT, info["global_top"], GLOBAL_PCT, LOCAL_PCT,
              info["dual"], info["red"], 100 * info["red"] / n, info["orange"]))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    use_gradient = "--gradient" in sys.argv
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "输出")
    os.makedirs(out_dir, exist_ok=True)
    for ply in files:
        base = os.path.splitext(os.path.basename(ply))[0]
        process(base, ply, out_dir, gradient=use_gradient)
    print("\n完成，输出目录: %s (梯度法=%s)" % (out_dir, use_gradient))


if __name__ == "__main__":
    main()
