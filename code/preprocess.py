# -*- coding: utf-8 -*-
"""
地下车库点云预处理（阶段一）
目标：把外墙点云从车库原始点云中剥离，投影到水平面得到俯视轮廓。

流程：LAS/PLY(0.5m 体素降采样) -> 去孤立噪点 -> 只保留最大聚合体
      -> 法向量判别去掉地面/天花板 -> 投影到水平面
      -> 输出 PLY + 可视化图 + 0.3m 热力密度图。

用法：
    python preprocess.py <点云文件...>    (支持 .las 和 .ply)
例如：
    python preprocess.py ../粟塘B1.las ../雅德B1.las
    python preprocess.py ../CLEAN_UNDER_GROUND.ply ../RESULT_B1.ply
"""
import os
import sys
import json
import struct
import numpy as np
from scipy.ndimage import convolve, label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 让图表正确显示中文（Windows 常见中文字体兜底）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import open3d as o3d

try:
    sys.stdout.reconfigure(encoding="utf-8")  # 避免中文日志乱码
except Exception:
    pass

VOXEL = 0.5        # 降采样体素边长(m)
MIN_NEIGHBORS = 3  # 去噪：26-邻域占用数低于此阈值判为孤立点
NORM_RADIUS = 1.2  # 法向估计邻域半径(m)
NORM_MAXNN = 30
HORIZ_TOL = 0.85   # |normal.z| > 此值判为水平面(地面/天花板)，剔除
DENSITY_GRID = 0.3 # 热力密度图栅格边长(m)


def read_las_header(path):
    """读取 LAS 头部，返回 (offset, pt_fmt, pt_len, scale, offset_xyz)。"""
    with open(path, "rb") as fh:
        d = fh.read(400)
    assert d[0:4] == b"LASF", "不是有效 LAS 文件"
    off = struct.unpack("<I", d[96:100])[0]
    pt_fmt = d[104]
    pt_len = d[105]
    scale = (struct.unpack("<d", d[131:139])[0],
             struct.unpack("<d", d[139:147])[0],
             struct.unpack("<d", d[147:155])[0])
    origin = (struct.unpack("<d", d[155:163])[0],
              struct.unpack("<d", d[163:171])[0],
              struct.unpack("<d", d[171:179])[0])
    return off, pt_fmt, pt_len, scale, origin


def read_las_voxel(path, voxel=VOXEL, chunk=2_000_000):
    """分块读取 LAS，做体素降采样。返回 (代表点xyz(N,3) float64, 体素键(N,3) int64)。"""
    off, pt_fmt, pt_len, scale, origin = read_las_header(path)
    xs, ys, zs = scale
    xo, yo, zo = origin
    rec = np.dtype({"names": ["X", "Y", "Z"],
                    "formats": ["<i4", "<i4", "<i4"],
                    "offsets": [0, 4, 8],
                    "itemsize": pt_len})
    total = (os.path.getsize(path) - off) // pt_len
    reps = []  # 每个元素: [cell_x,cell_y,cell_z, x,y,z]
    pos = 0
    while pos < total:
        n = min(chunk, total - pos)
        data = np.fromfile(path, dtype=rec, count=n, offset=off + pos * pt_len)
        x = data["X"] * xs + xo
        y = data["Y"] * ys + yo
        z = data["Z"] * zs + zo
        xyz = np.column_stack([x, y, z]).astype(np.float64)
        cells = np.floor(xyz / voxel).astype(np.int64)
        u, idx = np.unique(cells, axis=0, return_index=True)
        reps.append(np.column_stack([u, xyz[idx]]))
        pos += n
    if not reps:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    arr = np.vstack(reps)
    cells, xyz = arr[:, :3].astype(np.int64), arr[:, 3:6]
    # 跨块去重：同体素取第一次出现的代表点
    _, first = np.unique(cells, axis=0, return_index=True)
    return xyz[first], cells[first]


def voxelize(xyz, voxel=VOXEL):
    """对完整点阵做体素降采样。返回 (代表点(N,3), 体素键(N,3) int64)。"""
    cells = np.floor(np.asarray(xyz) / voxel).astype(np.int64)
    u, idx = np.unique(cells, axis=0, return_index=True)
    return np.asarray(xyz)[idx], u


def read_ply_points(path):
    """用 open3d 读取 PLY（兼容带 RGB / 不带 RGB），返回 (N,3) 的点阵。"""
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("PLY 文件中没有点云: %s" % path)
    return pts


def load_source(path):
    """按扩展名读取并做 0.5m 体素降采样，统一返回 (代表点, 体素键, 原始点数为None/LAS点数)。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".las":
        xyz, cells = read_las_voxel(path)
        return xyz, cells
    if ext == ".ply":
        full = read_ply_points(path)
        xyz, cells = voxelize(full)
        return xyz, cells
    raise ValueError("不支持的文件类型: %s" % ext)


def read_las_full(path, chunk=2_000_000):
    """读取 LAS 全部点 (N,3) float32，分块累积降低内存峰值。"""
    off, pt_fmt, pt_len, scale, origin = read_las_header(path)
    xs, ys, zs = scale
    xo, yo, zo = origin
    rec = np.dtype({"names": ["X", "Y", "Z"],
                    "formats": ["<i4", "<i4", "<i4"],
                    "offsets": [0, 4, 8],
                    "itemsize": pt_len})
    total = (os.path.getsize(path) - off) // pt_len
    out = []
    pos = 0
    while pos < total:
        n = min(chunk, total - pos)
        d = np.fromfile(path, dtype=rec, count=n, offset=off + pos * pt_len)
        out.append(np.column_stack([d["X"] * xs + xo,
                                    d["Y"] * ys + yo,
                                    d["Z"] * zs + zo]).astype(np.float32))
        pos += n
    return np.vstack(out) if out else np.empty((0, 3), dtype=np.float32)


def load_raw_full(path):
    """按扩展名返回原始分辨率点 (N,3) float32（用于密度统计）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".las":
        return read_las_full(path)
    if ext == ".ply":
        return read_ply_points(path).astype(np.float32)
    raise ValueError("不支持的文件类型: %s" % ext)


def denoise_and_largest(cells, min_neighbors=MIN_NEIGHBORS):
    """在体素占用网格上：去孤立噪点 + 只保留最大连通聚合体。
    返回 (keep_mask(按 cells 顺序的bool), grid 坐标偏移 shift)。"""
    mn = cells.min(axis=0) - 1
    shift = -mn  # 每个方向 +2 padding
    idx = cells + shift
    nx, ny, nz = idx.max(axis=0) + 1
    grid = np.zeros((nx, ny, nz), dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # 1) 去噪：统计 26-邻域占用数，过低视为孤立噪点
    kern = np.ones((3, 3, 3), dtype=np.uint8)
    kern[1, 1, 1] = 0
    cnt = convolve(grid.astype(np.uint8), kern, mode="constant")  # uint8 够用(<=26)
    denoise_grid = grid & (cnt >= min_neighbors)

    # 2) 只保留最大连通聚合体（26-邻域连通）
    lab, nlab = label(denoise_grid, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # 背景
    largest = int(sizes.argmax())
    comp_grid = lab == largest

    keep_mask = comp_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
    return keep_mask, shift


def remove_floor_ceiling(xyz):
    """用法向量剔除水平地面/天花板，保留立面(墙/柱)。返回保留的bool掩码。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=NORM_RADIUS, max_nn=NORM_MAXNN))
    normals = np.asarray(pcd.normals)
    keep = np.abs(normals[:, 2]) <= HORIZ_TOL
    return keep


def z_report(z, bins=40, title="Z"):
    """打印 Z 分布峰值，用于确认地面/天花板所在平面。"""
    h, edges = np.histogram(z, bins=bins)
    peaks = np.argsort(h)[-3:][::-1]
    lines = ["%s histogram (%.2f~%.2f m), 最大3个峰:" % (title, z.min(), z.max())]
    for p in peaks:
        if h[p] == 0:
            continue
        lines.append("    z=%.2f~%.2f : %d 点" % (edges[p], edges[p + 1], h[p]))
    print("\n".join(lines))


def save_ply(xyz, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz))
    o3d.io.write_point_cloud(path, pcd)


def plot_topview(xy, path, title, color=None, cbar_label=""):
    """俯视 XY 散点图（等比例）。xy 为 (N,2)。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    if color is None:
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="#1f77b4", marker=".")
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=0.5, c=color, cmap="viridis", marker=".")
        fig.colorbar(sc, ax=ax, label=cbar_label)
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * 1.05
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_perspective(xyz, path, title):
    """3D 透视散点图，按 Z 着色以展示立面竖直墙/柱的高度分布。"""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.5, c=xyz[:, 2], cmap="viridis", marker=".")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Z (m)")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2])))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def density_heatmap(raw_xyz, wall_cells, shift, voxel, grid, path, title):
    """用原始分辨率点统计墙体结构的密度，栅格化到 grid×grid(m) 并渲染热力图。
    raw_xyz: 原始点 (N,3); wall_cells: 墙体 0.5m 体素键 (M,3); shift: 体素网格偏移。"""
    idx = wall_cells + shift
    shape = idx.max(axis=0) + 1
    wgrid = np.zeros(shape, dtype=bool)
    wgrid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    # 原始点 -> 0.5m 体素，仅保留落在墙体体素内的点
    g = np.floor(raw_xyz / voxel).astype(np.int64) + shift
    inb = ((g[:, 0] >= 0) & (g[:, 0] < shape[0]) &
           (g[:, 1] >= 0) & (g[:, 1] < shape[1]) &
           (g[:, 2] >= 0) & (g[:, 2] < shape[2]))
    g_safe = np.where(inb[:, None], g, 0)
    wall_raw = raw_xyz[wgrid[g_safe[:, 0], g_safe[:, 1], g_safe[:, 2]] & inb]
    if len(wall_raw) == 0:
        print("  墙体原始点数为 0，跳过热力图")
        return None
    # 0.3m 栅格密度
    x0, y0 = wall_raw[:, 0].min(), wall_raw[:, 1].min()
    cxj = np.floor((wall_raw[:, 0] - x0) / grid).astype(np.int64)
    cyj = np.floor((wall_raw[:, 1] - y0) / grid).astype(np.int64)
    nx, ny = int(cxj.max()) + 1, int(cyj.max()) + 1
    hist = np.zeros((ny, nx), dtype=np.int64)
    np.add.at(hist, (cyj, cxj), 1)
    peak = int(hist.max())
    print("  墙体原始点=%d, 0.3m栅格峰值密度=%d 点/格" % (len(wall_raw), peak))
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(hist, origin="lower",
                   extent=(x0, x0 + nx * grid, y0, y0 + ny * grid),
                   cmap="inferno", norm=LogNorm(vmin=1, vmax=max(peak, 2)),
                   aspect="equal", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="点数/栅格(log)")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return peak


def process(name, path, out_dir):
    base = name
    print("\n========== 处理: %s ==========" % path)

    # 1) 读取 + 0.5m 体素降采样
    xyz, cells = load_source(path)
    print("0.5m 体素降采样后代表点数=%d" % len(cells))
    if len(cells) == 0:
        print("无数据，跳过")
        return

    # 处理前俯视图（按 Z 着色）
    plot_topview(xyz[:, :2], "%s/%s_俯视_原始.png" % (out_dir, base),
                 "%s 降采样后原始点云(按Z着色)" % base, color=xyz[:, 2], cbar_label="Z (m)")

    # 2) 去噪 + 最大聚合体
    keep_mask, shift = denoise_and_largest(cells)
    comp_cells = cells[keep_mask]
    comp_xyz = xyz[keep_mask]
    print("去噪+保留最大连通聚合体后代表点=%d (剔除孤立/小簇 %d)" %
          (len(comp_xyz), len(cells) - len(comp_xyz)))

    # 3) 去地面/天花板
    z_report(comp_xyz[:, 2], title="最大聚合体 Z")
    wall_keep = remove_floor_ceiling(comp_xyz)
    wall_cells = comp_cells[wall_keep]
    wall_xyz = comp_xyz[wall_keep]
    print("法向量剔除地面/天花板后=%d (其中水平面点删除 %d)" % (len(wall_xyz), (~wall_keep).sum()))

    # 4) 投影到水平面 (Z 归零)
    wall_xy = wall_xyz[:, :2].copy()
    wall_flat = np.column_stack([wall_xy, np.zeros(len(wall_xy))])

    save_ply(wall_flat, "%s/%s_wall.ply" % (out_dir, base))
    plot_topview(wall_xy, "%s/%s_俯视_外墙.png" % (out_dir, base),
                 "%s 外墙俯视图" % base)
    if len(wall_xyz) > 0:
        plot_perspective(wall_xyz, "%s/%s_透视_外墙.png" % (out_dir, base),
                         "%s 外墙3D(去掉地面/天花板)" % base)

    # 5) 0.3m 热力密度图（用原始点，限墙体结构内）
    density_peak = None
    if len(wall_cells) > 0:
        raw = load_raw_full(path)
        density_peak = density_heatmap(raw, wall_cells, shift, VOXEL, DENSITY_GRID,
                                       "%s/%s_热力密度图.png" % (out_dir, base),
                                       "%s 外墙密度热力图(%.1fm栅格)" % (base, DENSITY_GRID))

    bbox = {"Xmin": float(wall_xy[:, 0].min()), "Xmax": float(wall_xy[:, 0].max()),
            "Ymin": float(wall_xy[:, 1].min()), "Ymax": float(wall_xy[:, 1].max())}
    print("外墙投影 bbox: %.1fm x %.1fm" % (bbox["Xmax"] - bbox["Xmin"], bbox["Ymax"] - bbox["Ymin"]))
    return {"voxel_points": int(len(cells)), "component_points": int(len(comp_xyz)),
            "wall_points": int(len(wall_xy)),
            "density_grid_m": DENSITY_GRID, "density_peak": density_peak, "bbox": bbox}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "输出")
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for las in sys.argv[1:]:
        name = os.path.splitext(os.path.basename(las))[0]
        summary[name] = process(name, las, out_dir)
    with open(os.path.join(out_dir, "汇总.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n完成，输出目录: %s" % out_dir)


if __name__ == "__main__":
    main()
