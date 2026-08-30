# 地下车库外墙轮廓重建（vs轮廓重建）

把地下车库点云（LAS / PLY）里的**外部墙体轮廓**提取出来，并逐步做规则化处理。

当前管线（`code/preprocess.py`，阶段一：预处理 + 投影 + 密度图）：

```
读取 LAS/PLY
  -> 0.5m 体素降采样
  -> 去孤立噪点
  -> 只保留最大连通聚合体
  -> 法向量判别去掉地面/天花板（保留立面竖直墙/柱）
  -> 投影到水平面（Z 归零）
  -> 输出外墙点云(PLY) + 俯视/透视 + 0.3m 热力密度图
```

## 依赖
- Python 3.12
- `numpy`, `scipy`, `open3d`, `matplotlib`（用 `pip install numpy scipy open3d matplotlib`）

## 用法
```bash
python code/preprocess.py <点云文件...>
# 支持 .las 和 .ply，可一次传入多个文件
python code/preprocess.py 粟塘B1.las 雅德B1.las
python code/preprocess.py CLEAN_UNDER_GROUND.ply RESULT_B1.ply
```

默认参数：降采样 `0.5m`、密度栅格 `0.3m`。线程/阈值见 `code/preprocess.py` 顶部常量
（`VOXEL`、`MIN_NEIGHBORS`、`NORM_RADIUS`、`HORIZ_TOL`、`DENSITY_GRID`）。

## 输出（写入 `输出/`）
每个源文件 `NAME` 生成：
- `NAME_wall.ply` —— 投影后的外墙点云（Z 归零）
- `NAME_俯视_外墙.png` —— 外墙俯视轮廓
- `NAME_俯视_原始.png` —— 降采样后整车点云（按 Z 着色）
- `NAME_透视_外墙.png` —— 外墙 3D（按高度着色）
- `NAME_热力密度图.png` —— 0.3m 栅格密度热力图（外墙墙带高亮）
- `汇总.json` —— 各阶段点数、bbox、密度峰值

## 版本历史
每版的功能、用法与用户评价见 `版本记录.md`。

> 注：`.git/info/exclude` 已本地忽略 `输出/`、`.las/.laz`、根目录 `.ply` 等大体积数据，
> 本仓库只保存**代码与文档**。
