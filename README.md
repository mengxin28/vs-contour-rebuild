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
# 阶段一：预处理（读取 -> 0.5m降采样 -> 去噪 -> 最大聚合体 -> 去地面/天花板 -> 投影 + 密度图）
python code/preprocess.py <点云文件...>
python code/preprocess.py 粟塘B1.las 雅德B1.las
python code/preprocess.py CLEAN_UNDER_GROUND.ply RESULT_B1.ply

# 阶段二：贴合外墙轮廓线提取（读阶段一的 *_wall.ply）
python code/contour.py 输出/CLEAN_UNDER_GROUND_wall.ply 输出/RESULT_B1_wall.ply

# 阶段二（可视化）：外圈高密度点云高亮（红=外墙环，用原始点竖直密度+外圈位置）
python code/outer.py CLEAN_UNDER_GROUND.ply RESULT_B1.ply
```

默认参数：降采样 `0.5m`、密度栅格 `0.3m`（见 `code/preprocess.py` 顶部常量
`VOXEL`、`MIN_NEIGHBORS`、`NORM_RADIUS`、`HORIZ_TOL`、`DENSITY_GRID`）。
贴合轮廓线用 `RASTER_GRID`、`CLOSE_ITER`、`OPEN_KERNEL`、`SIMPLIFY_TOL`；
规则化（转正+Douglas-Peucker）用 `REG_TOL`（见 `code/contour.py` 顶部）。

## 输出（写入 `输出/`）
阶段一（`preprocess.py`），每个源文件 `NAME` 生成：
- `NAME_wall.ply` —— 投影后的外墙点云（Z 归零）
- `NAME_俯视_外墙.png` —— 外墙俯视轮廓
- `NAME_俯视_原始.png` —— 降采样后整车点云（按 Z 着色）
- `NAME_透视_外墙.png` —— 外墙 3D（按高度着色）
- `NAME_热力密度图.png` —— 0.3m 栅格密度热力图（外墙墙带高亮）
- `汇总.json` —— 各阶段点数、bbox、密度峰值

阶段二（`contour.py`），每个 `NAME_wall.ply` 生成：
- `NAME_轮廓线.png` —— **纯轮廓线图**（白底红粗线+顶点，最直观）
- `NAME_外轮廓.png` —— 墙点散点 + 规则化外轮廓(红粗线) + 贴合外沿原始线(灰)
- `NAME_外轮廓.json` —— 轮廓线顶点坐标、面积、周长（含栅格/容差参数）
- `外轮廓_汇总.json`

外圈高亮（`outer.py`），每个原始 `NAME.ply` 生成：
- `NAME_外圈点云.png` —— 红=外圈墙点，橙=双重高密度但不在外圈(内部柱)，紫=墙外噪声(已剔除)，浅蓝=其余
- 双密度门槛：**全局前10%**(`GLOBAL_PCT=90`) 且 **局域前10%**(`LOCAL_PCT=90`/`LOCAL_CELL=12`)，再 AND 距外边界≤`BAND`
- 连通域过滤：只保留最大连通块(`KEEP_ONLY_BIGGEST`)，剔除脱离主体的墙外噪声
- 密度=竖直堆叠密度(0.3m XY柱内点数)；参数 `COL_GRID=0.3`、`GRID=0.5`、`BAND=1.0`

## 版本历史
每版的功能、用法与用户评价见 `版本记录.md`。

> 注：`.git/info/exclude` 已本地忽略 `输出/`、`.las/.laz`、根目录 `.ply` 等大体积数据，
> 本仓库只保存**代码与文档**。
