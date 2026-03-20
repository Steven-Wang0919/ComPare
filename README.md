# 基于 KAN / MLP / GRNN 的排肥系统论文实验复现工程

## 1. 项目定位

本仓库仅保留论文实验与结果复现能力，包含：

1. 正向建模（开度、转速 -> 排肥量）
2. 反向建模（目标排肥量、开度 -> 转速）
3. 正反向模型统一对比
4. 泛化评估与反向开度留出评估

不再包含交互端/菜单式控制终端相关功能。

---

## 2. 目录结构（当前版本）

```text
.
├── data/
│   └── dataset.xlsx
├── runs/
├── common_utils.py
├── run_utils.py
├── train_kan.py
├── train_mlp.py
├── train_grnn.py
├── inverse_kan.py
├── inverse_mlp.py
├── inverse_grnn.py
├── compare_all.py
├── plot_figures.py
├── evaluate_generalization.py
├── evaluate_inverse_opening_holdout.py
├── select_opening_thresholds_research.py
├── requirements.txt
└── README.md
```

说明：

- 结果统一落盘到 `runs/<timestamp>_<entry>/`。
- 历史目录 `output_data/`、`output_picture/`、`ComPare_Pic/` 若存在可作为旧结果存档，不是当前默认输出目录。

---

## 3. 任务定义与实验口径

### 3.1 正向任务

- 输入：`[开度(mm), 转速(r/min)]`
- 输出：`排肥量(g/min)`

### 3.2 反向任务

- 输入：`[目标排肥量(g/min), 开度(mm)]`
- 输出：`转速(r/min)`

论文主结论采用“策略一致子集”口径：

- 先由目标排肥量映射策略开度
- 再在该策略开度下评估反向模型的转速预测

当前默认策略阈值：

- `< 2800` -> `20 mm`
- `2800 <= x < 4800` -> `35 mm`
- `>= 4800` -> `50 mm`

### 3.3 统一训练流程

所有模型统一遵循：

1. `train` 训练候选模型
2. `val` 选择超参数
3. `train + val` 训练最终模型
4. `test` 评估

归一化统计范围统一使用 `train`。

---

## 4. 核心脚本

### 4.1 训练脚本

- `train_kan.py`：正向 KAN
- `train_mlp.py`：正向 MLP
- `train_grnn.py`：正向 GRNN
- `inverse_kan.py`：反向 KAN
- `inverse_mlp.py`：反向 MLP
- `inverse_grnn.py`：反向 GRNN

### 4.2 对比与绘图

- `compare_all.py`：统一输出正向与反向三模型对比结果
- `plot_figures.py`：基于某次 `compare_all.py` 结果目录绘图

### 4.3 泛化与补充评估

- `evaluate_generalization.py`：多协议泛化评估
- `evaluate_inverse_opening_holdout.py`：反向开度留出评估
- `select_opening_thresholds_research.py`：策略阈值研究脚本

### 4.4 公共模块

- `common_utils.py`：数据读取、划分、指标
- `run_utils.py`：运行目录与 manifest 管理

---

## 5. 推荐运行顺序（实验专用）

### 5.1 安装依赖

```bash
pip install -r requirements.txt
```

### 5.2 运行正反向统一对比

```bash
python compare_all.py
```

输出到：`runs/<timestamp>_compare_all/`

典型文件：

- `forward_model_metrics.csv`
- `forward_model_predictions.csv`
- `inverse_model_metrics.csv`
- `inverse_model_predictions_all.csv`
- `inverse_model_predictions_main.csv`
- `run_manifest.json`

### 5.3 基于对比结果绘图

```bash
python plot_figures.py
```

默认自动查找最新 `runs/*_compare_all`，图片输出到该目录下 `figures/`。

如需显式指定：

```bash
python plot_figures.py --run-dir runs/<timestamp>_compare_all
```

### 5.4 可选扩展评估

```bash
python evaluate_generalization.py
python evaluate_inverse_opening_holdout.py
```

---

## 6. 输出目录约定

### 6.1 统一原则

当前版本默认不向仓库根目录写实验主结果，统一写入：

```text
runs/<timestamp>_<entry>/
```

### 6.2 典型输出示例

- `runs/<timestamp>_train_kan/results_kan.csv`
- `runs/<timestamp>_train_kan/artifacts/kan_forward.pth`
- `runs/<timestamp>_train_kan/artifacts/kan_forward_meta.json`
- `runs/<timestamp>_inverse_kan/inverse_kan_predictions_all.csv`
- `runs/<timestamp>_inverse_kan/inverse_kan_predictions_main.csv`
- `runs/<timestamp>_inverse_kan/artifacts/kan_inverse.pth`
- `runs/<timestamp>_inverse_kan/artifacts/kan_inverse_meta.json`

---

## 7. 对外接口说明

### 7.1 保留入口

- `python compare_all.py`
- `python plot_figures.py`
- `python evaluate_generalization.py`
- `python evaluate_inverse_opening_holdout.py`

### 7.2 变更说明

- 原交互端 CLI 已移除。
- 原交互配置文件接口已移除。

---

## 8. 复现建议

1. 固定随机种子（默认 `seed=42`）
2. 保持 `data/dataset.xlsx` 不变
3. 不混用不同日期 run 的工件与结果
4. 先跑 `compare_all.py` 再跑 `plot_figures.py`
5. 在论文报告中优先使用主口径（策略一致子集）结论

---

## 9. 环境说明

`requirements.txt` 记录作者实验环境版本，可作为复现基线。

若目标机器的 CUDA/CPU 条件不同，建议在保持主依赖兼容前提下调整 PyTorch 安装版本。
