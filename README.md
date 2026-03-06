# 基于 KAN / MLP / GRNN 的排肥系统建模与智能控制论文工程

## 1. 项目简介

本工程用于完成排肥系统的建模、对比实验与交互控制验证，主要包含以下两类任务：

### 1.1 正向建模任务
输入排肥口开度与排肥轴转速，预测排肥量。

- 输入：`[开度(mm), 转速(r/min)]`
- 输出：`排肥量(g/min)`

### 1.2 反向建模任务
输入目标排肥量与排肥口开度，预测所需排肥轴转速。

- 输入：`[目标排肥量(g/min), 开度(mm)]`
- 输出：`转速(r/min)`

### 1.3 模型类型
本工程包含以下模型：

- KAN
- MLP
- GRNN

### 1.4 项目目标
本工程主要用于：

1. 建立排肥系统正向预测模型  
2. 建立排肥系统反向控制模型  
3. 对比不同模型在任务中的表现  
4. 构建交互式智能控制终端  
5. 为论文实验与结果复现提供工程支撑  

---

## 2. 工程目录说明

建议目录结构如下：

```text
.
├── data/
│   └── dataset.xlsx
├── output_data/
├── ComPare_Pic/
├── path/
│   ├── kan_forward.pth
│   ├── kan_inverse.pth
│   └── model_meta.json
├── common_utils.py
├── train_kan.py
├── train_mlp.py
├── train_grnn.py
├── inverse_kan_V2.py
├── inverse_mlp.py
├── inverse_grnn.py
├── compare_all.py
├── compare_inverse_models.py
├── interactive_app.py
├── generate_requirements.py
├── requirements.txt
└── README.md
````

各文件含义如下：

* `data/dataset.xlsx`：原始实验数据
* `common_utils.py`：公共函数，包括数据读取、划分、误差指标等
* `train_kan.py`：正向 KAN 模型训练脚本
* `train_mlp.py`：正向 MLP 模型训练脚本
* `train_grnn.py`：正向 GRNN 模型训练脚本
* `inverse_kan_V2.py`：最终版反向 KAN 模型
* `inverse_mlp.py`：反向 MLP 模型
* `inverse_grnn.py`：反向 GRNN 模型
* `compare_all.py`：正向模型对比脚本
* `compare_inverse_models.py`：反向模型对比脚本
* `interactive_app.py`：交互式智能控制终端
* `generate_requirements.py`：自动提取依赖并生成 `requirements.txt`
* `requirements.txt`：作者当前真实环境依赖记录

---

## 3. 数据说明

数据文件路径为：

```text
data/dataset.xlsx
```

数据主要包含以下物理量：

* 排肥口开度（mm）
* 排肥轴转速（r/min）
* 排肥量（g/min）

在代码中通常采用以下约定：

### 3.1 正向任务

* 输入：`X = [开度, 转速]`
* 输出：`y = 排肥量`

### 3.2 反向任务

* 输入：`X = [目标排肥量, 开度]`
* 输出：`y = 转速`

---

## 4. 统一实验口径说明

为保证论文实验结果可复现、可比较，本工程统一采用如下口径。

### 4.1 数据划分

所有模型统一使用：

* train
* val
* test

划分方式由 `common_utils.get_train_val_test_indices()` 提供。

### 4.2 归一化

反向模型统一采用：

* 输入与输出均做 0-1 归一化
* 归一化统计范围使用 `train + val`

### 4.3 调参与最终训练

统一采用以下流程：

1. 用 `train` 训练候选模型
2. 用 `val` 选择最优超参数
3. 用 `train + val` 训练最终模型
4. 在 `test` 上进行最终评估

### 4.4 策略一致性评估

反向模型测试时，并非直接对全部测试样本评估，而是只对：

> 实际开度与策略推荐开度一致

的样本进行评估。

当前开度策略为：

* 目标排肥量 `< 2800` → 开度 `20 mm`
* `2800 <= 目标排肥量 < 4800` → 开度 `35 mm`
* `>= 4800` → 开度 `50 mm`

---

## 5. 各脚本功能说明

## 5.1 公共模块

### `common_utils.py`

提供以下公共能力：

* 数据读取
* train/val/test 划分
* 平均相对误差（MRS）计算

---

## 5.2 正向模型脚本

### `train_kan.py`

训练正向 KAN 模型。

* 输入：`[开度, 转速]`
* 输出：`排肥量`

### `train_mlp.py`

训练正向 MLP 模型。

### `train_grnn.py`

训练正向 GRNN 模型。

### `compare_all.py`

对比正向模型性能，并输出相应图表与结果。

---

## 5.3 反向模型脚本

### `inverse_kan_V2.py`

最终版反向 KAN 模型。

说明：

* 当前工程中，反向 KAN 以本文件为准
* 已统一为论文最终使用版本
* `interactive_app.py` 中的反向模型也基于该文件定义

### `inverse_mlp.py`

训练并评估反向 MLP 模型。

### `inverse_grnn.py`

训练并评估反向 GRNN 模型。

### `compare_inverse_models.py`

对比三种反向模型：

* KAN
* MLP
* GRNN

并输出：

* 指标柱状图
* 真实值与预测值散点图

---

## 5.4 交互系统脚本

### `interactive_app.py`

交互式智能控制终端，支持以下两类功能：

#### 1）正向预测

输入开度与转速，预测排肥量。

#### 2）智能控制

输入目标排肥量，系统自动推荐：

* 最优开度
* 对应转速

说明：

* 启动时优先加载 `path/` 中的已保存模型
* 若模型权重与当前模型结构不兼容，则自动重新训练并覆盖保存
* 当前交互系统默认使用与论文实验尽量一致的模型版本

---

## 6. 推荐运行顺序

建议按以下顺序运行工程。

### 第一步：确认数据文件存在

确保以下文件存在：

* `data/dataset.xlsx`
* `common_utils.py`

### 第二步：运行反向模型对比

```bash
python compare_inverse_models.py
```

用于生成论文中反向模型的主要实验结果。

### 第三步：运行正向模型对比

```bash
python compare_all.py
```

用于生成正向模型的主要实验结果。

### 第四步：启动交互系统

```bash
python interactive_app.py
```

用于进行功能展示与控制验证。

---

## 7. 交互系统与论文实验的关系

本工程中：

* **论文实验结果** 以各训练脚本和对比脚本输出结果为准
* **交互系统** 用于展示模型在智能控制任务中的应用方式

说明：

* 若 `path/` 中存在旧版权重文件，而当前模型结构已更新，则程序可能提示加载失败后自动重新训练
* 这是正常现象，不代表程序运行错误
* 建议在最终整理工程时清理旧版 `path/` 目录，避免模型版本混淆

---

## 8. 模型版本说明

当前工程建议采用如下“唯一权威版本”：

* 正向 KAN：`train_kan.py`
* 反向 KAN：`inverse_kan_V2.py`
* 反向 MLP：`inverse_mlp.py`
* 反向 GRNN：`inverse_grnn.py`

若目录中仍保留如下旧文件：

* `inverse_kan.py`
* `inverse_kan_V1.py`

建议：

* 不再用于论文最终实验
* 可归档或删除
* 避免与最终版混用

---

## 9. 输出结果说明

### 9.1 图像输出

反向模型对比图通常输出至：

```text
ComPare_Pic/
```

例如：

* `metrics_bar.png`
* `scatter_true_vs_pred.png`

### 9.2 模型权重输出

交互系统模型通常保存在：

```text
path/
```

包括：

* `kan_forward.pth`
* `kan_inverse.pth`
* `model_meta.json`

### 9.3 结果表输出

若训练脚本启用了结果导出，则通常位于：

```text
output_data/
```

---

## 10. 环境说明

本项目的 `requirements.txt` 记录的是作者实际运行本工程时所使用的真实 Python 环境版本。

当前依赖由 `generate_requirements.py` 从作者实际环境自动提取生成。

其中 PyTorch 版本为：

```text
torch==2.10.0.dev20251124+cu128
```

该版本为作者本机环境中的实际安装版本，用于保证工程结果与作者运行环境一致。

说明：

* 本文件优先保证“真实可追溯”
* `requirements.txt` 反映的是作者真实运行环境
* 若在其他设备上复现，请根据本机 CUDA / CPU 环境安装兼容版本的 PyTorch
* 若目标环境与作者环境不同，可能需要对 PyTorch 进行适配安装

项目常用依赖包括：

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `openpyxl`
* `torch`
* `seaborn`

安装依赖可使用：

```bash
pip install -r requirements.txt
```

若使用 GPU，请确保：

* 当前系统已安装可用 CUDA 环境
* PyTorch 与 CUDA 版本兼容

---

## 11. 依赖文件生成说明


* `requirements.txt`

该文件优先记录真实环境中的依赖版本。

---

## 12. 复现建议

为了提高论文结果复现性，建议：

1. 固定随机种子
2. 保持数据文件不变
3. 不混用旧版模型文件
4. 在最终运行前清理旧的 `path/` 权重目录
5. 按统一顺序重新生成结果图与结果表

推荐复现流程：

```bash
python compare_inverse_models.py
python compare_all.py
python interactive_app.py
```

---

## 13. 注意事项

1. 本工程当前重点保证的是：

   * 模型实验逻辑统一
   * 数据处理口径一致
   * 交互系统与论文模型版本尽量一致

2. 若修改网络结构或超参数：

   * 应同步更新交互系统
   * 应重新训练模型权重
   * 应重新生成结果图表

3. 若发现 `interactive_app.py` 启动时报“旧权重不兼容”：

   * 通常是旧版权重与当前模型结构不一致
   * 程序自动重训即可
   * 这不是程序错误

4. `requirements.txt` 中记录的是作者真实环境，尤其是 PyTorch 版本可能依赖具体 CUDA 环境；
   因此在其他机器上复现时，可能需要做适配安装。

---

