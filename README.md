

# Tomato Recognition

基于 PyTorch 的番茄识别项目（训练 + 推理 + 数据集处理脚本），包含自定义 `Dataset`、模型结构、损失函数（含类别加权）、以及 COCO → YOLO 的数据转换与数据划分修复工具。

## Features
- 训练：`train.py`
- 推理：`infer.py`
- 模型定义：`model.py`
- 数据读取：`dataset.py`
- 损失函数：`loss.py`（包含根据训练标签统计的 **class-weighted loss**）
- 数据处理工具：
  - `coco2yolo.py`：COCO 标注格式转换为 YOLO 格式
  - `fix_splits.py`：修复/重建数据集划分（train/val/test）
  - `make_dummy_data.py`：生成/构造用于快速跑通流程的 dummy 数据
- 训练日志与输出目录预留：`logs/`、`outputs/`（以及历史目录 `outputs_old/`）

## Repository Structure
```

.
├── data/                # 数据集与标注（建议放在此目录）
├── logs/                # 训练日志（如 loss、metric、tensorboard 等）
├── outputs/             # 推理/训练产出（可视化结果、预测文件等）
├── outputs_old/         # 历史输出备份
├── coco2yolo.py         # COCO -> YOLO 转换脚本
├── dataset.py           # 数据集读取与预处理
├── fix_splits.py        # 修复数据集划分
├── infer.py             # 推理脚本
├── loss.py              # 损失函数（含类别加权）
├── make_dummy_data.py   # 生成 dummy 数据，便于端到端验证
├── model.py             # 模型结构
├── train.py             # 训练入口
└── utils.py             # 通用工具函数

````

## Requirements
- Python 3.8+（建议）
- 依赖项以你的 `pip/conda` 环境为准

> 推荐你补充一个 `requirements.txt`（或在 README 中写清关键依赖），例如：
> - torch / torchvision
> - numpy
> - opencv-python
> - tqdm
> - pycocotools（如果使用 COCO 标注）

## Quick Start

### 1) 克隆与进入目录
```bash
git clone https://github.com/Lenuwm/tomato-recognition.git
cd tomato-recognition
````

### 2) 安装依赖（建议你在仓库补充 requirements.txt 后使用）

```bash
pip install -r requirements.txt
```

### 3) 数据准备（推荐流程）

1. 将原始数据与标注放入 `data/`
2. 若你的数据是 COCO 标注（`instances_*.json`），可先用：

```bash
python coco2yolo.py
```

3. 若存在 train/val/test 划分问题或需要自动划分，可用：

```bash
python fix_splits.py
```

> 具体输入/输出路径、类别映射、标注格式等细节请以脚本内注释与参数为准（见下一节）。

### 4) 训练

```bash
python train.py
```

### 5) 推理

```bash
python infer.py
```

## Script Usage Notes

由于不同实现会在脚本中定义不同的参数与默认路径，推荐你用以下方式查看准确用法：

* 查看帮助（若脚本使用 argparse）：

```bash
python train.py -h
python infer.py -h
python coco2yolo.py -h
python fix_splits.py -h
python make_dummy_data.py -h
```

* 若脚本未提供 `-h`，请直接打开对应 `.py` 文件，检查：

  * 数据根目录（`data_dir` / `root` / `img_dir` / `ann_path`）
  * 类别列表与 id 映射（`classes` / `class_names`）
  * 训练超参（epoch、batch size、lr、img size）
  * 输出位置（`logs/`、`outputs/`、`checkpoints/`等）

## Outputs

默认建议：

* 训练日志与曲线：`logs/`
* 推理可视化/预测结果：`outputs/`
* 旧版输出备份：`outputs_old/`

你也可以在 README 中补充：

* 最佳模型权重保存路径（如 `outputs/best.pt` 或 `logs/ckpt_best.pth`）
* 推理结果示例图（建议新增 `assets/` 并在此处展示）


```
