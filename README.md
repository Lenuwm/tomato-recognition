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
