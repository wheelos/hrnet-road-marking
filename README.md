# HRNet Road Marking

Mapillary Vistas v2 上的 **8 类 HRNet-W48 + OCR** 语义分割训练工具。
用于把车道漆线与箭头、填充标记、文字等干扰元素分开，提供训练、断点续训、
评估、原生分辨率推理和 ONNX 导出。运行不依赖 Apollo、Cyber、TensorRT 或原工程目录。

本仓库管理模型前端，不包含 LiDAR 投影、几何融合、车道拓扑或地图导出。
权重、数据集、实车图像和运行产物不入 Git。

## 类别与边界

| ID | 类别 | Vistas v2 原始 ID |
| --- | --- | --- |
| 0 | background | 未归入下面类别的其他 ID |
| 1 | flat_surface | 13–25 |
| 2 | lane_marking | 35、36、55 |
| 3 | exclude_marking | 39–44、46–49、52–54、58 |
| 4 | crosswalk | 45、56 |
| 5 | stop_line | 51 |
| 6 | other_marking | 50、57 |
| 7 | zigzag_marking | 37 |

`exclude_marking` 合并箭头、填充、让行标记、文字和符号，不输出箭头方向。
`other_marking` 与 `zigzag_marking` 独立保留，后端如何处理不由模型定义决定。
没有独立减速带类别；黄色禁停网格可能落入 `other_marking`，不能认为所有禁停区
都已归入 `exclude_marking`。原始 `evaluate=false` 标签及补边区域训练时忽略。
标签 PNG 必须读取 palette 索引，不能通过 RGB 颜色反查类别。

## 安装

验证环境：Python 3.10、PyTorch 2.1.0+cu121、torchvision 0.16.0、RTX 3060 12GB。
详细版本见 `requirements-tested.txt`。训练默认 BF16，要求支持 BF16 的 NVIDIA GPU；
CPU 可以执行测试和 FP32 推理。其他版本组合需要自行验证。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[export,dev]'
```

使用项目自己的 `hrnet_road_marking` 命名空间，不需要 Apex、runx 或把上游代码加入 PYTHONPATH。

## 数据准备

自行按数据集使用条件取得 Mapillary Vistas v2，解压为：

```text
/path/to/vistas/
  config_v2.0.json
  training/images/*.jpg
  training/v2.0/labels/*.png
  validation/images/*.jpg
  validation/v2.0/labels/*.png
```

标准划分为 18,000 张训练、2,000 张验证。实例 mask 是原始 uint16 编码，
语义 ID 为 `instance_id // 256`。命令同时校验实例 mask 与语义 mask 一致性。
可以传入包含所有实例 mask 的目录，或本地数据 ZIP；按图像 ID 对齐，
不依赖 ZIP 内部前缀或镜像的错误嵌套目录。此工具不会下载数据或更改原数据。

```bash
hrnet-prepare --data /path/to/vistas --instances /path/to/vistas.zip --output data/index
```

生成完整箭头实例索引 `arrow_instances.json` 和验证图清单 `validation_ids.json`。
每四个训练样本中三个为普通道路 crop，一个为完整可见箭头 crop；完整指不截断
已标注轮廓，不补全被遮挡部分。`configs/validation_570.json` 保存已用基线清单：
固定随机 200 张与全部含箭头验证图的并集。剩余 1,430 张没有箭头 GT，不能拿来
证明额外的箭头泛化能力。

仅检查图像/标签配对与划分时省略 `--instances`。`--allow-subset` 仅供小样本测试。

## 训练和续训

已经有 8 类权重时，直接严格加载，不再改分类头：

```bash
hrnet-train --config configs/train8.json \
  --data /path/to/vistas --arrow-index data/index/arrow_instances.json \
  --validation-ids configs/validation_570.json \
  --initial-checkpoint /path/to/coarse8.pth --output runs/new_training
```

以上是**新训练**，会新建优化器。`train8.json` 保存原 4,000 步训练配置，
不能把从 8 类 best 再训练一轮称为原始训练的精确复现，也不保证效果提高。

复现原 124→8 初始化路线，把 `--initial-checkpoint` 改为
`--initial-124 /path/to/vistas124.pth`。该文件必须是原架构的已训练 124 类 checkpoint，
不是任意 ImageNet 权重。分类头使用原实验的均值初始化，非分类头严格复制；
初始化不等价于原 124 类概率相加，需要后续训练。已选基线的父权重信息见
`docs/baseline.md`。本仓库不包含原先的 65→124 训练过程。

默认设置：512×768 crop、batch=1、梯度累积4、冻结 BN 统计但训练 affine、
SGD momentum=0.9、weight decay=1e-4、100步 warmup 和 poly 衰减、梯度裁剪10。
损失为主输出加0.4倍辅助输出的加权交叉熵。具体值以 JSON 配置为准。

Ctrl+C / SIGTERM 在完成当前更新和必要验证后保存 `latest.pth`。续训：

```bash
hrnet-train --config configs/train8.json \
  --data /path/to/vistas --arrow-index data/index/arrow_instances.json \
  --validation-ids configs/validation_570.json \
  --resume runs/new_training/latest.pth --output runs/new_training
```

续训必须使用同一目录和配置；恢复模型、优化器、更新步数、随机状态和样本位置。
CUDA/BF16反向计算存在非确定性，不承诺续训与连续训练的参数逐位一致。
`best.pth` 按验证集 lane/exclude 平均 IoU 选择，只有模型权重，不是完整续训文件。
已完成目录拒绝重新训练；新试验使用新目录。`--stop-after N` 可验证安全停止，
不改变原定学习率调度。旧脚本的 latest 文件可作为新训练初值，但不支持当作
本仓库格式的完整续训状态。

快速检查可使用 `--stop-after 2 --limit-validation 2 --skip-final-validation`；
仍按完整 crop/梯度累积运行，仅用于执行正确性，不是训练质量验证。

## 评估

```bash
hrnet-evaluate --data /path/to/vistas --checkpoint /path/to/coarse8.pth \
  --validation-ids configs/validation_570.json --output outputs/eval_panel
hrnet-evaluate --data /path/to/vistas --checkpoint /path/to/coarse8.pth \
  --output outputs/eval_full
```

默认与历史验证一致：图像最长边最多1536、BF16、保留原图比例。
可显式选择 `--precision fp32`，不同精度/尺寸下的结果不能不加说明地混用。
报告包含8×8混淆矩阵、原始124真值×8预测矩阵、各类IoU、箭头/填充过滤召回、
lane误过滤和路面误判标记比例。它们是像素指标，不是实际地图删除率。
`--limit` 会在报告中明确标记，不能当作全量评价。

## 图片推理

```bash
hrnet-predict --checkpoint /path/to/coarse8.pth \
  --input /path/to/images --output outputs/predictions
```

原生尺寸、RGB、FP32。输出原始0–7类别ID PNG及叠加图；`--probabilities`
额外输出8×H×W概率。支持单图或递归目录，保留相对子目录。图像预处理为
`(RGB / 255 - mean) / std`，mean=(.485,.456,.406)，std=(.229,.224,.225)。
模型使用原0.5/1.0 attention融合，输出8类logits，直接argmax；不是124类分组解码。

## ONNX 导出

```bash
hrnet-export --checkpoint /path/to/coarse8.pth --output outputs/hrnet.onnx
```

opset17；输入 `image` 为已归一化 RGB FP32 `[1,3,H,W]`，输出 `logits` 为
FP32 `[1,8,H,W]`。H/W动态，batch固定1。归一化、类别softmax、argmax均在图外，
不要重复执行。保留OCR内部attention，不将其与类别softmax混淆。
模型元数据包含类别顺序、输入输出约定、权重SHA256和训练步数。
`--expected-sha256` 可锁定导出权重。TensorRT engine由应用端按硬件构建，不入Git。

## 已有基线与验证

所选8类权重为4,000步训练中的step1500：
`fbcbc1312837b8e8e31c3d4c9521aab3c5bb44be8c28360a142db29f2c313ffc`。
权重需要另外提供；仓库不会隐式从旧工程找文件或联网下载权重。
A/B精调未获得兼顾过滤召回的整体改进，参数为实验记录，不是推荐替代基线。
详见 `docs/baseline.md` 和 `docs/validation.md`。

```bash
python -m pytest -q
ruff check hrnet_road_marking tests
```

目录：`hrnet_road_marking/`为训练与推理工具，`_vendor/`为保留来源说明的模型代码，
`configs/`为训练参数与验证划分，`tests/`为无需数据和权重的CPU检查。
上游来源与许可证见 `THIRD_PARTY_NOTICES.md`；源快照哈希见 `docs/source_provenance.json`。
