# 模型权重

这三份权重保留原始文件内容，通过Git LFS版本管理，总计约1.16GB。
`manifest.json`记录各文件的字节数、SHA256、类别数、训练步数和用途；
`SHA256SUMS`可在仓库根目录直接校验：

```bash
git lfs pull
sha256sum -c models/SHA256SUMS
```

| 文件 | 输出类别 | 用途 |
| --- | ---: | --- |
| `hrnet_coarse8_best_step1500.pth` | 8 | 当前最佳模型，推理、ONNX导出及后续微调 |
| `hrnet_vistas124_best_step10000.pth` | 124 | 8类模型训练的直接父权重，用于124→8初始化 |
| `hrnet_mapillary65_pretrained.pth` | 65 | 最早采用的原始HRNet-OCR预训练模型，保留训练来源 |
| `mapillary65_labels.json` | 类别表 | 原始65类模型配套配置，按原文件保留 |

训练来源为：原始65类预训练权重 → 124类训练step10000 → 8类训练step1500。
此仓库目前实现8类训练及124→8初始化；没有实现65→124的完整训练流程。
原始65类文件原名为`mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth`，只改存放名称，
不裁剪、重存或转换权重。类别表有66个条目，最后的unlabeled条目不对应模型输出通道；
模型实际输出为65通道。基础模型不等于8类模型，也不是仅有backbone的ImageNet权重。

## 最佳8类模型

```bash
hrnet-predict --checkpoint models/hrnet_coarse8_best_step1500.pth \
  --input /path/to/images --output outputs/predictions

hrnet-export --checkpoint models/hrnet_coarse8_best_step1500.pth \
  --expected-sha256 fbcbc1312837b8e8e31c3d4c9521aab3c5bb44be8c28360a142db29f2c313ffc \
  --output outputs/hrnet.onnx
```

该文件是已完成训练的8类checkpoint，直接严格加载；不要再次转换分类头。
它不包含完整优化器状态；新训练可使用`--initial-checkpoint`，完整续训则使用
对应新训练目录下的`latest.pth`。

## 124类父权重

```bash
hrnet-train --config configs/train8.json \
  --data /path/to/vistas --arrow-index data/index/arrow_instances.json \
  --validation-ids configs/validation_570.json \
  --initial-124 models/hrnet_vistas124_best_step10000.pth \
  --output runs/from_vistas124
```

## 原始65类模型

8类CLI会拒绝直接加载65类权重。研究原始模型时可显式构建65类架构：

```python
import torch
from hrnet_road_marking.model import build

payload = torch.load("models/hrnet_mapillary65_pretrained.pth", map_location="cpu")
weights = payload.get("state_dict", payload)
weights = {name.removeprefix("module."): value for name, value in weights.items()}
model = build(classes=65).eval()
model.load_state_dict(weights, strict=True)
```

该模型输出的类别ID应使用`mapillary65_labels.json`解释，不能套用本仓库8类类别表。
三个模型的来源和使用条件独立于本仓库代码许可证；此处保留权重不改变上游条款。
模型文件不随Python wheel/sdist打包，也不包含硬件相关的TensorRT engine。
