# Protagonist Detection in Movie Clips

基于多模态特征的特定类别（主人公）出现时段预测，结合跨镜头跟踪（Re-ID）。

数据来源：YT8M 数据库（标签 1014 = Movie Clips），TFRecord 格式，RGB 1024D + Audio 128D，1 FPS。

---

## 方法概述

```
RGB [B,T,1024] + Audio [B,T,128]
        ↓
CrossModalAdapter      Q=RGB, K/V=Audio，音频引导视觉注意力
        ↓
AdaptiveHierarchicalPool   自适应10段切分，masked mean pooling
        ↓
SegmentTransformer         跨段时序推理（Pre-LN Transformer）
        ↓
IdentityHead               CEA降维 → L2归一化 embedding [B,10,128]
        ↓
K-Means 聚类 → 出现最多的 cluster = 主角 → 时间区间 JSON
```

损失函数（无监督，无需人工标注）：

- **L1** AudioVisualAlignmentLoss — 音画余弦对齐（Papalampidi et al. 2023）
- **L2** TemporalConsistencyLoss — 相邻段平滑约束（Zheng et al. 2015）
- **L3** InfoNCEPrototypeLoss — 视频内对比学习，防止 prototype 退化（Oord et al. 2018）

---

## 文件结构

```
Project_Root/
├── main.py                          # 训练 / 推理入口
├── requirements.txt
├── src/
│   ├── data_utils/
│   │   └── loader.py                # YT8MLoader，SequenceExample 解析
│   ├── models/
│   │   ├── multimodal_fusion.py     # 核心模型（4个子模块 + ProtagonistDetector）
│   │   └── loss_functions.py        # 三项无监督损失 + ProtagonistLoss
│   └── inference.py                 # K-Means 聚类推理，输出 JSON
├── generate_annotation_excel.py     # 生成队友标注表
├── evaluate.py                      # 计算 F1 / mAP / MOTA
├── ablation.py                      # 5组消融实验
└── visualization/
    ├── download_mp4.sh              # yt-dlp 下载视频
    └── overlay_results.py           # 预测结果叠加到视频帧
```

---

## 快速开始

```bash
pip install -r requirements.txt
```

**训练**
```bash
python main.py --mode train \
    --train_tfrecord data/train_1014.tfrecord \
    --val_tfrecord   data/val_1014.tfrecord \
    --epochs 30
```

**推理**
```bash
python main.py --mode inference \
    --val_tfrecord data/test_1014.tfrecord \
    --checkpoint   outputs/best_model.pt \
    --output       outputs/test_predictions.json
```

**评估**（队友填写完标注表后）
```bash
python evaluate.py \
    --predictions outputs/test_predictions.json \
    --annotations outputs/annotation_task.xlsx \
    --output      outputs/evaluation_report.xlsx
```

**消融实验**
```bash
python ablation.py \
    --tfrecord    data/test_1014.tfrecord \
    --checkpoint  outputs/best_model.pt \
    --annotations outputs/annotation_task.xlsx
```

---

## 评估指标

| 指标 | 说明 | 参考文献 |
|------|------|----------|
| Precision / Recall / F1 | Segment 级分类精度 | Papalampidi et al. (2023) |
| mAP | final_score 排序质量 | Xiao et al. (2017) |
| MOTA / MOTP | 跨段跟踪一致性 | Bernardin & Stiefelhagen (2008) |

---

## 主要参考文献

- Papalampidi & Lapata (2023). *Hierarchical3D Adapters for Long Video-to-text Summarization*
- Xiao et al. (2017). *End-to-end Deep Learning for Person Search*
- Fu & Huang (2008). *Human Age Estimation with Regression on Discriminative Aging Manifold*
- Zheng et al. (2015). *Scalable Person Re-identification: A Benchmark*
- Bernardin & Stiefelhagen (2008). *Evaluating Multiple Object Trackers and Detectors*
- Oord et al. (2018). *Representation Learning with Contrastive Predictive Coding*
