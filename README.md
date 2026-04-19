# Protagonist Detection in Movie Clips

Category-specific (protagonist) appearance period prediction based on multi-modal features, combined with cross-shot tracking (Re-ID).

Data source: YT8M database (Tag 1014 = Movie Clips), TFRecord format, RGB 1024D + Audio 128D, 1 FPS.

Data and model address: [Miraaa030/ProtagonistDetect_Data_v1](https://huggingface.co/datasets/Miraaa030/ProtagonistDetect_Data_v1)

# Add fusion module

If anyone wants to add the fusion module: change multimodal_fusion.py
The output dict of ProtagonistDetector.forward() keeps the following keys unchanged:
rgb [B,T,1024];  embeds [B,10,128];  audio [B,T,128];  final_score [B,10];  mask[B,T];  seg_mask [B,10];  length[B]

---
  
## Method overview

```
input: RGB [B,T,1024] + Audio [B,T,128]
Step1: CrossModalAdapter      Q=RGB, K/V=Audio, audio guides visual attention
Step2: AdaptiveHierarchicalPool   Adaptive 10-segment segmentation, masked mean pooling
Step3: SegmentTransformer         Cross-segment timing reasoning（Pre-LN Transformer）
Step4: IdentityHead               CEA dimensionality reduction → L2 normalization embedding [B,10,128]
Step5: K-Means clustering → most frequent cluster = protagonist → time interval JSON
```

Loss function (unsupervised, no manual labeling required):

- **L1** AudioVisualAlignmentLoss — Sound and picture cosine alignment (Papalampidi et al. 2023)
- **L2** TemporalConsistencyLoss — Adjacent segment smoothing constraints (Zheng et al. 2015)
- **L3** InfoNCEPrototypeLoss — Comparative learning within videos to prevent prototype degradation
---

## File structure

```
ProtagonistDetect/
├── main.py                          # Training/inference
├── requirements.txt
├── src/
│   ├── data_utils/
│   │   └── loader.py                # YT8MLoader, SequenceExample parse
│   ├── models/
│   │   ├── multimodal_fusion.py     # Core model (4 submodules + ProtagonistDetector)
│   │   └── loss_functions.py        # Three unsupervised losses + ProtagonistLoss
│   └── inference.py                 # K-Means clustering inference, output JSON
├── evaluate.py                      # calculate F1 / mAP / MOTA
├── ablation.py                      # 5 groups of ablation experiments(to be completed)

```

---

## Order

```bash
pip install -r requirements.txt
```

**train**
```bash
python main.py --mode train --train_tfrecord data/train_1014.tfrecord --val_tfrecord   data/val_1014.tfrecord --epochs 30
```

**inference**
```bash
python main.py --mode inference --val_tfrecord data/test_1014.tfrecord --checkpoint outputs/best_model.pt --output outputs/test_predictions.json
```

**evaluate**
```bash
python evaluate.py --predictions outputs/test_predictions.json --annotations outputs/annotation_task.xlsx --output outputs/evaluation_report.xlsx
```

**ablation experiment**
```bash
python ablation.py --tfrecord data/test_1014.tfrecord --checkpoint  outputs/best_model.pt --annotations outputs/annotation_task.xlsx
```

---

## Evaluation indicators

| index | illustration | References |
|------|------|----------|
| Precision / Recall / F1 | Segment level classification accuracy | Papalampidi et al. (2023) |
| mAP | final_score sort quality | Xiao et al. (2017) |
| MOTA / MOTP | Track consistency across segments | Bernardin & Stiefelhagen (2008) |

---

## 主要参考文献

- Xiao et al. (2017). *End-to-end Deep Learning for Person Search*
- Fu & Huang (2008). *Human Age Estimation with Regression on Discriminative Aging Manifold*
- Zheng et al. (2015). *Scalable Person Re-identification: A Benchmark*

