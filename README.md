# Cross-Tactile-Visual-Mapping

Code for **Leveraging CV to haptic processing: Cross tactile-visual mapping based on shared information**, published in *Pattern Recognition*, Volume 174, 2026, Article 113016.

**Authors:** Jin Chen, Ying Fang, Yiwen Xu, Qian Liu, and Tiesong Zhao.

[Paper](https://doi.org/10.1016/j.patcog.2025.113016)

## Overview

This work investigates bidirectional mapping between tactile signals and material images. Two U-Net-based encoder–decoder networks generate visual representations from tactile signals and reconstruct tactile signals from images. Reconstruction objectives and an information-theoretic cross-modal consistency objective connect the two modalities through their latent representations.

The generated visual representations enable the use of computer vision methods for tactile information analysis and provide an intuitive way to visualize tactile information.

本工作研究触觉信号与材料图像之间的双向映射，通过两个基于 U-Net 的编码器–解码器实现触觉到视觉、视觉到触觉的转换，并利用重建损失与跨模态一致性约束学习两种模态的共享信息。

## Files

| File | Description |
| --- | --- |
| `model.py` | Bidirectional U-Net architecture and tactile projection layers. |
| `train.py` | Model training, validation, checkpoint saving, and result export. |
| `datasets.py` | Loading and preprocessing of paired tactile signals and grayscale images. |
| `utils.py` | Loss functions, Adam optimizer, evaluation metrics, logging, and command-line options. |
| `test.py` | Auxiliary classification evaluation script. |
| `CompleterModel.py` | Auxiliary multi-view representation model from COMPLETER. |
| `ChangeFilename.py` | Dataset filename processing utility. |
| `tool.py` | Signal visualization utility. |

## Environment

The training code uses Python, PyTorch, torchvision, NumPy, Pillow, pytorch-msssim, tensorboardX, matplotlib, and tqdm. The auxiliary classification and multi-view scripts also use scikit-learn.

```bash
pip install torch torchvision numpy Pillow pytorch-msssim tensorboardX matplotlib tqdm scikit-learn
```

## Data

The experiments use the LMT-108 material dataset. Training expects paired tactile text files and material images arranged as follows:

```text
data/
  TrainDataFile/
    Source/*.txt
    Target/*.jpg
  TestDataFile/
    Source/*.txt
    Target/*.jpg
```

Each tactile file stores one scalar per line. The model uses 1601-point tactile inputs; shorter signals are padded to this length. Material images are converted to grayscale and resized to 256 × 256. Source and target files are paired in sorted filename order.

## Training

Run from the repository root with CUDA available:

```bash
python train.py --train_dataset ./data/TrainDataFile --test_dataset ./data/TestDataFile --batch-size 2 --num-workers 0 --learning-rate 0.0005
```

Training includes validation and saves checkpoints and generated outputs under `logs/`. To resume a run:

```bash
python train.py --train_dataset ./data/TrainDataFile --test_dataset ./data/TestDataFile --checkpoint ./logs/YOUR_RUN/latest.pth --batch-size 2 --num-workers 0
```

## Citation

```bibtex
@article{chen2026cross_tactile_visual,
  title   = {Leveraging CV to haptic processing: Cross tactile-visual mapping based on shared information},
  author  = {Chen, Jin and Fang, Ying and Xu, Yiwen and Liu, Qian and Zhao, Tiesong},
  journal = {Pattern Recognition},
  volume  = {174},
  pages   = {113016},
  year    = {2026},
  doi     = {10.1016/j.patcog.2025.113016}
}
```
