# Cross tactile–visual mapping based on shared information

Code prepared from the author's uploaded `最终版本1` archive for:

**Leveraging CV to haptic processing: Cross tactile-visual mapping based on shared information**  
Jin Chen, Ying Fang, Yiwen Xu, Qian Liu, Tiesong Zhao.  
*Pattern Recognition*, 174 (2026), 113016. DOI: [10.1016/j.patcog.2025.113016](https://doi.org/10.1016/j.patcog.2025.113016).

## Status / 状态

这是基于作者本次上传的双向 U-Net 代码整理的修正版，保留原始网络层、1601 点输入及输出。它与此前的 48000 点残差网络包不同，不使用此前临时新增的低秩触觉输出层。已完成 Python 语法、静态接口和 NumPy 数据读取检查。**尚未在本环境运行 PyTorch 前向/反向传播或真实数据训练；未复现论文表格。** 环境缺少 PyTorch，此前安装尝试下载超时。

The supplied architecture and bidirectional reconstruction losses broadly match the paper. The source has been corrected and documented; this is not a claim that the published metrics have been independently reproduced. Four runtime self-tests are provided but have not been executed in the editing environment.

## Architecture and objective / 模型与损失

| Path | Input | Output |
|---|---|---|
| Tactile → visual | `[B,1601]` → FCL → `[B,1,256,256]` → U-Net A | grayscale `[B,1,256,256]` |
| Visual → tactile | grayscale `[B,1,256,256]` → U-Net B → FCL | `[B,1601]` |
| Consistency | two bottlenecks `[B,512,1,1]` | scalar CCL |

U-Nets use eight downsampling stages, seven upsampling stages with skip connections, instance normalization, dropout and Tanh output layers. The last tactile FCL is linear.

```python
mse_visual = mse(TargetPredict, target)
mse_tactile = mse(SourcePredict, source)
loss = 0.2 * ccl + 0.4 * mse_visual + 0.4 * mse_tactile
```

保留本次上传实现的 0.2/0.4/0.4 权重。它是论文 `CCL + 2*MSE_tactile + 2*MSE_visual` 的整体 0.2 倍，**相对权重一致**，不应误判为遗漏损失；整体缩放仍会改变梯度大小，与梯度裁剪及 Adam epsilon 等细节有关。`mse_loss_T` 中 T 代表 Target（图像），S 代表 Source（触觉），不是论文 tactile/visual 上标。

Adam: learning rate `5e-4`, betas `(0.9, 0.999)`, matching the paper's stated optimizer settings. MS-SSIM is logged, not added as another training loss.

## Corrections / 修改记录

- Fixed the visual-to-tactile decoder's final call from `A_final(vz7)` to `B_final(vz7)`. Both final layers already existed; no new decoder layers were invented. This changes the original shared-final-layer behavior and makes the B output layer trainable.
- Added softmax over the 512 bottleneck channels before estimating joint probabilities. The original directly used signed latent values as probabilities. Joint probabilities are smoothed and renormalized to avoid invalid logs and zero denominators. The original CCL algebra, lambda=9 and `/k²` normalization are retained. **This probability mapping is an explicit implementation correction; its training impact must be evaluated.**
- Changed default learning rate from `1e-3` to `5e-4`. Default batch size is reduced from 16 to 2 for memory; configurable through CLI. Epoch default remains 800.
- Replaced deprecated `np.float` with float32. Mean-padding handles odd deficits exactly, rejects empty/non-finite values and rejects signals longer than 1601 rather than silently cropping.
- Fixed CUDA-to-NumPy export, `best_psnr` updates, next-epoch resume, full scheduler restoration, scalar logging and multi-image validation. Added CPU/device selection. Historical checkpoints are rejected for direct resume because final-layer use and CCL probability handling changed.
- Replaced unrelated classifier `test.py` with a paired mapping evaluation entry point.
- Left out unused/incomplete COMPLETER reference code, filename-renaming scripts, IDE files and `tool.py`'s embedded signal list. These were not used by the training entry point. Original attachments remain the provenance source.

## Install / 安装

Install compatible PyTorch and torchvision builds for your CPU/CUDA environment, then:

```bash
python -m pip install -r requirements.txt
python selftest.py
```

Requirements specify version ranges, not a tested lockfile. Self-tests check the weighted objective, numerical loss gradients, valid probabilities, Adam, and the complete U-Net's meta-device shape/gradient graph. Meta tests do not measure real model gradient values or GPU memory. See `VALIDATION.json` for executed vs unexecuted checks.

## Data / 数据

```text
data/TrainDataFile/Source/*.txt
data/TrainDataFile/Target/*.jpg
data/TestDataFile/Source/*.txt
data/TestDataFile/Target/*.jpg
```

Each tactile TXT contains one scalar per line, no more than 1601 samples. Shorter signals are padded to 1601 using the original mean-padding policy. Images are converted to grayscale and resized to 256×256, with values in [0,1]. The original Tanh image output is preserved (range [-1,1]); supervision remains against [0,1] images, as in the supplied implementation. This is valid for MSE, but does not imply a symmetric image normalization pipeline.

**重要的数据限制**：代码直接读入最多 1601 点信号，未包含把其他长度原始振动转换成 1601 点的预处理。不能把 48000 点原始信号直接传给它。需要作者提供最终实验的实际数据预处理，不能凭此代码假定采样率、裁切或重采样方式。

Source/Target files are sorted independently and paired by index, preserving the supplied loader. Verify the actual pair mapping and fixed train/test split before training. Equal file counts do not prove correct pairing. No dataset, preprocessing recovery, train/test overlap audit or pretrained checkpoint is included.

## Train and evaluate / 运行

Run commands in the directory containing `train.py`:

```bash
python train.py --train_dataset ./data/TrainDataFile --test_dataset ./data/TestDataFile --device cuda --batch-size 2 --num-workers 0 --learning-rate 0.0005
python test.py --test_dataset ./data/TestDataFile --checkpoint ./logs/YOUR_RUN/best.pth --device cuda
```

Replace paths with real locations. `--device auto` chooses CUDA when available; otherwise CPU. The two dense tactile projection layers and U-Nets are large; adjust batch size to actual memory. No GPU-memory benchmark has been performed.

```bash
python train.py --train_dataset ./data/TrainDataFile --test_dataset ./data/TestDataFile --checkpoint ./logs/YOUR_RUN/latest.pth --device cuda --num-workers 0
```

Only checkpoints created by this revision (`unet_bidirectional_v2`) are directly resumable. Optimizer, scheduler, next epoch, global step and best PSNR are restored; RNG states are not restored, so resume is not bitwise reproducible. Inference from historical checkpoints requires a separate compatibility decision; do not silently load them into the corrected architecture behavior.

Training PSNR is derived from the raw visual MSE (floor 1e-12), while training MS-SSIM clips predictions to [0,1]. Validation retains the supplied clipping and uint8 quantization convention, evaluated per image. Training and validation PSNR therefore use different preprocessing. Every 20 epochs, preview images, tactile predictions and diagnostic latents are exported.

## Citation

```bibtex
@article{chen2026cross_tactile_visual,
  title = {Leveraging CV to haptic processing: Cross tactile-visual mapping based on shared information},
  author = {Chen, Jin and Fang, Ying and Xu, Yiwen and Liu, Qian and Zhao, Tiesong},
  journal = {Pattern Recognition},
  volume = {174},
  pages = {113016},
  year = {2026},
  doi = {10.1016/j.patcog.2025.113016}
}
```

No new open-source license has been assigned. The author should choose a license and verify third-party provenance before granting reuse rights. The publisher PDF is not redistributed in this repository source.
