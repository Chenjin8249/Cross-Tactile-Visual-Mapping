"""Bidirectional reconstruction objective and CCL estimator."""
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim


def latent_probabilities(features):
    """Map signed latent logits to probabilities before estimating a joint.

    The supplied U-Net latent is [B,512,1,1], so flattening and channel
    categories both yield [B,512]. Softmax is a numerical/modeling correction
    relative to the supplied unnormalized signed-latent estimator.
    """
    if features.ndim == 4:
        logits = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    elif features.ndim == 2:
        logits = features
    else:
        raise ValueError("CCL features must have shape [B,C,H,W] or [N,C]")
    if logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("CCL requires observations and at least two channels")
    return torch.softmax(logits.float(), dim=1)


def compute_joint(view1, view2):
    if view1.shape != view2.shape:
        raise ValueError("CCL needs equal latent shapes and aligned samples")
    p, q = latent_probabilities(view1), latent_probabilities(view2)
    joint = p.transpose(0, 1).matmul(q)
    joint = (joint + joint.transpose(0, 1)) * 0.5
    # Smooth and renormalize: all log arguments are positive probabilities.
    joint = joint.clamp_min(torch.finfo(joint.dtype).eps)
    return joint / joint.sum()


def crossview_contrastive_Loss(view1, view2, lamb=9, EPS=None):
    """Retain the supplied CCL algebra and lambda=9, with valid probabilities.

    The original k² scaling is retained; this is not a claim of exact
    reproduction of the paper's unspecified probability estimator.
    """
    joint = compute_joint(view1, view2)
    k = joint.shape[0]
    p_i = joint.sum(dim=1, keepdim=True)
    p_j = joint.sum(dim=0, keepdim=True)
    return -(joint * (joint.log() - (lamb + 1) * p_i.log()
                     - (lamb + 1) * p_j.log())).sum() / (k * k)


class Loss(nn.Module):
    """L = 0.2*L_ccl + 0.4*MSE(tactile) + 0.4*MSE(image); original relative weights 1:2:2."""

    def __init__(self, metrics='ccl'):
        super().__init__()
        if metrics != 'ccl':
            raise ValueError("This revision uses the combined bidirectional objective only")
        self.mse = nn.MSELoss()

    def forward(self, SourceLatent, TargetLatent, SourcePredict, TargetPredict,
                source, target):
        if SourcePredict.shape != source.shape:
            raise ValueError(f"Tactile prediction/GT mismatch: {SourcePredict.shape} vs {source.shape}")
        if TargetPredict.shape != target.shape:
            raise ValueError(f"Visual prediction/GT mismatch: {TargetPredict.shape} vs {target.shape}")
        tactile_mse = self.mse(SourcePredict, source)
        visual_mse = self.mse(TargetPredict, target)
        ccl = crossview_contrastive_Loss(SourceLatent, TargetLatent)
        # MS-SSIM is a reporting metric, not an additional training loss.
        with torch.no_grad():
            ms_ssim_loss = 1 - ms_ssim(TargetPredict.clamp(0, 1), target, data_range=1.0)
        return {
            "mse_loss_S": tactile_mse,  # S = Source (tactile), NOT paper T/V notation
            "mse_loss_T": visual_mse,   # T = Target (visual)
            "mse_loss": visual_mse,     # backwards-compatible metric name
            "ms_ssim_loss": ms_ssim_loss,
            "crossview_contrastive_Loss": ccl,
            "loss": 0.2 * ccl + 0.4 * tactile_mse + 0.4 * visual_mse,
        }


