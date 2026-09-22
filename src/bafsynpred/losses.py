"""Evidential training objectives for BAF-SynPred."""

import torch
import torch.nn.functional as F


def adaptive_edl_loss(
    evidence,
    target,
    epoch,
    num_classes=2,
    annealing_step=30,
    class_weights=None,
    adaptive_kl_beta=1.0,
):
    """Digamma Bayes risk with annealed, sample-adaptive KL regularization."""
    y = F.one_hot(target.long(), num_classes=num_classes).float()
    alpha = evidence + 1.0
    strength = torch.sum(alpha, dim=1, keepdim=True)
    loss_error = torch.sum(y * (torch.digamma(strength) - torch.digamma(alpha)), dim=1)

    alpha_tilde = y + (1 - y) * alpha
    strength_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
    kl_term = (
        torch.lgamma(strength_tilde)
        - torch.lgamma(torch.tensor(num_classes, dtype=torch.float32, device=evidence.device))
        - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        + torch.sum(
            (alpha_tilde - 1)
            * (torch.digamma(alpha_tilde) - torch.digamma(strength_tilde)),
            dim=1,
            keepdim=True,
        )
    ).squeeze(-1)

    annealing = min(1.0, epoch / annealing_step)
    if adaptive_kl_beta > 0:
        correct_probability = torch.sum(y * (alpha / strength), dim=1).detach()
        kl_weight = 1.0 + adaptive_kl_beta * (1.0 - correct_probability)
        loss = loss_error + annealing * kl_weight * kl_term
    else:
        loss = loss_error + annealing * kl_term

    if class_weights is not None:
        sample_weights = class_weights.to(evidence.device)[target.long()]
        return (loss * sample_weights).sum() / (sample_weights.sum() + 1e-8)
    return loss.mean()
