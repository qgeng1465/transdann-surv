#!/usr/bin/env python3
"""
method_utils.py; Domain-Adaptation / Domain-Generalization methods on TransDANNSurvV3
========================================================================================
Implements the method-comparison toolbox for Experiment J.  Every method shares the
SAME backbone (TransDANNSurvV3 + DeepHit 32-bin survival head) and the SAME training
configuration (AdamW 5e-4, cosine annealing, early-stop patience=40, per-cohort loss).
The only difference between methods is the extra regularisation / adversarial /
alignment term added on top of the survival loss.

Methods
-------
  ERM        : Empirical Risk Minimisation (= Baseline, no extra term)
  DANN       : Domain-adversarial (Gradient Reversal Layer)            [Ganin et al. 2016]
  CORAL      : Feature covariance alignment                            [Sun & Saenko 2016]
  IRM        : Invariant Risk Minimisation, pen. = Σ_d ||∇_w L_d||²    [Arjovsky et al. 2019]
  GroupDRO   : Worst-group (max-domain) loss                           [Sagawa et al. 2020]
  V-REx      : Mean + λ·var of domain losses                           [Krueger et al. 2021]
  Fish       : Gradient-agreement penalty −Σ cos(∇L_i, ∇L_j)           [Shi et al. 2021]
  Mixup      : Cross-domain feature interpolation (two-label loss)     [Zhang et al. 2018]
  MLDG       : Leave-one-domain-out meta-learning                      [Li et al. 2018]

NOTE ON FISH / MLDG implementation:
  - Fish uses per-domain gradients computed via ``autograd.grad(..., retain_graph=True)``.
  - MLDG uses a first-order variant (inner update simulated with ``create_graph=False``
    then the meta-test loss is evaluated at the *inner* parameters via
    ``torch.func.functional_call``).  This is the Reptile-style first-order approximation
    of Li et al.'s meta-learning objective and is far more stable on this dataset.
"""

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Per-domain survival losses
# ---------------------------------------------------------------------------
def per_domain_losses(loss_fn, surv_out, t, e, d, n_domains, min_samples=5):
    """Compute the survival loss separately for each domain (list of scalars).

    Domains with fewer than ``min_samples`` rows are skipped (mirrors the
    existing ``per_cohort_loss`` behaviour in ``02_train_dann_lihc.py``).
    """
    losses = []
    for dom in range(n_domains):
        mask = d == dom
        if int(mask.sum()) < min_samples:
            continue
        losses.append(loss_fn(surv_out[mask], t[mask], e[mask]))
    return losses


# ---------------------------------------------------------------------------
# Method-specific extra terms (all return a scalar Tensor)
# ---------------------------------------------------------------------------
def coral_penalty(cls_feats, d, n_domains):
    """CORAL: penalise the Frobenius distance between per-domain covariances.

    L_coral = (1/4d²) Σ_{i<j} ||C_i - C_j||²_F   where C_d is the d×d feature
    covariance of domain d and d = number of feature dimensions.
    """
    dim = cls_feats.size(1)
    covs = []
    for dom in range(n_domains):
        mask = d == dom
        if int(mask.sum()) < 2:
            continue
        fd = cls_feats[mask]
        fd_c = fd - fd.mean(dim=0, keepdim=True)
        cov = (fd_c.t() @ fd_c) / max(fd.size(0) - 1, 1)
        covs.append(cov)
    if len(covs) < 2:
        return torch.zeros((), device=cls_feats.device)
    pen = torch.zeros((), device=cls_feats.device)
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            pen = pen + torch.sum((covs[i] - covs[j]) ** 2)
    return pen / (4.0 * dim * dim)


def irmv1_penalty(loss_fn, hazard_logits, t, e, d, n_domains):
    """IRMv1 penalty (DomainBed-style): Σ_d (d/ds L_d(hazard_logits·s))² at s=1.

    Penalises the *sensitivity* of each domain's survival loss to a scalar
    scaling of the survival-head output.  This is the canonical stable form of
    Invariant Risk Minimisation (Arjovsky et al. 2019) as implemented in
    DomainBed; the full-parameter ||∇_w L_d||² form is notoriously unstable and
    collapses to near-random solutions on this data.

    ``create_graph=True`` keeps the (scalar) second-order path alive so the
    outer ``loss_total.backward()`` differentiates through the penalty.
    """
    device = hazard_logits.device
    s = torch.ones((), device=device, requires_grad=True)
    penalty = torch.zeros((), device=device)
    for dom in range(n_domains):
        mask = d == dom
        if int(mask.sum()) < 5:
            continue
        loss_i = loss_fn(hazard_logits[mask] * s, t[mask], e[mask])
        g = torch.autograd.grad(loss_i, [s], create_graph=True)[0]
        penalty = penalty + g ** 2
    return penalty


def fish_penalty(params, losses):
    """Fish: −Σ_{i≠j} cos(∇L_i, ∇L_j); penalise gradient disagreement.

    ``create_graph=True`` keeps the agreement term differentiable w.r.t. the
    parameters (same rationale as IRM).
    """
    device = losses[0].device
    gs = []
    for L_d in losses:
        g = torch.autograd.grad(
            L_d, params, retain_graph=True, allow_unused=True, create_graph=True
        )
        flat = torch.cat([x.reshape(-1) for x in g if x is not None])
        gs.append(flat)
    if len(gs) < 2:
        return torch.zeros((), device=device)
    pen = torch.zeros((), device=device)
    cnt = 0.0
    for i in range(len(gs)):
        for j in range(len(gs)):
            if i == j:
                continue
            denom = gs[i].norm() * gs[j].norm() + 1e-8
            cos = torch.sum(gs[i] * gs[j]) / denom
            pen = pen - cos
            cnt += 1.0
    return pen / cnt


def vrex_penalty(losses):
    """V-REx: variance of the per-domain losses (biased/unbiased variance)."""
    if len(losses) < 2:
        return torch.zeros((), device=losses[0].device)
    stacked = torch.stack(losses)
    return torch.var(stacked, unbiased=True)


def groupdro_loss(losses):
    """GroupDRO: worst-group loss L = max_d L_d."""
    return torch.stack(losses).max()


def mixup_pair_perm(d, rng):
    """Build a permutation that pairs every sample with a *different* domain.

    Used so that Mixup interpolates *across* domains rather than within one.
    Returns an int array ``perm`` with d[i] != d[perm[i]] for every i where at
    least two domains are present in the batch.
    """
    n = len(d)
    d_np = np.asarray(d.cpu().numpy())
    perm = np.empty(n, dtype=np.int64)
    for i in range(n):
        cand = np.where(d_np != d_np[i])[0]
        if len(cand) == 0:
            perm[i] = i
        else:
            perm[i] = cand[rng.randint(0, len(cand))]
    return torch.from_numpy(perm).to(d.device)


# ---------------------------------------------------------------------------
# Survival-probability helper (DeepHit hazards → S(t|x) at arbitrary times)
# ---------------------------------------------------------------------------
def deephit_survival_probabilities(hazard_logits, bin_edges):
    """Convert DeepHit hazard logits to survival probabilities S(t).

    Returns a dict-like function isn't convenient; instead we return the
    per-bin survival values S(bin_center_k) = 1 − Σ_{j≤k} h_j.

    Parameters
    ----------
    hazard_logits : (batch, n_bins) tensor of logits
    bin_edges     : (n_bins+1,) tensor of bin edges

    Returns
    -------
    surv_at_centers : (batch, n_bins) tensor; S evaluated at each bin centre
    bin_centers     : (n_bins,) tensor
    """
    hazards = F.softmax(hazard_logits, dim=1)
    cdf = torch.cumsum(hazards, dim=1)
    surv = (1.0 - cdf).clamp(min=1e-8, max=1.0)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return surv, centers


def survival_at_times(surv_at_centers, bin_centers, times):
    """Evaluate S(t|x) at arbitrary times using the per-bin survival curve.

    surv_at_centers : (batch, n_bins)
    bin_centers     : (n_bins,)
    times           : (n_eval,) evaluation grid (on model device)
    Returns S at each eval time: (batch, n_eval).
    """
    # For each eval time, S(τ) = S at the last bin centre ≤ τ.
    idx = torch.searchsorted(bin_centers.to(surv_at_centers.device), times.to(surv_at_centers.device), right=True) - 1
    idx = idx.clamp(0, surv_at_centers.size(1) - 1)
    # out[:, k] = surv_at_centers[:, idx[k]]
    out = surv_at_centers[:, idx]
    return out
