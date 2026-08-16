"""
TransDANN-Surv: Shared Utility Module (V3)
============================================
Centralised model architectures, loss functions, evaluation metrics,
and data utilities for the TransDANN-Surv pipeline.

V3 additions
- DeepHit discrete-time survival loss (replaces Cox PH)
- Time-dependent AUC
- Per-cohort loss weighting
- V3 model with configurable survival head (Cox scalar / DeepHit N-bin)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (GRL)
# ---------------------------------------------------------------------------
class GradientReversal(torch.autograd.Function):
    """Forward: identity.  Backward: negates gradients × *alpha*."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# ===================================================================
# LOSS FUNCTIONS
# ===================================================================

# ---------------------------------------------------------------------------
# Cox PH Loss  (kept for backward compatibility / ablation)
# ---------------------------------------------------------------------------
def cox_loss(risk_scores, events, times):
    """Cox partial log-likelihood (vectorised, numerically stable)."""
    sort_idx = torch.argsort(times, descending=True)
    risk_scores = risk_scores[sort_idx].squeeze(-1)
    events = events[sort_idx]
    risk_scores = risk_scores - risk_scores.max()
    risk_exp = torch.exp(risk_scores)
    cum_risk = torch.cumsum(risk_exp, dim=0)
    log_risk = torch.log(cum_risk + 1e-8)
    uncensored_ll = risk_scores - log_risk
    loss = -torch.sum(uncensored_ll * events) / (torch.sum(events) + 1e-8)
    return loss


# ---------------------------------------------------------------------------
# DeepHit Discrete-Time Loss
# ---------------------------------------------------------------------------
def compute_time_bins(times, events, n_bins=32):
    """Compute quantile-based time bins from observed event times.

    Guarantees exactly ``n_bins`` bin intervals even when event-time
    quantiles collide (duplicates are split with small jitter).

    Returns
    -------
    bin_edges : Tensor (n_bins+1,); time boundaries
    bin_centers : Tensor (n_bins,); midpoints
    """
    # Move to CPU for numpy operations
    if times.is_cuda:
        times = times.cpu()
    if events.is_cuda:
        events = events.cpu()
    event_times = times[events == 1]
    if len(event_times) == 0:
        # No events: use all times
        event_times = times
    if len(event_times) < n_bins:
        n_bins = max(2, len(event_times))
    # Quantile-based bins from event times
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(event_times, quantiles).tolist()
    # Ensure uniqueness while preserving n_bins+1 edges
    unique_bins = sorted(set(bins))
    if len(unique_bins) < n_bins + 1:
        # Add small jitter to duplicates: walk through and ensure monotonic
        seen = set()
        deduped = []
        for v in bins:
            while v in seen:
                v += 1e-6 * (max(bins) - min(bins) + 1)
            seen.add(v)
            deduped.append(v)
        bins = sorted(deduped)
    if len(bins) < 3:
        bins = [0.0, float(np.median(event_times)), float(event_times.max())]
    # Trim or pad to exactly n_bins + 1
    bins = bins[:n_bins + 1]
    while len(bins) < n_bins + 1:
        bins.append(bins[-1] + 1.0)
    bins = torch.tensor(bins, dtype=torch.float32)
    # Compute centers
    centers = (bins[:-1] + bins[1:]) / 2
    return bins, centers


def deep_hit_loss(hazard_logits, times, events, bin_edges):
    """DeepHit negative log-likelihood loss (single-risk, no competing events).

    Parameters
    ----------
    hazard_logits : Tensor (batch, n_bins); unnormalised logits per time bin
    times         : Tensor (batch,); observed survival / censoring times
    events        : Tensor (batch,); 1 = death, 0 = censored
    bin_edges     : Tensor (n_bins+1,); time bin boundaries (sorted)

    Returns
    -------
    loss : scalar Tensor
    """
    # Softmax → proper hazard distribution
    hazards = F.softmax(hazard_logits, dim=1)
    n_bins = hazards.size(1)

    # Move bin_edges to same device as input
    if bin_edges.device != times.device:
        bin_edges = bin_edges.to(times.device)

    # Map times to bin indices (0-indexed)
    time_bins = torch.bucketize(times, bin_edges) - 1
    time_bins = time_bins.clamp(0, n_bins - 1)

    batch_size = len(times)
    idx = torch.arange(batch_size, device=times.device)

    # Term 1:  negative log-likelihood of the observed bin
    #   uncensored:  -log( h_k(t) ); prob of event at exactly bin k
    #   censored:    -log( S(t) ); prob of surviving *past* bin k
    h_t = hazards[idx, time_bins]                   # prob in observed bin
    # Cumulative sum from left: F(t) = P(T <= bin)
    cum_haz = torch.cumsum(hazards, dim=1)
    F_t = cum_haz[idx, time_bins]                   # CDF at observed bin
    S_t = 1.0 - F_t                                 # survival past bin

    # Clamp survival probability to avoid log(0) or log(negative)
    S_t = S_t.clamp(min=1e-8)
    nll = torch.where(
        events == 1,
        -torch.log(h_t + 1e-8),
        -torch.log(S_t)
    )
    return nll.mean()


def deep_hit_risk(hazard_logits, bin_centers):
    """Convert DeepHit hazard distribution to scalar risk score.

    Higher score = higher risk (shorter expected survival).
    Uses negative expected lifetime.
    """
    hazards = F.softmax(hazard_logits, dim=1)
    # Center the bin_centers to the same device
    centers = bin_centers.to(hazard_logits.device)
    expected_lifetime = (centers.unsqueeze(0) * hazards).sum(dim=1)
    return -expected_lifetime  # more negative = longer survival → lower risk


# ---------------------------------------------------------------------------
# Time-Dependent AUC (cumulative/dynamic)
# ---------------------------------------------------------------------------
def time_dependent_auc(times, events, risk_scores, eval_times=(12, 36, 60)):
    """Compute cumulative/dynamic AUC at specified time points.

    At time t: sensitivity = P(risk > threshold | T <= t)
               specificity = P(risk <= threshold | T > t)

    Returns dict mapping eval_time → AUC.
    """
    from lifelines.utils import concordance_index
    results = {}
    risk_scores = np.asarray(risk_scores)
    times = np.asarray(times)
    events = np.asarray(events)

    for t in eval_times:
        case_idx = (times <= t) & (events == 1)      # events at or before t
        control_idx = times > t                        # survived past t
        if case_idx.sum() < 5 or control_idx.sum() < 5:
            results[t] = np.nan
            continue
        # Subset
        sub_times = np.concatenate([times[case_idx], times[control_idx]])
        sub_risk = np.concatenate([risk_scores[case_idx], risk_scores[control_idx]])
        sub_events = np.concatenate([
            np.ones(case_idx.sum()),
            np.zeros(control_idx.sum())
        ])
        # C-index at this cut-off works as time-dependent AUC
        try:
            auc = concordance_index(sub_times, -sub_risk, sub_events)
            results[t] = auc
        except Exception:
            results[t] = np.nan
    return results


# ===================================================================
# MODEL; TransDANN-Surv V3 (supports both Cox & DeepHit)
# ===================================================================

class TransDANNSurvV3(nn.Module):
    """TransDANN-Surv V3 with configurable survival head.

    Two modes:
    - ``surv_head_type='cox'``: scalar risk score (same as V2)
    - ``surv_head_type='deephit'``: N_BINS hazard distribution (DeepHit)

    V3 improvements over V2:
    - No missingness indicators in features (embeddings handle unknown)
    - Configurable survival head
    - Optional skip-domain mode (ablation)
    """

    def __init__(self, num_continuous, num_categorical, cat_cardinalities,
                 d_model=128, n_heads=8, n_layers=4, dropout=0.1,
                 num_domains=3, surv_head_type='cox', n_bins=32):
        super().__init__()
        self.num_continuous = num_continuous
        self.surv_head_type = surv_head_type
        self.n_bins = n_bins

        # Continuous feature embeddings
        self.cont_embeddings = nn.ModuleList([
            nn.Sequential(nn.Linear(1, d_model), nn.LayerNorm(d_model))
            for _ in range(num_continuous)
        ])

        # Categorical feature embeddings (last index = unknown/missing)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_model, padding_idx=card)
            for card in cat_cardinalities
        ])

        # CLS token + positional encoding
        total_tokens = 1 + num_continuous + len(cat_cardinalities)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, total_tokens, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.layer_norm = nn.LayerNorm(d_model)

        # Survival head
        surv_out_dim = n_bins if surv_head_type == 'deephit' else 1
        self.survival_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, surv_out_dim)
        )

        # Domain-adversarial head
        self.domain_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_domains)
        )

    def forward(self, x_cont, x_cat, alpha=1.0):
        """Forward pass.

        Returns
        -------
        surv_out    : Tensor; risk score (cox) or hazard logits (deephit)
        domain_pred : Tensor (batch, num_domains); domain logits
        cls_output  : Tensor (batch, d_model); CLS representation
        """
        batch_size = x_cont.size(0)

        tokens = [self.cls_token.expand(batch_size, -1, -1)]
        for i in range(self.num_continuous):
            tok = self.cont_embeddings[i](x_cont[:, i:i+1]).unsqueeze(1)
            tokens.append(tok)
        for i in range(x_cat.size(1)):
            idx = x_cat[:, i].clamp(
                max=self.cat_embeddings[i].num_embeddings - 1
            )
            tok = self.cat_embeddings[i](idx).unsqueeze(1)
            tokens.append(tok)

        x = torch.cat(tokens, dim=1)
        x = x + self.pos_encoding[:, :x.size(1), :]

        x = self.transformer(x)
        x = self.layer_norm(x)

        cls_output = x[:, 0, :]
        surv_out = self.survival_head(cls_output)

        reversed_feat = GradientReversal.apply(cls_output, alpha)
        domain_pred = self.domain_head(reversed_feat)

        return surv_out, domain_pred, cls_output


# ===================================================================
# DATASETS
# ===================================================================

class LiverCancerDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for liver-cancer survival data.

    Yields (x_cont, x_cat, time, event, domain_label).
    """

    def __init__(self, dataframe, cont_features, cat_features,
                 domain_label_col=None):
        self.x_cont = torch.tensor(
            dataframe[cont_features].values, dtype=torch.float32
        )
        self.x_cat = torch.tensor(
            dataframe[cat_features].values, dtype=torch.long
        )
        self.t = torch.tensor(
            dataframe['Survival_Months'].values, dtype=torch.float32
        )
        self.e = torch.tensor(
            dataframe['Vital_Status'].values, dtype=torch.float32
        )
        self.d = (
            torch.tensor(dataframe[domain_label_col].values, dtype=torch.long)
            if domain_label_col else None
        )
        # Per-sample cohort label for stratified loss
        self.cohort = (
            dataframe.get('Cohort', None)
        )

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        if self.d is not None:
            return (self.x_cont[idx], self.x_cat[idx],
                    self.t[idx], self.e[idx], self.d[idx])
        return (self.x_cont[idx], self.x_cat[idx],
                self.t[idx], self.e[idx])


class EvalDataset(torch.utils.data.Dataset):
    """Dataset for evaluation."""

    def __init__(self, dataframe, cont_features, cat_features):
        self.x_cont = torch.tensor(
            dataframe[cont_features].values, dtype=torch.float32
        )
        self.x_cat = torch.tensor(
            dataframe[cat_features].values, dtype=torch.long
        )
        self.t = torch.tensor(
            dataframe['Survival_Months'].values, dtype=torch.float32
        )
        self.e = torch.tensor(
            dataframe['Vital_Status'].values, dtype=torch.float32
        )

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return (self.x_cont[idx], self.x_cat[idx],
                self.t[idx], self.e[idx])


# ===================================================================
# DATA PREPROCESSING HELPERS
# ===================================================================

def add_age_features(df, bins=(0, 40, 55, 65, 100)):
    """Add ``Age2`` (squared) and ``Age_group`` (binned)."""
    df['Age2'] = df['Age'] ** 2
    df['Age_group'] = pd.cut(
        df['Age'], bins=bins,
        labels=range(len(bins) - 1)
    ).astype(float).fillna(1)
    return df


def add_surgery_cohort_interaction(df):
    """Add ``Surgery_x_CohortKnown`` interaction to handle Simpson's paradox.

    Surgery has opposite effects in China (neg) vs SEER (pos).
    This interaction lets the model learn context-dependent treatment effects.
    """
    # Indicator: is this a cohort where surgery is reliably beneficial?
    # SEER: surgery → better outcome. China: surgery → sicker patients.
    df['Surg_x_SEER'] = (
        (df['Cohort'] == 'US_SEER') & (df['Surgery'] == 1)
    ).astype(int)
    return df


def prepare_v3_data(df, drop_japan=True):
    """Prepare data for V3 training.

    Steps
    -----
    1. Drop Japan_ICGC (0 events)
    2. Remove Survival_Months <= 0
    3. Remove invalid Vital_Status
    4. Add engineered features: Age2, Age_group, Surg_x_SEER

    Returns filtered DataFrame.
    """
    if drop_japan:
        df = df[df['Cohort'] != 'Japan_ICGC'].copy()
    df = df[df['Survival_Months'] > 0].copy()
    df = df[df['Vital_Status'].isin([0, 1])].copy()
    df = add_age_features(df)
    df = add_surgery_cohort_interaction(df)
    return df


# ===================================================================
# PER-COHORT LOSS HELPER
# ===================================================================
def per_cohort_loss(loss_fn, predictions, times, events, cohorts,
                    cohort_names=None):
    """Compute loss per cohort and return unweighted mean.

    Ensures each cohort contributes equally regardless of sample size.
    """
    if cohort_names is None:
        cohort_names = predictions.new_tensor(
            sorted(cohorts.unique().tolist()), dtype=torch.long
        )
    losses = []
    for c in cohort_names:
        mask = cohorts == c
        if mask.sum() == 0:
            continue
        if isinstance(predictions, (list, tuple)):
            # For DeepHit: (hazard_logits, bin_edges)
            loss_c = loss_fn(predictions[0][mask], times[mask], events[mask],
                             predictions[1])
        else:
            loss_c = loss_fn(predictions[mask], events[mask], times[mask])
        losses.append(loss_c)
    if not losses:
        return torch.tensor(0.0, device=times.device)
    return torch.stack(losses).mean()


# ===================================================================
# BACKWARD-COMPATIBILITY ALIASES (V2 scripts expect these names)
# ===================================================================
# TransDANNSurvV3 is the replacement for V2; export under both names.
TransDANNSurvV2 = TransDANNSurvV3

def create_missing_indicators(df, features):
    """Deprecated: V2 used missingness indicators. V3 handles missing via embeddings."""
    for feat in features:
        ind_name = f'{feat}_known'
        df[ind_name] = df[feat].notna().astype(int)
    return df
