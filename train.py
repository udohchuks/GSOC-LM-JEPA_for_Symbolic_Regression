
import os
import sys
import json
import pickle
import argparse
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.plotting import generate_all_plots

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from data.dataset      import FeynmanDataset, collate_fn
from tokenizer.vocab   import load_vocab
from model.vanilla_transformer import (
    CustomTransformer, CONFIG,
    build_causal_mask, build_jepa_mask
)
from model.sigreg import sigreg_loss, monitor_collapse

EXPERIMENTS = {
    # Baselines — no JEPA
    'B1': {'use_jepa': False, 'lambda_lejepa': 0.0},
    # JEPA + SIGReg — your core innovation
    'E2': {'use_jepa': True,  'lambda_lejepa': 0.1},
}


TRAIN_CONFIG = {
    'batch_size':    16,
    'n_epochs':      300,
    'lr':            3e-4,    # AdamW learning rate
    'weight_decay':  0.4,
    'grad_clip':     1.0,     # gradient clipping — prevents exploding gradients
    'alpha_drop':    0.3,     # JEPA dropout — skip JEPA loss 30% of steps
    'lambda_lejepa': 0.1,
    'save_every':    5,       # save checkpoint every N epochs
    'log_every':     10,      # print metrics every N steps
    'collapse_every':10,      # check for collapse every N steps
    'n_view_a':      40,      # rows sampled per equation for View A
    'n_bins':        64,
    'max_eq_len':    40,
    'num_projections': 1024,
}

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def build_dataloaders(splits, tok2id, cfg):
    """Build train, val, test DataLoaders."""
    vocab_size = len(tok2id)

    train_ds = FeynmanDataset(
        splits['train'], tok2id, vocab_size,
        n_view_a=cfg['n_view_a'],
        n_bins=cfg['n_bins'],
        max_eq_len=cfg['max_eq_len']
    )
    val_ds = FeynmanDataset(
        splits['val'], tok2id, vocab_size,
        n_view_a=cfg['n_view_a'],
        n_bins=cfg['n_bins'],
        max_eq_len=cfg['max_eq_len']
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg['batch_size'],
        shuffle     = True,
        collate_fn  = collate_fn,
        drop_last   = True,    # avoid batch size 1 which breaks BatchNorm
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        collate_fn  = collate_fn,
    )

    return train_loader, val_loader

def jepa_step(model, view_a, view_b, device,
              lambda_lejepa, global_step, num_projections ,alpha_drop):
    """
    One forward pass with JEPA + SIGReg loss.

    Returns:
        loss_lm:   language modelling loss
        loss_jepa: JEPA cosine distance loss (0 if dropped)
        loss_sig:  SIGReg collapse prevention loss
        skipped:   True if JEPA was dropped this step
    """
    B      = view_a.shape[0]
    len_a  = view_a.shape[1]
    len_b  = view_b.shape[1]

    # JEPA dropout — randomly skip JEPA loss alpha_drop fraction of steps
    # Reduces compute and acts as regularisation
    skip_jepa = (torch.rand(1).item() < alpha_drop)

    # Build PRED token — ID 3
    pred_tok = torch.full((B, 1), 3, dtype=torch.long, device=device)

    # Full sequence: [View A | View B | PRED]
    full_seq = torch.cat([view_a, view_b, pred_tok], dim=1)

    # Build JEPA attention mask
    mask = build_jepa_mask(len_a, len_b, device)

    # Forward pass
    hidden = model(full_seq, mask)
    
     # ── LM loss on View B positions ───────────────────────────────────
    # Use hidden states at View B positions to predict next token
    # Shift: predict token i+1 from position i
    hidden_b  = hidden[:, len_a:-1, :]        # [B, len_b, d]
    lm_logits = model.get_lm_logits(hidden_b) # [B, len_b, vocab]

    # Target is View B shifted left by 1
    lm_targets = view_b[:, 1:]                # [B, len_b-1]

    loss_lm = F.cross_entropy(
        lm_logits[:, :-1, :].reshape(-1, model.cfg['vocab_size']),
        lm_targets.reshape(-1),
        ignore_index=0   # ignore PAD
    )
    # ── JEPA loss ─────────────────────────────────────────────────────
    loss_jepa = torch.tensor(0.0, device=device)
    loss_sig  = torch.tensor(0.0, device=device)

    if not skip_jepa:
        # PRED hidden state — the prediction vector
        z_pred = hidden[:, -1, :]              # [B, d_model]

        # View B mean pooling — the target vector
        # Exclude PAD tokens from the mean
        pad_mask = (view_b != 0).float().unsqueeze(-1)  # [B, len_b, 1]
        z_target = (hidden_b * pad_mask).sum(1) / \
                    pad_mask.sum(1).clamp(min=1e-8)      # [B, d_model]

        # L2 distance
        loss_jepa = ((z_pred - z_target) ** 2).mean()

        # SIGReg — force z_pred toward isotropic Gaussian
        # This prevents collapse without needing an EMA teacher
        loss_sig = (
            sigreg_loss(z_pred, global_step=global_step*2, num_projections=num_projections)
            + sigreg_loss(z_target, global_step=global_step*2+1, num_projections=num_projections)
        )


    
    loss_lejepa = lambda_lejepa * loss_sig + (1 - lambda_lejepa) * loss_jepa

    loss = loss_lm + loss_lejepa
    return loss_lm, loss_jepa, loss_sig, loss, skip_jepa


def baseline_step(model, view_b, device):
    """
    One forward pass with standard LM loss only — no JEPA.
    Used for baseline experiments B1-B4.
    """
    len_b = view_b.shape[1]
    mask  = build_causal_mask(len_b, device)
    hidden = model(view_b, mask)

    lm_logits = model.get_lm_logits(hidden)
    loss_lm   = F.cross_entropy(
        lm_logits[:, :-1, :].reshape(-1, model.cfg['vocab_size']),
        view_b[:, 1:].reshape(-1),
        ignore_index=0
    )
    return loss_lm

def train_one_epoch(model, loader, optimiser,
                    exp_cfg, train_cfg, device, epoch, collapse_log=None):
    """Train for one epoch. Returns dict of average losses."""
    model.train()

    total_lm   = 0.0
    total_jepa = 0.0
    total_sig  = 0.0
    total_loss = 0.0
    n_steps    = 0

    global_step = (epoch - 1) * len(loader)

    for step, (view_a, view_b, n_vars) in enumerate(loader):
        view_a = view_a.to(device)
        view_b = view_b.to(device)

        optimiser.zero_grad()

        # ── Forward pass ──────────────────────────────────────────────
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            if exp_cfg['use_jepa']:
                loss_lm, loss_jepa, loss_sig, loss, skipped = jepa_step(
                    model, view_a, view_b, device,
                    lambda_lejepa = exp_cfg['lambda_lejepa'],
                    global_step   = global_step, 
                    alpha_drop  = train_cfg['alpha_drop'],
                    num_projections= train_cfg['num_projections']
                )
            else:
                loss_lm   = baseline_step(model, view_b, device)
                loss_jepa = torch.tensor(0.0)
                loss_sig  = torch.tensor(0.0)
                loss      = loss_lm

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_lm   += loss_lm.item()
        total_jepa += loss_jepa.item()
        total_sig  += loss_sig.item()
        total_loss += loss.item()
        n_steps    += 1
        global_step += 1

        if step % train_cfg['log_every'] == 0:
            print(f"  Ep{epoch:02d} step{step:03d} | "
                  f"loss={loss.item():.4f} "
                  f"lm={loss_lm.item():.4f} "
                  f"jepa={loss_jepa.item():.4f} "
                  f"sig={loss_sig.item():.4f}")

        if exp_cfg['use_jepa'] and step % train_cfg['collapse_every'] == 0:
            with torch.no_grad():
                # Get z_pred for monitoring
                pred_tok  = torch.full((view_a.shape[0], 1), 3,
                                       dtype=torch.long, device=device)
                full_seq  = torch.cat([view_a, view_b, pred_tok], dim=1)
                mask      = build_jepa_mask(view_a.shape[1],
                                            view_b.shape[1], device)
                hidden    = model(full_seq, mask)
                z_pred    = hidden[:, -1, :]
                health    = monitor_collapse(z_pred)
                health['step']  = (epoch - 1) * len(loader) + step
                health['epoch'] = epoch
                if collapse_log is not None:
                    collapse_log.append(health)

                if health['collapsed']:
                    print(f"  ⚠️  COLLAPSE DETECTED at step {health['step']}!")
                    print(f"      variance={health['variance']:.4f}  "
                          f"cosine={health['mean_cosine']:.4f}")
                    print(f"      Consider increasing lambda_sig.")
    return {
        'loss_lm':   total_lm   / n_steps,
        'loss_jepa': total_jepa / n_steps,
        'loss_sig':  total_sig  / n_steps,
        'loss_total':total_loss / n_steps,
    }
            
@torch.no_grad()
def validate(model, loader, exp_cfg, train_cfg, device):
    """Validation pass. Returns average LM loss."""
    model.eval()
    total_lm = 0.0
    n_steps  = 0

    for view_a, view_b, n_vars in loader:
        view_a = view_a.to(device)
        view_b = view_b.to(device)

        # Always use LM loss only for validation
        # (JEPA is a training signal, not a validation metric)
        len_b  = view_b.shape[1]
        mask   = build_causal_mask(len_b, device)
        hidden = model(view_b, mask)

        lm_logits = model.get_lm_logits(hidden)
        loss_lm   = F.cross_entropy(
            lm_logits[:, :-1, :].reshape(-1, model.cfg['vocab_size']),
            view_b[:, 1:].reshape(-1),
            ignore_index=0
        )
        total_lm += loss_lm.item()
        n_steps  += 1

    return {'val_loss_lm': total_lm / n_steps}

def save_checkpoint(model, optimiser, epoch, metrics,
                    exp_name, save_dir):
    """Save model checkpoint to disk."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{exp_name}_epoch{epoch:02d}.pt')
    torch.save({
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimiser':  optimiser.state_dict(),
        'metrics':    metrics,
        'config':     model.cfg,
    }, path)
    print(f"  Checkpoint saved: {path}")
    return path

def load_checkpoint(path, model, optimiser=None):
    """Load model checkpoint from disk."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimiser is not None:
        optimiser.load_state_dict(ckpt['optimiser'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    return ckpt['epoch'], ckpt['metrics']

def main(exp_name: str):
    device     = get_device()
    exp_cfg    = EXPERIMENTS[exp_name]
    train_cfg  = TRAIN_CONFIG
    save_dir   = f'checkpoints/{exp_name}'
    results_path = f'results/{exp_name}_train_log.json'

    print(f"Experiment : {exp_name}")
    print(f"Device     : {device}")
    print(f"Use JEPA   : {exp_cfg['use_jepa']}")
    print()

    # ── Load data ─────────────────────────────────────────────────────
    with open('data/processed/records.pkl', 'rb') as f:
        splits = pickle.load(f)
    tok2id, id2tok = load_vocab('data/processed')

    train_loader, val_loader = build_dataloaders(
        splits, tok2id, train_cfg
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # ── Build model ───────────────────────────────────────────────────
    model = CustomTransformer(CONFIG).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # ── Optimiser ─────────────────────────────────────────────────────
    # AdamW with cosine annealing — standard for transformer training
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr           = train_cfg['lr'],
        weight_decay = train_cfg['weight_decay'],
        betas        = (0.9, 0.95),   # slightly higher beta2 for stability
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max  = train_cfg['n_epochs'],
        eta_min= train_cfg['lr'] / 10,   # decay to 10% of initial LR
    )

    # ── Training loop ─────────────────────────────────────────────────
    history = []
    collapse_log = []
    best_val_loss = float('inf')
    patience     = 30
    no_improve   = 0


    for epoch in range(1, train_cfg['n_epochs'] + 1):
        print(f"\nEpoch {epoch}/{train_cfg['n_epochs']}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimiser,
            exp_cfg, train_cfg, device, epoch,
            collapse_log=collapse_log
        )

        # Validate
        val_metrics = validate(
            model, val_loader, exp_cfg, train_cfg, device
        )

        # Step scheduler
        scheduler.step()

        # Log
        metrics = {
            'epoch': epoch,
            **train_metrics,
            **val_metrics,
            'lr': scheduler.get_last_lr()[0],
        }
        history.append(metrics)

        print(f"  train_loss={train_metrics['loss_total']:.4f} "
              f"val_loss={val_metrics['val_loss_lm']:.4f} "
              f"lr={metrics['lr']:.2e}")

        # Save checkpoint
        if epoch % train_cfg['save_every'] == 0:
            save_checkpoint(
                model, optimiser, epoch, metrics,
                exp_name, save_dir
            )

        if val_metrics['val_loss_lm'] < best_val_loss:
            best_val_loss = val_metrics['val_loss_lm']
            no_improve    = 0
            save_checkpoint(model, optimiser, epoch, metrics,
                            f'{exp_name}_best', save_dir)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                    f"— no val improvement for {patience} epochs")
                break
        
    # ── Save training log ─────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining log saved to {results_path}")

    return history, exp_cfg, collapse_log

if __name__ == '__main__':
    exp_name = 'E2'
    history, exp_cfg, collapse_log = main(exp_name)
    generate_all_plots(
        history      = history,
        exp_name     = exp_name,
        exp_cfg      = exp_cfg,
        collapse_log = collapse_log,   # collected during training
        save_dir     = 'results/plots'
    )