"""
Main Entry Point for Training LLM-JEPA with SIGReg.

Usage:
    python train.py
    python train.py --config configs/my_experiment.yaml

SIGReg Notes:
- Both context and target encoders are trainable
- NO EMA updates
- Collapse prevented by SIGReg loss on concatenated embeddings
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TensorFlow
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import os
from pathlib import Path
from torch.utils.data import DataLoader, Subset, Sampler

from training.trainer import LLMJEPAModule
from data.aif_dataset import build_aif_dataloader
from data.synthetic_dataset import build_synthetic_dataloader, LazySyntheticDataset
# Enable Tensor Core utilization on NVIDIA GPUs (A100, H100, L4, etc.)
torch.set_float32_matmul_precision('medium')


class ContiguousChunkSampler(Sampler):
    """Samples indices chunk-by-chunk to prevent massive Google Drive I/O thrashing."""
    def __init__(self, subset_start, subset_end, dataset):
        self.subset_start = subset_start
        self.subset_end = subset_end
        self.chunks = []
        
        # Guard against empty dataset
        if not hasattr(dataset, 'file_offsets'):
            self.chunks.append(list(range(subset_end - subset_start)))
            return
            
        for offset, size in zip(dataset.file_offsets, dataset.file_sizes):
            c_start = max(offset, subset_start)
            c_end = min(offset + size, subset_end)
            if c_start < c_end:
                self.chunks.append(list(range(c_start - subset_start, c_end - subset_start)))
                
    def __iter__(self):
        chunk_order = torch.randperm(len(self.chunks)).tolist()
        for c in chunk_order:
            items = self.chunks[c]
            shuffled_inner = torch.randperm(len(items)).tolist()
            for idx in shuffled_inner:
                yield items[idx]
                
    def __len__(self):
        return self.subset_end - self.subset_start


def main():
    parser = argparse.ArgumentParser(description="Train LLM-JEPA")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ── 1. Read Configuration ────────────────────────────────────────────────
    cfg_data     = config['data']
    cfg_model    = config['model']
    cfg_loss     = config['loss']
    cfg_train    = config['training']
    cfg_log      = config['logging']
    cfg_ckpt     = config.get('checkpoint', config.get('checkpoints', {}))
    cfg_hw       = config['hardware']
    cfg_pred     = cfg_model.get('predictor', {})

    # Data
    BATCH_SIZE   = cfg_data['batch_size']
    N_ROWS       = cfg_data.get('n_rows', 400)
    NUM_WORKERS  = cfg_data.get('num_workers', 2)

    # Model
    D_MODEL      = cfg_model['d_model']
    N_HEADS      = cfg_model['n_heads']
    N_ENC_LAYERS = cfg_model['n_enc_layers']
    N_DEC_LAYERS = cfg_model['n_dec_layers']
    N_ISAB       = cfg_model.get('n_isab', 2)
    N_COL_ATTN   = cfg_model.get('n_col_attn', 2)
    M_INDUCING   = cfg_model.get('m_inducing', 32)
    MAX_N_VARS   = cfg_model.get('max_n_vars', 9)
    DROPOUT      = cfg_model.get('dropout', 0.1)

    # Predictor
    PRED_N_HEADS          = cfg_pred.get('pred_n_heads', 4)
    PRED_BOTTLENECK_RATIO = cfg_pred.get('pred_bottleneck_ratio', 0.5)
    PRED_DROPOUT          = cfg_pred.get('pred_dropout', 0.1)

    # Training
    MAX_EPOCHS   = cfg_train['max_epochs']
    LR           = float(cfg_train['lr'])
    WEIGHT_DECAY = float(cfg_train.get('weight_decay', 1e-2))
    WARMUP_STEPS = cfg_train.get('warmup_steps', 500)
    GRADIENT_CLIP = cfg_train.get('gradient_clip', 1.0)
    N_SYNTHETIC  = cfg_data.get('n_synthetic', 10000)

    # ── 2. Data Loaders ──────────────────────────────────────────────────────
    # OPTION 4: Separate DataLoaders for Synthetic (train) and AIF (val/test)
    # Synthetic data has Y for MSE loss, AIF does not
    
    # Training: Synthetic data only
    full_dataset = LazySyntheticDataset(
        cache_dir=cfg_data.get('synthetic_cache', 'cache/synthetic_1M'),
        max_n_vars=MAX_N_VARS,
        n_rows=N_ROWS
    )

    # Contiguous split to preserve chunk boundaries
    val_size     = max(1, int(0.1 * len(full_dataset)))
    train_size   = len(full_dataset) - val_size

    train_dataset = Subset(full_dataset, range(0, train_size))
    val_dataset   = Subset(full_dataset, range(train_size, len(full_dataset)))

    # Use custom sampler to prevent Drive thrashing
    train_sampler = ContiguousChunkSampler(0, train_size, full_dataset)

    from data.aif_dataset import collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )
    # Validation: Also synthetic (same distribution as training)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2, # Validation doesn't need high parallelism
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    # Test/Evaluation: AIF (Feynman) dataset - separate loader, never mixed with train
    test_loader = build_aif_dataloader(
        csv_path=cfg_data['csv_path'],
        data_dir=cfg_data['data_dir'],
        batch_size=BATCH_SIZE,
        cache_dir=cfg_data.get('cache_dir', 'cache/'),
        num_workers=NUM_WORKERS,
    )
    
    print(f"✅ Data loaders ready:")
    print(f"   Train: {len(train_dataset)} synthetic equations")
    print(f"   Val:   {len(val_dataset)} synthetic equations")
    print(f"   Test:  {len(test_loader.dataset)} AIF equations (separate)")

    # ── 3. Model ─────────────────────────────────────────────────────────────
    model = LLMJEPAModule(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_encoder_layers=N_ENC_LAYERS,
        n_decoder_layers=N_DEC_LAYERS,
        max_n_vars=MAX_N_VARS,
        n_isab=N_ISAB,
        n_col_attn=N_COL_ATTN,
        m_inducing=M_INDUCING,
        dropout=DROPOUT,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        # Predictor
        pred_n_heads=PRED_N_HEADS,
        pred_bottleneck_ratio=PRED_BOTTLENECK_RATIO,
        pred_dropout=PRED_DROPOUT,
        # Loss weights
        lambda_jepa=cfg_loss.get('lambda_jepa', 1.0),
        lambda_sigreg=cfg_loss.get('lambda_sigreg', 1.0),
        lambda_lm=cfg_loss.get('lambda_lm', 1.0),
        lambda_units=cfg_loss.get('lambda_units', 1.0),
        # SIGReg / LM params
        sigreg_num_slices=cfg_loss.get('num_slices', 512),
        sigreg_num_points=cfg_loss.get('num_points', 17),
        invalid_weight=cfg_loss.get('alpha_lm', 2.0),
    )

    # Dynamic run name (now near-instant thanks to metadata manifest)
    from datetime import datetime
    n_k = len(full_dataset) // 1000
    timestamp = datetime.now().strftime("%m%d_%H%M")
    default_name = f"llmjepa_{n_k}k_eqs_{timestamp}"
    run_name = cfg_log.get('run_name', default_name)
    
    logger = TensorBoardLogger(
        "/content/drive/MyDrive/SymbolicRegression/tb_logs",  # Save logs to Drive
        name=run_name,
        log_graph=False,
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg_ckpt.get('dirpath', 'checkpoints/'),
        filename=cfg_ckpt.get('filename', 'jepa-{step:06d}-{val/total:.4f}'),
        monitor=cfg_ckpt.get('monitor', 'val/total'),
        mode=cfg_ckpt.get('mode', 'min'),
        save_top_k=cfg_ckpt.get('save_top_k', 3),
        every_n_train_steps=cfg_ckpt.get('save_every_n_steps', 500),
        save_last=True,
    )
    
    early_stop = EarlyStopping(
        monitor=cfg_ckpt.get('monitor', 'val/total'),
        patience=20,
        mode=cfg_ckpt.get('mode', 'min'),
        verbose=True,
    )

    # ── 5. Trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=cfg_hw.get('accelerator', 'auto'),
        devices=cfg_hw.get('devices', 1),
        strategy=cfg_hw.get('strategy', 'auto'),
        precision=cfg_hw.get('precision', '16-mixed'),
        logger=logger,
        callbacks=[
            checkpoint_callback, 
            early_stop, 
            LearningRateMonitor(), 
        ],
        log_every_n_steps=cfg_log.get('log_every_n_steps', 10),
        val_check_interval=int(cfg_train.get('val_check_interval', 500)),
        limit_val_batches=cfg_train.get('limit_val_batches', 1.0),
        gradient_clip_val=GRADIENT_CLIP,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ── 6. Train ─────────────────────────────────────────────────────────────
    # Auto-resume from last checkpoint if it exists
    ckpt_dir = cfg_ckpt.get('dirpath', 'checkpoints/')
    last_ckpt_path = Path(ckpt_dir) / "last.ckpt"
    resume_path = str(last_ckpt_path) if last_ckpt_path.exists() else None
    
    if resume_path:
        print(f"Resuming training from: {resume_path}")
    else:
        print(f"Starting fresh training. Logs at: {logger.log_dir}")
        
    print("SIGReg mode: Both encoders trainable, NO EMA updates")
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)

    print("Training complete. Best checkpoint saved.")

    # ── 7. Test on AIF dataset ────────────────────────────────────────────────
    print("Running test evaluation on AIF (Feynman) dataset ...")
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    print("Test complete. View logs with: tensorboard --logdir tb_logs")


if __name__ == '__main__':
    main()