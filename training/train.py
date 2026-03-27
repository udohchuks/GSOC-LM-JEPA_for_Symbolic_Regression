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
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader, random_split

from training.trainer import LLMJEPAModule
from data.aif_dataset import build_aif_dataloader
from data.synthetic_dataset import build_synthetic_dataloader


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
    cfg_ckpt     = config['checkpoint']
    cfg_hw       = config['hardware']
    cfg_pred     = cfg_model.get('predictor', {})

    # Data
    BATCH_SIZE   = cfg_data['batch_size']
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
    # Training & Validation: always use synthetic dataset (pre-split 90/10)
    full_synthetic_loader = build_synthetic_dataloader(
        n_equations=N_SYNTHETIC,
        batch_size=BATCH_SIZE,
        cache_path=cfg_data.get('synthetic_cache', 'cache/synthetic_10k.pt'),
        n_data_points=cfg_data.get('n_data_points', 1000),
        max_n_vars=MAX_N_VARS,
        num_workers=NUM_WORKERS,
    )

    # Split synthetic into 90% train / 10% val
    full_dataset = full_synthetic_loader.dataset
    val_size     = max(1, int(0.1 * len(full_dataset)))
    train_size   = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # reproducible split
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=full_synthetic_loader.collate_fn,
        num_workers=full_synthetic_loader.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=full_synthetic_loader.collate_fn,
        num_workers=full_synthetic_loader.num_workers,
    )

    # Test: always use the AIF (Feynman) dataset
    test_loader = build_aif_dataloader(
        csv_path=cfg_data['csv_path'],
        data_dir=cfg_data['data_dir'],
        batch_size=BATCH_SIZE,
        cache_dir=cfg_data.get('cache_dir'),
        num_workers=NUM_WORKERS,
    )

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

    # ── 4. Callbacks & Logger ────────────────────────────────────────────────
    logger = TensorBoardLogger(
        "tb_logs",
        name=cfg_log.get('run_name', 'llmjepa_sr_base'),
        log_graph=False,
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg_ckpt.get('dirpath', 'checkpoints/'),
        filename=cfg_ckpt.get('filename', 'jepa-{epoch:02d}-{val/total:.2f}'),
        monitor=cfg_ckpt.get('monitor', 'val/total'),
        mode=cfg_ckpt.get('mode', 'min'),
        save_top_k=cfg_ckpt.get('save_top_k', 3),
        save_last=True,
    )
    
    early_stop = EarlyStopping(
        monitor=cfg_ckpt.get('monitor', 'val/total'),
        patience=10,
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
        callbacks=[checkpoint_callback, early_stop, LearningRateMonitor()],
        log_every_n_steps=cfg_log.get('log_every_n_steps', 10),
        gradient_clip_val=GRADIENT_CLIP,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ── 6. Train ─────────────────────────────────────────────────────────────
    print(f"Starting training. Logs at: {logger.log_dir}")
    print("SIGReg mode: Both encoders trainable, NO EMA updates")
    trainer.fit(model, train_loader, val_loader)

    print("Training complete. Best checkpoint saved.")

    # ── 7. Test on AIF dataset ────────────────────────────────────────────────
    print("Running test evaluation on AIF (Feynman) dataset ...")
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    print("Test complete. View logs with: tensorboard --logdir tb_logs")


if __name__ == '__main__':
    main()