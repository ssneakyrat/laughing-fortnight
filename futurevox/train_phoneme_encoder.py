import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from utils.config import load_config, setup_logger
from modules.phoneme_encoder_lightning import PhonemeEncoderLightning
from data.dataset import create_dataloader


def main(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Set up logger
    logger = setup_logger(config)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = create_dataloader(config, split='train')
    val_dataloader = create_dataloader(config, split='valid')
    
    # Create model
    logger.info("Creating phoneme encoder model...")
    model = PhonemeEncoderLightning(config)
    
    # Set up callbacks
    logger.info("Setting up callbacks...")
    callbacks = [
        # Learning rate monitor
        LearningRateMonitor(logging_interval='epoch'),
        
        # Save checkpoint on validation loss improvement
        ModelCheckpoint(
            dirpath=config['logging']['checkpoint_dir'],
            filename='phoneme_encoder-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
    ]
    
    # Set up TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name='phoneme_encoder'
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        max_epochs=config['model']['max_epochs'],
        gradient_clip_val=config['model']['clip_grad_norm'],
        check_val_every_n_epoch=config['model']['validate_every_n_epochs'],
        log_every_n_steps=10,
        accelerator='auto',  # Use GPU if available
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Save final model
    logger.info("Saving final model...")
    final_model_path = os.path.join(
        config['logging']['checkpoint_dir'],
        'phoneme_encoder_final.ckpt'
    )
    trainer.save_checkpoint(final_model_path)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the FutureVox+ Phoneme Encoder')
    parser.add_argument('--config', type=str, default='futurevox/config/default.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    main(args)