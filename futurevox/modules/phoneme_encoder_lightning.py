import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from models.phoneme_encoder import PhonemeEncoder
import os


class PhonemeEncoderLightning(pl.LightningModule):
    """
    PyTorch Lightning module for training the PhonemeEncoder
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create the phoneme encoder model
        self.phoneme_encoder = PhonemeEncoder(
            num_phonemes=config['model']['phoneme_encoder']['num_phonemes'],
            embedding_dim=config['model']['phoneme_encoder']['embedding_dim'],
            hidden_dim=config['model']['phoneme_encoder']['hidden_dim'],
            n_layers=config['model']['phoneme_encoder']['n_layers'],
            kernel_size=config['model']['phoneme_encoder']['kernel_size'],
            dropout=config['model']['phoneme_encoder']['dropout'],
            dilation_rates=config['model']['phoneme_encoder']['dilation_rates']
        )
        
        # Create a simple decoder to reconstruct phoneme sequence as a pretraining task
        # This helps to ensure the encoder learns meaningful representations
        self.decoder = nn.Linear(
            self.phoneme_encoder.get_output_dim(), 
            config['model']['phoneme_encoder']['num_phonemes']
        )
        
        # Loss function - Cross Entropy for phoneme classification (pretraining)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, phoneme_ids, midi_notes=None):
        """
        Forward pass through the phoneme encoder
        
        Args:
            phoneme_ids: [B, T] - Batch of phoneme sequences
            midi_notes: [B, T] - Optional batch of MIDI note sequences
            
        Returns:
            linguistic_features: [B, T, H] - Batch of linguistic features
        """
        return self.phoneme_encoder(phoneme_ids, midi_notes)
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Dictionary containing:
                - phoneme_ids: [B, T] - Batch of phoneme sequences
                - midi_notes: [B, T] - Optional batch of MIDI note sequences (if available)
            batch_idx: Index of the batch
            
        Returns:
            loss: Training loss
        """
        # Get data from batch
        phoneme_ids = batch['phoneme_ids']
        midi_notes = batch.get('midi_notes', None)
        
        # Create targets for reconstruction (shifted right)
        # This is a simple pretraining task to ensure meaningful representations
        targets = torch.roll(phoneme_ids, -1, dims=1)
        targets[:, -1] = 0  # Pad the last position
        
        # Forward pass
        linguistic_features = self(phoneme_ids, midi_notes)
        
        # Reconstruction task
        logits = self.decoder(linguistic_features)
        
        # Reshape for loss calculation
        B, T, C = logits.shape
        logits = logits.reshape(B * T, C)
        targets = targets.reshape(B * T)
        
        # Calculate loss - ignore padded positions
        mask = (phoneme_ids != 0).reshape(B * T)
        loss = self.criterion(logits[mask], targets[mask])
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step
        
        Args:
            batch: Dictionary containing:
                - phoneme_ids: [B, T] - Batch of phoneme sequences
                - midi_notes: [B, T] - Optional batch of MIDI note sequences (if available)
            batch_idx: Index of the batch
            
        Returns:
            loss: Validation loss
        """
        # Get data from batch
        phoneme_ids = batch['phoneme_ids']
        midi_notes = batch.get('midi_notes', None)
        
        # Create targets for reconstruction (shifted right)
        targets = torch.roll(phoneme_ids, -1, dims=1)
        targets[:, -1] = 0  # Pad the last position
        
        # Forward pass
        linguistic_features = self(phoneme_ids, midi_notes)
        
        # Reconstruction task
        logits = self.decoder(linguistic_features)
        
        # Reshape for loss calculation
        B, T, C = logits.shape
        logits = logits.reshape(B * T, C)
        targets = targets.reshape(B * T)
        
        # Calculate loss - ignore padded positions
        mask = (phoneme_ids != 0).reshape(B * T)
        loss = self.criterion(logits[mask], targets[mask])
        
        # Log validation loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        # Log feature visualizations on the first validation batch only
        if batch_idx == 0:
            self._log_feature_visualizations(linguistic_features, phoneme_ids)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizers for training
        
        Returns:
            optimizer: PyTorch optimizer
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['model']['learning_rate'],
            weight_decay=self.config['model']['weight_decay']
        )
        
        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.config['model']['lr_decay']
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch'
            }
        }
    
    def _log_feature_visualizations(self, features, phoneme_ids):
        """
        Log feature visualizations to TensorBoard
        
        Args:
            features: [B, T, H] - Batch of linguistic features
            phoneme_ids: [B, T] - Batch of phoneme sequences
        """
        # Take the first sample from the batch
        feature_sample = features[0].detach().cpu().numpy()
        phoneme_sample = phoneme_ids[0].detach().cpu().numpy()
        
        # Create feature heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            feature_sample.T,  # Transpose for better visualization
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        plt.colorbar(im, ax=ax)
        
        # Add titles and labels
        ax.set_title('Phoneme Encoder Features')
        ax.set_xlabel('Time')
        ax.set_ylabel('Feature Dimension')
        
        # Log to TensorBoard
        self.logger.experiment.add_figure('features/heatmap', fig, self.current_epoch)
        plt.close(fig)
        
        # Create feature activation plot for dimensions
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mean activation across feature dimensions
        mean_activations = np.mean(feature_sample, axis=1)
        ax.plot(mean_activations)
        
        # Add non-zero phoneme markers
        non_zero_indices = np.where(phoneme_sample > 0)[0]
        for idx in non_zero_indices:
            ax.axvline(x=idx, color='r', linestyle='--', alpha=0.3)
        
        # Add titles and labels
        ax.set_title('Mean Feature Activation')
        ax.set_xlabel('Time')
        ax.set_ylabel('Activation')
        
        # Log to TensorBoard
        self.logger.experiment.add_figure('features/mean_activation', fig, self.current_epoch)
        plt.close(fig)
        
        # Feature dimension correlation
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Calculate correlation matrix between feature dimensions
        corr_matrix = np.corrcoef(feature_sample.T)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        
        # Add titles
        ax.set_title('Feature Dimension Correlation')
        
        # Log to TensorBoard
        self.logger.experiment.add_figure('features/correlation', fig, self.current_epoch)
        plt.close(fig)
        
        # Principal component analysis for dimensionality reduction
        try:
            from sklearn.decomposition import PCA
            
            # Apply PCA to reduce to 2D for visualization
            pca = PCA(n_components=2)
            feature_2d = pca.fit_transform(feature_sample)
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(8, 8))
            scatter = ax.scatter(
                feature_2d[:, 0], 
                feature_2d[:, 1], 
                c=np.arange(len(feature_2d)),  # Color by time
                cmap='viridis'
            )
            plt.colorbar(scatter, ax=ax, label='Time')
            
            # Add titles and labels
            ax.set_title('PCA of Phoneme Features')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            
            # Log to TensorBoard
            self.logger.experiment.add_figure('features/pca', fig, self.current_epoch)
            plt.close(fig)
            
        except ImportError:
            print("scikit-learn not available, skipping PCA visualization")