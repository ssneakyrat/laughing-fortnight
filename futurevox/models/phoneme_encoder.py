import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual block with dilated convolutions for the phoneme encoder
    """
    def __init__(self, dim, kernel_size, dilation):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                dim, 
                dim, 
                kernel_size=kernel_size, 
                dilation=dilation, 
                padding='same'
            ),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(
                dim, 
                dim, 
                kernel_size=kernel_size, 
                dilation=1, 
                padding='same'
            ),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            [B, C, T]
        """
        residual = x
        x = self.conv_layer(x)
        x = x + residual
        x = self.relu(x)
        return x


class PhonemeEncoder(nn.Module):
    """
    Phoneme encoder for FutureVox+ model
    
    Takes phoneme IDs and optionally MIDI notes as input and 
    produces linguistic features.
    """
    def __init__(
            self, 
            num_phonemes, 
            embedding_dim=192, 
            hidden_dim=256,
            n_layers=3, 
            kernel_size=3, 
            dropout=0.1,
            dilation_rates=None
        ):
        super().__init__()
        
        if dilation_rates is None:
            dilation_rates = [1, 3, 9]
        
        # Embedding layer for phoneme IDs
        self.embedding = nn.Embedding(num_phonemes, embedding_dim)
        
        # Initial convolution to adjust dimensions
        self.prenet = nn.Sequential(
            nn.Conv1d(
                embedding_dim, 
                hidden_dim, 
                kernel_size=kernel_size, 
                padding='same'
            ),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Stack of ResBlocks
        self.res_blocks = nn.ModuleList([
            ResBlock(
                hidden_dim, 
                kernel_size, 
                dilation_rates[i % len(dilation_rates)]
            ) for i in range(n_layers)
        ])
        
        # Final dropout
        self.dropout = nn.Dropout(dropout)
        
        # Save dimensions for later use
        self.hidden_dim = hidden_dim

    def forward(self, phoneme_ids, midi_notes=None):
        """
        Args:
            phoneme_ids: [B, T] - Batch of phoneme sequences
            midi_notes: [B, T] - Optional batch of MIDI note sequences
            
        Returns:
            linguistic_features: [B, T, H] - Batch of linguistic features
        """
        # Get phoneme embeddings [B, T] -> [B, T, E]
        x = self.embedding(phoneme_ids)
        
        # Transpose for 1D convolution [B, T, E] -> [B, E, T]
        x = x.transpose(1, 2)
        
        # Apply prenet
        x = self.prenet(x)
        
        # Apply ResBlocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply final dropout
        x = self.dropout(x)
        
        # Transpose back for output [B, E, T] -> [B, T, E]
        x = x.transpose(1, 2)
        
        # If MIDI notes provided, we could condition on them here
        # This is a placeholder for future development
        if midi_notes is not None:
            # For now just print a warning that this isn't implemented yet
            if not hasattr(self, 'midi_warning_shown'):
                print("Warning: MIDI conditioning not yet implemented")
                self.midi_warning_shown = True
        
        return x
    
    def get_output_dim(self):
        """Return the output dimension of the encoder"""
        return self.hidden_dim