import torch
import torch.nn.functional as F
import numpy as np
import h5py
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils


class PhonemeDataset(Dataset):
    """
    Dataset for loading phoneme sequences and related features from HDF5 file
    """
    def __init__(self, h5_path, max_seq_len=None):
        """
        Initialize dataset
        
        Args:
            h5_path: Path to the HDF5 file containing data
            max_seq_len: Maximum sequence length to use
        """
        self.h5_path = h5_path
        self.max_seq_len = max_seq_len
        
        # Open HDF5 file
        with h5py.File(h5_path, 'r') as f:
            # Get metadata
            metadata = f['metadata']
            file_list_bytes = metadata['file_list'][:]
            self.file_list = [name.decode('utf-8') for name in file_list_bytes]
            
            # Get phoneme to ID mapping (if available)
            if 'phone_to_id' in metadata:
                phone_map_bytes = metadata['phone_to_id'][:]
                self.phone_to_id = {
                    name.decode('utf-8'): idx for idx, name in enumerate(phone_map_bytes)
                }
            else:
                # Create phoneme mapping from data
                unique_phones = set()
                for sample_id in self.file_list:
                    if sample_id in f:
                        sample = f[sample_id]
                        if 'phonemes' in sample and 'phones' in sample['phonemes']:
                            phones_bytes = sample['phonemes']['phones'][:]
                            phones = [p.decode('utf-8') for p in phones_bytes]
                            unique_phones.update(phones)
                
                # Create phoneme to ID mapping
                unique_phones = sorted(list(unique_phones))
                self.phone_to_id = {phone: idx + 1 for idx, phone in enumerate(unique_phones)}
                # Add padding token
                self.phone_to_id['<PAD>'] = 0
            
            # Invert mapping for ID to phoneme
            self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}
            
            # Store number of phonemes
            self.num_phonemes = len(self.phone_to_id)
            
            print(f"Loaded dataset with {len(self.file_list)} samples")
            print(f"Number of unique phonemes: {self.num_phonemes}")
    
    def __len__(self):
        """Return number of samples in the dataset"""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Get sample from dataset
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            sample: Dictionary containing:
                - phoneme_ids: [T] - Phoneme sequence as IDs
                - midi_notes: [T] - MIDI note sequence (if available)
                - f0_values: [T'] - F0 values (if available)
                - sample_id: Sample ID for reference
        """
        sample_id = self.file_list[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            # Check if sample exists
            if sample_id not in f:
                # Return empty sample if not found
                return {
                    'phoneme_ids': torch.zeros(1, dtype=torch.long),
                    'sample_id': sample_id
                }
            
            sample = f[sample_id]
            
            # Get phoneme data
            if 'phonemes' in sample and 'phones' in sample['phonemes']:
                phones_bytes = sample['phonemes']['phones'][:]
                phones = [p.decode('utf-8') for p in phones_bytes]
                
                # Convert phonemes to IDs
                phoneme_ids = np.array([self.phone_to_id.get(p, 0) for p in phones])
            else:
                phoneme_ids = np.array([0])  # Padding if no phonemes
            
            # Truncate if needed
            if self.max_seq_len is not None and len(phoneme_ids) > self.max_seq_len:
                phoneme_ids = phoneme_ids[:self.max_seq_len]
            
            # Initialize return dictionary
            sample_dict = {
                'phoneme_ids': torch.tensor(phoneme_ids, dtype=torch.long),
                'sample_id': sample_id
            }
            
            # Get MIDI data if available
            if 'midi' in sample and sample['midi'].attrs.get('num_notes', 0) > 0:
                midi_notes = sample['midi']['notes'][:]
                midi_start_frames = sample['midi']['start_frames'][:]
                midi_end_frames = sample['midi']['end_frames'][:]
                
                # Create sequence of MIDI notes aligned with phonemes
                # This is a simplification and might need improvement for real data
                midi_sequence = np.zeros_like(phoneme_ids)
                
                for note, start, end in zip(midi_notes, midi_start_frames, midi_end_frames):
                    # Find corresponding phoneme indices
                    start_idx = max(0, int(start))
                    end_idx = min(len(phoneme_ids), int(end))
                    
                    if start_idx < len(midi_sequence) and end_idx <= len(midi_sequence):
                        midi_sequence[start_idx:end_idx] = note
                
                sample_dict['midi_notes'] = torch.tensor(midi_sequence, dtype=torch.long)
            
            # Get F0 values if available
            if 'features' in sample and 'f0_values' in sample['features']:
                f0_values = sample['features']['f0_values'][:]
                
                # Transform to torch tensor
                sample_dict['f0_values'] = torch.tensor(f0_values, dtype=torch.float)
            
            return sample_dict


def collate_fn(batch):
    """
    Collate function for DataLoader
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        batch_dict: Dictionary containing batched data
    """
    # Check for empty samples and remove them
    batch = [sample for sample in batch if sample['phoneme_ids'].shape[0] > 0]
    
    if not batch:
        # Return empty batch if all samples were empty
        return {
            'phoneme_ids': torch.zeros((0, 0), dtype=torch.long),
            'seq_lengths': torch.zeros(0, dtype=torch.long),
            'sample_ids': []
        }
    
    # Get sequence lengths
    seq_lengths = torch.tensor([sample['phoneme_ids'].shape[0] for sample in batch])
    
    # Get sample IDs
    sample_ids = [sample['sample_id'] for sample in batch]
    
    # Pad phoneme sequences
    phoneme_ids = rnn_utils.pad_sequence(
        [sample['phoneme_ids'] for sample in batch],
        batch_first=True,
        padding_value=0
    )
    
    # Initialize batch dictionary
    batch_dict = {
        'phoneme_ids': phoneme_ids,
        'seq_lengths': seq_lengths,
        'sample_ids': sample_ids
    }
    
    # Pad MIDI sequences if available
    if 'midi_notes' in batch[0]:
        midi_notes = rnn_utils.pad_sequence(
            [sample['midi_notes'] for sample in batch if 'midi_notes' in sample],
            batch_first=True,
            padding_value=0
        )
        batch_dict['midi_notes'] = midi_notes
    
    # Pad F0 sequences if available
    if 'f0_values' in batch[0]:
        f0_values = rnn_utils.pad_sequence(
            [sample['f0_values'] for sample in batch if 'f0_values' in sample],
            batch_first=True,
            padding_value=0
        )
        batch_dict['f0_values'] = f0_values
    
    return batch_dict


def create_dataloader(config, split='train'):
    """
    Create DataLoader for training/validation
    
    Args:
        config: Configuration dictionary
        split: 'train' or 'valid'
        
    Returns:
        dataloader: PyTorch DataLoader
    """
    # Determine data directory based on split
    if split == 'train':
        data_dir = config['data']['train_data_dir']
    else:
        data_dir = config['data']['valid_data_dir']
    
    # Get data path
    h5_path = config['data']['h5_path']
    
    # Create dataset
    dataset = PhonemeDataset(
        h5_path=h5_path,
        max_seq_len=config['data']['max_seq_len']
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader