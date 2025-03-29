import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import yaml
import librosa
import librosa.display
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # This must be done before importing pyplot
from matplotlib.patches import Rectangle
import argparse

def read_config(config_path="futurevox/config/default.yaml"):
    """Read the configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_phone_color(phone):
    """Get color for phoneme type."""
    vowels = ['iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'ao', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ey', 'ay', 'oy', 'aw', 'ow']
    nasals = ['m', 'n', 'ng', 'em', 'en', 'eng']
    fricatives = ['f', 'v', 'th', 'dh', 's', 'z', 'sh', 'zh', 'hh']
    stops = ['p', 'b', 't', 'd', 'k', 'g']
    affricates = ['ch', 'jh']
    liquids = ['l', 'r', 'el']
    glides = ['w', 'y']
    
    if phone in ['pau', 'sil', 'sp']:
        return '#999999'  # Silence/pause
    elif phone in vowels:
        return '#e74c3c'  # Vowels
    elif phone in nasals:
        return '#3498db'  # Nasals
    elif phone in fricatives:
        return '#2ecc71'  # Fricatives
    elif phone in stops:
        return '#f39c12'  # Stops
    elif phone in affricates:
        return '#9b59b6'  # Affricates
    elif phone in liquids:
        return '#1abc9c'  # Liquids
    elif phone in glides:
        return '#d35400'  # Glides
    else:
        return '#34495e'  # Others

def create_visualization(sample_id, h5_path, output_path, config):
    """Create visualization with mel spectrogram, F0, and phoneme alignment from the HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        # Get the sample group
        sample_group = f[sample_id]
        
        # Extract data
        mel_spec = sample_group['features']['mel_spectrogram'][:]
        f0_times = sample_group['features']['f0_times'][:]
        f0_values = sample_group['features']['f0_values'][:]
        
        # Get phoneme data
        phones_bytes = sample_group['phonemes']['phones'][:]
        phones = [p.decode('utf-8') for p in phones_bytes]
        start_times = sample_group['phonemes']['start_times'][:]
        end_times = sample_group['phonemes']['end_times'][:]
        durations = sample_group['phonemes']['durations'][:]
        
        # Recreate phoneme frames for visualization
        phoneme_frames = []
        for i, phone in enumerate(phones):
            phoneme_frames.append({
                'phone': phone,
                'start_time': start_times[i],
                'end_time': end_times[i],
                'duration': durations[i]
            })

    plt.figure(figsize=(14, 10), dpi=100)
    
    # First subplot: Mel spectrogram
    ax1 = plt.subplot(3, 1, 1)
    img = librosa.display.specshow(
        mel_spec, 
        x_axis='time', 
        y_axis='mel', 
        sr=config['audio']['sample_rate'], 
        hop_length=config['audio']['hop_length'],
        fmin=config['audio']['fmin'],
        fmax=config['audio']['fmax']
    )
    #plt.colorbar(img, format='%+2.0f dB')
    ax1.set_title('Mel Spectrogram')
    
    # Second subplot: F0 contour
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(f0_times, f0_values, 'r-', linewidth=1.5, alpha=0.8)
    ax2.set_title('F0 Contour')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlim(0, f0_times[-1] if len(f0_times) > 0 else 0)
    ax2.set_ylim(config['audio']['f0_min'], config['audio']['f0_max'])  # Use F0 range from config
    ax2.grid(True, alpha=0.3)
    
    # Third subplot: Phoneme alignment
    ax3 = plt.subplot(3, 1, 3)
    ax3.set_title('Phoneme Alignment')
    ax3.set_xlabel('Time (s)')
    ax3.set_xlim(0, f0_times[-1] if len(f0_times) > 0 else 0)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    
    # Add phoneme segments
    for i, p in enumerate(phoneme_frames):
        rect = Rectangle(
            (p['start_time'], 0), 
            p['duration'], 
            1, 
            facecolor=get_phone_color(p['phone']), 
            edgecolor='black', 
            alpha=0.7
        )
        ax3.add_patch(rect)
        
        # Add phoneme text
        text_x = p['start_time'] + p['duration'] / 2
        ax3.text(
            text_x, 
            0.5, 
            p['phone'], 
            horizontalalignment='center', 
            verticalalignment='center', 
            fontweight='bold',
            fontsize=9
        )
        
        # Add vertical alignment lines across all plots
        if i > 0:
            ax1.axvline(x=p['start_time'], color='gray', linestyle='--', alpha=0.4)
            ax2.axvline(x=p['start_time'], color='gray', linestyle='--', alpha=0.4)
    
    # Add phoneme duration table
    table_text = "Phoneme durations (seconds):\n"
    table_text += "------------------------\n"
    table_text += "Phone | Start  | End    | Duration\n"
    table_text += "------+--------+--------+----------\n"
    
    for p in phoneme_frames:
        table_text += f"{p['phone']:<6}| {p['start_time']:.3f} | {p['end_time']:.3f} | {p['duration']:.3f}\n"
    
    plt.figtext(0.5, 0.01, table_text, fontsize=7, family='monospace')
    
    # Add sample ID
    plt.figtext(0.02, 0.01, f"Sample ID: {sample_id}", fontsize=8)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    return plt.gcf()

def analyze_alignment(sample_id, h5_path):
    """Analyze alignment between F0, mel spectrogram, and phonemes."""
    with h5py.File(h5_path, 'r') as f:
        # Get the sample group
        sample_group = f[sample_id]
        
        # Extract data
        mel_spec = sample_group['features']['mel_spectrogram'][:]
        f0_times = sample_group['features']['f0_times'][:]
        f0_values = sample_group['features']['f0_values'][:]
        
        # Get phoneme data
        phones_bytes = sample_group['phonemes']['phones'][:]
        phones = [p.decode('utf-8') for p in phones_bytes]
        start_times = sample_group['phonemes']['start_times'][:]
        end_times = sample_group['phonemes']['end_times'][:]
        
        # Check alignment issues
        issues = []
        
        # 1. Check if F0 length matches mel spectrogram frames
        if len(f0_values) != mel_spec.shape[1]:
            issues.append(f"F0 length ({len(f0_values)}) doesn't match mel spectrogram frames ({mel_spec.shape[1]})")
        
        # 2. Check if phoneme end times exceed audio length
        if len(end_times) > 0 and end_times[-1] > f0_times[-1]:
            issues.append(f"Phoneme end time ({end_times[-1]:.3f}) exceeds audio duration ({f0_times[-1]:.3f})")
        
        # 3. Check for overlapping phonemes
        for i in range(1, len(start_times)):
            if start_times[i] < end_times[i-1]:
                issues.append(f"Phoneme overlap: '{phones[i-1]}' end ({end_times[i-1]:.3f}) > '{phones[i]}' start ({start_times[i]:.3f})")
        
        # 4. Check for gaps between phonemes
        for i in range(1, len(start_times)):
            gap = start_times[i] - end_times[i-1]
            if gap > 0.05:  # Gap larger than 50 ms
                issues.append(f"Gap between phonemes: {gap:.3f}s between '{phones[i-1]}' and '{phones[i]}'")
        
        return issues

def process_dataset(h5_path, output_dir, config, max_samples=None):
    """Process the dataset and create visualizations for samples."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_path, 'r') as f:
        # Get list of samples (excluding metadata)
        samples = [k for k in f.keys() if k != 'metadata']
        
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"Found {len(samples)} samples in the dataset")
        print(f"Processing {len(samples)} samples...")
        
        alignment_report = []
        
        # Process each sample
        for i, sample_id in enumerate(samples):
            print(f"Processing sample {i+1}/{len(samples)}: {sample_id}")
            
            # Analyze alignment
            issues = analyze_alignment(sample_id, h5_path)
            
            if issues:
                print(f"  Found {len(issues)} alignment issues:")
                for issue in issues:
                    print(f"  - {issue}")
                alignment_report.append((sample_id, issues))
            
            # Create visualization
            output_path = os.path.join(output_dir, f"{sample_id}_alignment.png")
            create_visualization(sample_id, h5_path, output_path, config)
        
        # Save alignment report
        if alignment_report:
            report_path = os.path.join(output_dir, "alignment_issues.txt")
            with open(report_path, 'w') as f:
                f.write(f"Alignment Issues Report\n")
                f.write(f"======================\n\n")
                f.write(f"Dataset: {h5_path}\n")
                f.write(f"Total samples analyzed: {len(samples)}\n")
                f.write(f"Samples with issues: {len(alignment_report)}\n\n")
                
                for sample_id, issues in alignment_report:
                    f.write(f"Sample: {sample_id}\n")
                    for issue in issues:
                        f.write(f"  - {issue}\n")
                    f.write("\n")
            
            print(f"Alignment report saved to {report_path}")
        else:
            print("No alignment issues found in the analyzed samples.")

def main():
    parser = argparse.ArgumentParser(description='Analyze alignment in binary dataset')
    parser.add_argument('--config', type=str, default="config/default.yaml", help='Path to config file')
    parser.add_argument('--output', type=str, default="output", help='Output directory for visualizations')
    parser.add_argument('--max_samples', type=int, default=5, help='Maximum number of samples to process')
    parser.add_argument('--sample_id', type=str, help='Process a specific sample ID only')
    args = parser.parse_args()
    
    print("Starting binary dataset alignment analysis...")
    
    # Read config file
    config = read_config(args.config)
    
    # Get dataset path from config
    h5_path = config['datasets']['data_set']
    
    # Check if we're processing a specific sample or the entire dataset
    if args.sample_id:
        print(f"Processing sample: {args.sample_id}")
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, f"{args.sample_id}_alignment.png")
        
        # Analyze alignment
        issues = analyze_alignment(args.sample_id, h5_path)
        
        if issues:
            print(f"Found {len(issues)} alignment issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Create visualization
        create_visualization(args.sample_id, h5_path, output_path, config)
    else:
        # Process the dataset
        process_dataset(h5_path, args.output, config, args.max_samples)
    
    print("Alignment analysis complete!")

if __name__ == '__main__':
    main()