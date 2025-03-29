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
import math

def read_config(config_path="futurevox/config/default.yaml"):
    """Read the configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def hz_to_midi(frequency):
    """Convert frequency in Hz to MIDI note number."""
    if frequency <= 0:
        return 0  # Silence or undefined
    return 69 + 12 * math.log2(frequency / 440.0)

def midi_to_hz(midi_note):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))

def midi_to_note_name(midi_note):
    """Convert MIDI note number to note name (e.g., C4, A#3)."""
    if midi_note == 0:
        return "Rest"
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = notes[midi_note % 12]
    return f"{note}{octave}"

def is_vowel(phoneme):
    """Check if a phoneme is a vowel."""
    vowels = ['iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'ao', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ey', 'ay', 'oy', 'aw', 'ow']
    return phoneme in vowels

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

def analyze_midi_alignment(sample_id, h5_path, config):
    """
    Analyze alignment between MIDI notes, phonemes, and F0 contour.
    
    Args:
        sample_id: Sample ID in the HDF5 file
        h5_path: Path to the HDF5 file
        config: Configuration dictionary
        
    Returns:
        List of alignment issues
    """
    issues = []
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check if sample exists
            if sample_id not in f:
                return [f"Sample {sample_id} not found in dataset"]
                
            sample = f[sample_id]
            
            # Check if MIDI data exists
            if 'midi' not in sample:
                return [f"Sample {sample_id} does not contain MIDI data"]
                
            # Get MIDI data
            midi_group = sample['midi']
            num_midi_notes = midi_group.attrs.get('num_notes', 0)
            
            if num_midi_notes == 0:
                return [f"Sample {sample_id} has no MIDI notes"]
                
            # Get MIDI note data
            midi_notes = midi_group['notes'][:]
            midi_start_times = midi_group['start_times'][:]
            midi_end_times = midi_group['end_times'][:]
            midi_phones_bytes = midi_group['phones'][:]
            midi_phones = [p.decode('utf-8') for p in midi_phones_bytes]
            
            # Get F0 data
            f0_times = sample['features']['f0_times'][:]
            f0_values = sample['features']['f0_values'][:]
            
            # Get phoneme data
            phone_group = sample['phonemes']
            phones_bytes = phone_group['phones'][:]
            phones = [p.decode('utf-8') for p in phones_bytes]
            phone_start_times = phone_group['start_times'][:]
            phone_end_times = phone_group['end_times'][:]
            
            # Check 1: Make sure MIDI notes are within valid range
            for i, note in enumerate(midi_notes):
                if note < 0 or note > 127:
                    issues.append(f"MIDI note {i} has invalid value: {note}")
            
            # Check 2: Compare MIDI notes with F0 values
            for i, (note, start, end) in enumerate(zip(midi_notes, midi_start_times, midi_end_times)):
                # Find corresponding F0 frames
                start_idx = np.searchsorted(f0_times, start)
                end_idx = np.searchsorted(f0_times, end)
                
                if start_idx >= end_idx or start_idx >= len(f0_values) or end_idx > len(f0_values):
                    issues.append(f"MIDI note {i} ({midi_to_note_name(note)}) has invalid time range")
                    continue
                    
                # Get F0 values for this note duration
                note_f0 = f0_values[start_idx:end_idx]
                
                # Filter out NaN values (unvoiced frames)
                valid_f0 = note_f0[~np.isnan(note_f0)]
                
                if len(valid_f0) == 0:
                    issues.append(f"MIDI note {i} ({midi_to_note_name(note)}) has no valid F0 values")
                    continue
                
                # Calculate median F0 and convert to MIDI
                median_f0 = np.median(valid_f0)
                estimated_midi = hz_to_midi(median_f0)
                
                # Check if MIDI note is significantly different from F0
                midi_diff = abs(estimated_midi - note)
                if midi_diff > 2.0:  # Threshold of 2 semitones
                    issues.append(f"MIDI note {i} ({midi_to_note_name(note)}) differs from F0 by {midi_diff:.1f} semitones")
            
            # Check 3: Verify that MIDI notes align with phoneme boundaries
            for i, (start, end, phone) in enumerate(zip(midi_start_times, midi_end_times, midi_phones)):
                # Find the corresponding phoneme
                found_match = False
                for j, (p_start, p_end, p_phone) in enumerate(zip(phone_start_times, phone_end_times, phones)):
                    if abs(start - p_start) < 0.01 and abs(end - p_end) < 0.01 and phone == p_phone:
                        found_match = True
                        break
                
                if not found_match:
                    issues.append(f"MIDI note {i} ({phone}, {midi_to_note_name(midi_notes[i])}) does not align with any phoneme")
            
            # Check 4: Verify all vowels have MIDI notes
            for i, (p_start, p_end, p_phone) in enumerate(zip(phone_start_times, phone_end_times, phones)):
                if is_vowel(p_phone):
                    found_match = False
                    for start, end, phone in zip(midi_start_times, midi_end_times, midi_phones):
                        if abs(start - p_start) < 0.01 and phone == p_phone:
                            found_match = True
                            break
                    
                    if not found_match:
                        issues.append(f"Vowel phoneme {p_phone} at {p_start:.3f}s has no associated MIDI note")
            
            # Check 5: Look for overlapping MIDI notes
            if len(midi_start_times) > 1:
                for i in range(1, len(midi_start_times)):
                    if midi_start_times[i] < midi_end_times[i-1]:
                        issues.append(f"MIDI notes {i-1} and {i} overlap: {midi_phones[i-1]} ({midi_to_note_name(midi_notes[i-1])}) and {midi_phones[i]} ({midi_to_note_name(midi_notes[i])})")
            
            return issues
            
    except Exception as e:
        return [f"Error analyzing MIDI alignment: {str(e)}"]

def create_midi_alignment_visualization(sample_id, h5_path, output_path, config, show_issues=True):
    """
    Create visualization of MIDI notes alignment with F0 and phonemes.
    
    Args:
        sample_id: Sample ID in the HDF5 file
        h5_path: Path to the HDF5 file
        output_path: Path to save the visualization
        config: Configuration dictionary
        show_issues: Whether to highlight alignment issues in the visualization
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check if sample exists
            if sample_id not in f:
                print(f"Sample {sample_id} not found in dataset")
                return False
                
            sample = f[sample_id]
            
            # Check if MIDI data exists
            if 'midi' not in sample:
                print(f"Sample {sample_id} does not contain MIDI data")
                return False
                
            # Get MIDI data
            midi_group = sample['midi']
            num_midi_notes = midi_group.attrs.get('num_notes', 0)
            
            if num_midi_notes == 0:
                print(f"Sample {sample_id} has no MIDI notes")
                return False
                
            # Get MIDI note data
            midi_notes = midi_group['notes'][:]
            midi_start_times = midi_group['start_times'][:]
            midi_end_times = midi_group['end_times'][:]
            midi_durations = midi_group['durations'][:]
            midi_phones_bytes = midi_group['phones'][:]
            midi_phones = [p.decode('utf-8') for p in midi_phones_bytes]
            
            # Get F0 data
            f0_times = sample['features']['f0_times'][:]
            f0_values = sample['features']['f0_values'][:]
            
            # Get phoneme data
            phone_group = sample['phonemes']
            phones_bytes = phone_group['phones'][:]
            phones = [p.decode('utf-8') for p in phones_bytes]
            phone_start_times = phone_group['start_times'][:]
            phone_end_times = phone_group['end_times'][:]
            phone_durations = phone_group['durations'][:]
            
            # Analyze alignment issues if needed
            issues = []
            if show_issues:
                issues = analyze_midi_alignment(sample_id, h5_path, config)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False

    # Create the visualization
    plt.figure(figsize=(14, 12))
    
    # First subplot: F0 contour with MIDI notes overlay
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(f0_times, f0_values, 'b-', linewidth=1.5, alpha=0.7, label='F0')
    
    # Overlay MIDI notes as horizontal lines
    for i, (note, start, end) in enumerate(zip(midi_notes, midi_start_times, midi_end_times)):
        # Convert MIDI note to Hz for plotting
        note_freq = midi_to_hz(note)
        ax1.plot([start, end], [note_freq, note_freq], 
                'r-', linewidth=2.5, alpha=0.7)
        
        # Add note text
        note_name = midi_to_note_name(note)
        ax1.text(start, note_freq, f"{note_name}", fontsize=8, verticalalignment='bottom')
    
    ax1.set_title('F0 Contour with MIDI Notes')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_ylim(config['audio']['f0_min'], config['audio']['f0_max'])
    ax1.set_xlim(0, f0_times[-1] if len(f0_times) > 0 else 0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['F0 Contour', 'MIDI Notes'])
    
    # Second subplot: MIDI piano roll
    ax2 = plt.subplot(4, 1, 2)
    
    # Get min and max MIDI note for y-axis limits
    min_midi = max(min(midi_notes) - 2, 0)
    max_midi = min(max(midi_notes) + 2, 127)
    
    # Create piano roll display
    for i, (note, start, end, phone) in enumerate(zip(midi_notes, midi_start_times, midi_end_times, midi_phones)):
        is_vowel_phone = is_vowel(phone)
        color = 'green' if is_vowel_phone else 'blue'
        alpha = 0.8 if is_vowel_phone else 0.5
        
        # Draw note rectangle
        rect = Rectangle(
            (start, note - 0.4),
            end - start,
            0.8,
            color=color,
            alpha=alpha,
            edgecolor='black',
            linewidth=1
        )
        ax2.add_patch(rect)
        
        # Add note name and phoneme
        ax2.text(start + (end - start)/2, note + 0.5, 
                f"{phone}\n{midi_to_note_name(note)}", 
                fontsize=8, verticalalignment='bottom', horizontalalignment='center')
    
    ax2.set_title('MIDI Piano Roll')
    ax2.set_ylabel('MIDI Note')
    ax2.set_yticks(range(int(min_midi), int(max_midi)+1, 2))
    ax2.set_xlim(0, f0_times[-1] if len(f0_times) > 0 else 0)
    ax2.set_ylim(min_midi, max_midi)
    ax2.grid(True, alpha=0.3)
    
    # Third subplot: Phoneme alignment
    ax3 = plt.subplot(4, 1, 3)
    
    # Add phoneme segments
    for i, (start, duration, phone) in enumerate(zip(phone_start_times, phone_durations, phones)):
        # Check if this phoneme has a corresponding MIDI note
        has_midi = False
        for m_start, m_phone in zip(midi_start_times, midi_phones):
            if abs(start - m_start) < 0.01 and phone == m_phone:
                has_midi = True
                break
        
        # Set alpha based on whether it has MIDI note
        alpha = 0.8 if has_midi else 0.4
        
        rect = Rectangle(
            (start, 0), 
            duration, 
            1, 
            facecolor=get_phone_color(phone), 
            edgecolor='black', 
            alpha=alpha
        )
        ax3.add_patch(rect)
        
        # Add phoneme text
        ax3.text(
            start + duration/2, 
            0.5, 
            phone, 
            horizontalalignment='center', 
            verticalalignment='center', 
            fontweight='bold',
            fontsize=9
        )
    
    ax3.set_title('Phoneme Alignment')
    ax3.set_xlim(0, f0_times[-1] if len(f0_times) > 0 else 0)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    
    # Fourth subplot: Alignment issues (if any)
    ax4 = plt.subplot(4, 1, 4)
    ax4.axis('off')
    
    if not issues:
        ax4.text(0.5, 0.5, "No alignment issues detected", 
                horizontalalignment='center', fontsize=12, fontweight='bold')
    else:
        issue_text = "Alignment Issues:\n\n"
        for i, issue in enumerate(issues[:10]):  # Show up to 10 issues
            issue_text += f"{i+1}. {issue}\n"
            
        if len(issues) > 10:
            issue_text += f"\n...and {len(issues) - 10} more issues"
            
        ax4.text(0.05, 0.95, issue_text, verticalalignment='top', fontsize=10)
    
    # Add a detailed table at the bottom
    midi_table = "MIDI Note Data:\n"
    midi_table += "Index | Phoneme | MIDI Note | Note Name | Start Time | End Time | Duration\n"
    midi_table += "------+---------+-----------+----------+------------+----------+----------\n"
    
    for i, (note, phone, start, end, duration) in enumerate(zip(
            midi_notes, midi_phones, midi_start_times, midi_end_times, midi_durations)):
        note_name = midi_to_note_name(note)
        midi_table += f"{i:<6}| {phone:<7} | {note:<9} | {note_name:<8} | {start:.3f}s | {end:.3f}s | {duration:.3f}s\n"
    
    plt.figtext(0.5, 0.01, midi_table, fontsize=7, family='monospace', ha='center')
    
    # Add title and metadata
    plt.suptitle(f"MIDI Alignment Analysis - Sample: {sample_id}", fontsize=16)
    
    # Tight layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MIDI alignment visualization saved to {output_path}")
    return True

def process_dataset(h5_path, output_dir, config, max_samples=None):
    """Process the dataset and create MIDI alignment visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get list of samples (excluding metadata)
            samples = [k for k in f.keys() if k != 'metadata']
            
            if max_samples:
                samples = samples[:min(max_samples, len(samples))]
            
            print(f"Found {len(samples)} samples in the dataset")
            print(f"Processing up to {len(samples)} samples...")
            
            all_issues = {}
            processed_count = 0
            
            # Process each sample
            for i, sample_id in enumerate(samples):
                print(f"Processing sample {i+1}/{len(samples)}: {sample_id}")
                
                # Analyze MIDI alignment
                issues = analyze_midi_alignment(sample_id, h5_path, config)
                
                if issues:
                    print(f"  Found {len(issues)} alignment issues")
                    all_issues[sample_id] = issues
                
                # Create visualization
                output_path = os.path.join(output_dir, f"{sample_id}_midi_alignment.png")
                if create_midi_alignment_visualization(sample_id, h5_path, output_path, config):
                    processed_count += 1
            
            # Save alignment issues report
            if all_issues:
                report_path = os.path.join(output_dir, "midi_alignment_issues.txt")
                with open(report_path, 'w') as f:
                    f.write(f"MIDI Alignment Issues Report\n")
                    f.write(f"==========================\n\n")
                    f.write(f"Dataset: {h5_path}\n")
                    f.write(f"Total samples analyzed: {len(samples)}\n")
                    f.write(f"Samples with issues: {len(all_issues)}\n\n")
                    
                    for sample_id, issues in all_issues.items():
                        f.write(f"Sample: {sample_id}\n")
                        for issue in issues:
                            f.write(f"  - {issue}\n")
                        f.write("\n")
                
                print(f"Alignment report saved to {report_path}")
            else:
                print("No alignment issues found in the analyzed samples.")
            
            print(f"Successfully processed {processed_count} samples with MIDI data")
            
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Analyze MIDI alignment in binary dataset')
    parser.add_argument('--config', type=str, default="futurevox/config/default.yaml", 
                        help='Path to config file')
    parser.add_argument('--output', type=str, default="output/midi_alignment", 
                        help='Output directory for visualizations')
    parser.add_argument('--max_samples', type=int, default=5, 
                        help='Maximum number of samples to process')
    parser.add_argument('--sample_id', type=str, 
                        help='Process a specific sample ID only')
    args = parser.parse_args()
    
    print("Starting MIDI alignment analysis...")
    
    # Read config file
    config = read_config(args.config)
    
    # Get dataset path from config
    h5_path = config['datasets']['data_set']
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Check if we're processing a specific sample or multiple samples
    if args.sample_id:
        print(f"Processing single sample: {args.sample_id}")
        output_path = os.path.join(args.output, f"{args.sample_id}_midi_alignment.png")
        
        # Analyze alignment
        issues = analyze_midi_alignment(args.sample_id, h5_path, config)
        
        if issues:
            print(f"Found {len(issues)} alignment issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Create visualization
        create_midi_alignment_visualization(args.sample_id, h5_path, output_path, config)
    else:
        # Process multiple samples
        process_dataset(h5_path, args.output, config, args.max_samples)
    
    print("MIDI alignment analysis complete!")

if __name__ == '__main__':
    main()