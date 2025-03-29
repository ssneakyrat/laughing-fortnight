import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import librosa
import yaml
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import argparse

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

def estimate_midi_from_phoneme(f0_values, start_frame, end_frame, method='median'):
    """
    Estimate a single MIDI note from F0 values within a phoneme.
    
    Args:
        f0_values: Array of F0 values
        start_frame: Start frame of the phoneme
        end_frame: End frame of the phoneme
        method: Method to estimate the note ('median', 'mean', or 'mode')
        
    Returns:
        Estimated MIDI note number
    """
    # Make sure we have valid frame indices
    start_frame = max(0, min(start_frame, len(f0_values)-1))
    end_frame = max(start_frame + 1, min(end_frame, len(f0_values)))
    
    # Extract F0 values for the phoneme duration
    phoneme_f0 = f0_values[start_frame:end_frame]
    
    # Filter out NaN values (unvoiced frames)
    valid_f0 = phoneme_f0[~np.isnan(phoneme_f0)]
    
    if len(valid_f0) == 0:
        return 0  # Unvoiced phoneme
    
    # Estimate pitch
    if method == 'median':
        f0_estimate = np.median(valid_f0)
    elif method == 'mean':
        f0_estimate = np.mean(valid_f0)
    elif method == 'mode':
        # Simple mode implementation using histogram
        hist, bin_edges = np.histogram(valid_f0, bins=24)  # Bins for each quarter tone
        bin_idx = np.argmax(hist)
        f0_estimate = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to MIDI note
    midi_note = hz_to_midi(f0_estimate)
    
    # Round to nearest semitone
    midi_note_rounded = round(midi_note)
    
    return midi_note_rounded

def is_vowel(phoneme):
    """Check if a phoneme is a vowel."""
    vowels = ['iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'ao', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ey', 'ay', 'oy', 'aw', 'ow']
    return phoneme in vowels

def get_phone_color(phone):
    """Get color for phoneme type (borrowed from existing codebase)."""
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

def create_midi_visualization(sample_id, h5_path, output_dir, config, method='median'):
    """
    Create visualization of F0 contour with estimated MIDI notes.
    
    Args:
        sample_id: Sample ID in the HDF5 file
        h5_path: Path to the HDF5 file
        output_dir: Directory to save output files
        config: Configuration dictionary
        method: Method to estimate MIDI notes ('median', 'mean', or 'mode')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get the sample group
            if sample_id not in f:
                print(f"Sample {sample_id} not found in HDF5 file")
                return None
                
            sample_group = f[sample_id]
            
            # Extract data
            f0_times = sample_group['features']['f0_times'][:]
            f0_values = sample_group['features']['f0_values'][:]
            
            # Get phoneme data
            phones_bytes = sample_group['phonemes']['phones'][:]
            phones = [p.decode('utf-8') for p in phones_bytes]
            start_frames = sample_group['phonemes']['start_frames'][:]
            end_frames = sample_group['phonemes']['end_frames'][:]
            start_times = sample_group['phonemes']['start_times'][:]
            end_times = sample_group['phonemes']['end_times'][:]
            durations = sample_group['phonemes']['durations'][:]
    except Exception as e:
        print(f"Error reading HDF5 file: {str(e)}")
        return None
    
    # Create list to store MIDI note data
    midi_notes = []
    
    # Estimate MIDI notes for each phoneme
    for i, phone in enumerate(phones):
        start_frame = int(start_frames[i])
        end_frame = int(end_frames[i])
        
        # Check if phoneme contains enough frames
        if end_frame <= start_frame:
            continue
            
        # Make sure end frame is within bounds
        end_frame = min(end_frame, len(f0_values))
        
        # Estimate MIDI note
        midi_note = estimate_midi_from_phoneme(f0_values, start_frame, end_frame, method)
        
        # Only store notes with valid MIDI values (skip unvoiced phonemes)
        if midi_note > 0:
            # Store note data
            note_data = {
                'phone': phone,
                'start_time': start_times[i],
                'end_time': end_times[i],
                'duration': durations[i],
                'midi_note': midi_note,
                'is_vowel': is_vowel(phone)
            }
            midi_notes.append(note_data)
    
    # Post-process the MIDI notes to ensure no overlaps
    if midi_notes:
        # Sort by start time to ensure sequential processing
        midi_notes.sort(key=lambda x: x['start_time'])
        
        # Adjust end times to match start times of the next note
        for i in range(len(midi_notes) - 1):
            current_note = midi_notes[i]
            next_note = midi_notes[i + 1]
            
            # If there's overlap or gap, adjust the current note's end time
            if current_note['end_time'] != next_note['start_time']:
                current_note['end_time'] = next_note['start_time']
                current_note['duration'] = current_note['end_time'] - current_note['start_time']
    
    # Create the visualization
    plt.figure(figsize=(14, 10))
    
    # First subplot: F0 contour with MIDI note overlay
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(f0_times, f0_values, 'b-', linewidth=1.5, alpha=0.7, label='F0')
    
    # Overlay MIDI notes as horizontal lines
    for note in midi_notes:
        # Convert MIDI note to Hz for plotting on same axis
        note_freq = midi_to_hz(note['midi_note'])
        ax1.plot([note['start_time'], note['end_time']], [note_freq, note_freq], 
                'r-', linewidth=2.5, alpha=0.7)
    
    ax1.set_title('F0 Contour with Estimated MIDI Notes')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_ylim(config['audio']['f0_min'], config['audio']['f0_max'])
    ax1.set_xlim(0, f0_times[-1] if len(f0_times) > 0 else 0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['F0 Contour', 'MIDI Notes'])
    
    # Second subplot: MIDI piano roll
    ax2 = plt.subplot(3, 1, 2)
    
    # Create piano roll display
    for i, note in enumerate(midi_notes):
        # Use different colors for vowels and consonants
        color = 'green' if note['is_vowel'] else 'blue'
        alpha = 0.8 if note['is_vowel'] else 0.4
        
        # Create rectangle for note (better for non-overlapping visualization)
        rect = Rectangle(
            (note['start_time'], note['midi_note'] - 0.4),
            note['duration'],
            0.8,
            color=color,
            alpha=alpha,
            edgecolor='black',
            linewidth=1
        )
        ax2.add_patch(rect)
        
        # Add note name text
        ax2.text(note['start_time'] + note['duration']/2, note['midi_note'] + 0.5, 
                f"{note['phone']}\n{midi_to_note_name(note['midi_note'])}", 
                fontsize=8, verticalalignment='bottom', 
                horizontalalignment='center')
    
    ax2.set_title('MIDI Notes from Phonemes (Piano Roll)')
    ax2.set_ylabel('MIDI Note Number')
    ax2.set_xlim(0, f0_times[-1] if len(f0_times) > 0 else 0)
    
    # Set y-axis limits to show a reasonable pitch range
    midi_values = [note['midi_note'] for note in midi_notes if note['midi_note'] > 0]
    if midi_values:
        min_midi = max(min(midi_values) - 5, 0)
        max_midi = min(max(midi_values) + 5, 127)
        ax2.set_ylim(min_midi, max_midi)
    
    ax2.grid(True, alpha=0.3)
    
    # Add legend for piano roll
    vowel_patch = mpatches.Patch(color='green', alpha=0.8, label='Vowels')
    consonant_patch = mpatches.Patch(color='blue', alpha=0.4, label='Consonants')
    ax2.legend(handles=[vowel_patch, consonant_patch])
    
    # Third subplot: Phoneme alignment (similar to existing visualization)
    ax3 = plt.subplot(3, 1, 3)
    ax3.set_title('Phoneme Alignment with MIDI Notes')
    ax3.set_xlabel('Time (s)')
    ax3.set_xlim(0, f0_times[-1] if len(f0_times) > 0 else 0)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    
    # Get all phonemes for complete timeline display
    all_phonemes = []
    for i, phone in enumerate(phones):
        phoneme_data = {
            'phone': phone,
            'start_time': start_times[i],
            'end_time': end_times[i],
            'duration': durations[i],
            'has_midi': False
        }
        
        # Check if this phoneme has an associated MIDI note
        for note in midi_notes:
            if abs(note['start_time'] - phoneme_data['start_time']) < 0.001:
                phoneme_data['has_midi'] = True
                phoneme_data['midi_note'] = note['midi_note']
                break
                
        all_phonemes.append(phoneme_data)
    
    # Add phoneme segments
    for i, p in enumerate(all_phonemes):
        # Determine color intensity based on whether it has a MIDI note
        alpha = 0.7 if p.get('has_midi', False) else 0.3
        
        rect = Rectangle(
            (p['start_time'], 0), 
            p['duration'], 
            1, 
            facecolor=get_phone_color(p['phone']), 
            edgecolor='black', 
            alpha=alpha
        )
        ax3.add_patch(rect)
        
        # Add phoneme text with MIDI note if available
        text_x = p['start_time'] + p['duration'] / 2
        if p.get('has_midi', False):
            note_text = f"{p['phone']}\n{midi_to_note_name(p['midi_note'])}"
        else:
            note_text = p['phone']
            
        ax3.text(
            text_x, 
            0.5, 
            note_text, 
            horizontalalignment='center', 
            verticalalignment='center', 
            fontweight='bold',
            fontsize=8
        )
    
    # Add phoneme-to-MIDI mapping table
    table_text = "Phoneme to MIDI Note Mapping:\n"
    table_text += "Phone | Start Time | End Time | Duration | MIDI Note | Note Name\n"
    table_text += "------+------------+----------+----------+-----------+----------\n"
    
    # First show phonemes with MIDI notes
    for note in midi_notes:
        note_name = midi_to_note_name(note['midi_note'])
        table_text += f"{note['phone']:<6}| {note['start_time']:.3f}s | {note['end_time']:.3f}s | {note['duration']:.3f}s | {note['midi_note']:<9} | {note_name}\n"
    
    # Then list skipped phonemes
    table_text += "\nSkipped Phonemes (No MIDI Note):\n"
    table_text += "------+------------+----------+----------\n"
    
    # Find phonemes not in midi_notes
    skipped_phonemes = []
    for i, phone in enumerate(phones):
        found = False
        for note in midi_notes:
            if abs(note['start_time'] - start_times[i]) < 0.001:
                found = True
                break
        
        if not found:
            skipped_phonemes.append({
                'phone': phone,
                'start_time': start_times[i],
                'end_time': end_times[i],
                'duration': durations[i]
            })
    
    for p in skipped_phonemes:
        table_text += f"{p['phone']:<6}| {p['start_time']:.3f}s | {p['end_time']:.3f}s | {p['duration']:.3f}s\n"
    
    plt.figtext(0.5, 0.01, table_text, fontsize=7, family='monospace', ha='center')
    
    # Save the visualization
    output_path = os.path.join(output_dir, f"{sample_id}_midi_estimation.png")
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    
    # Return the MIDI notes data for further use
    return midi_notes

def save_midi_file(midi_notes, sample_id, output_dir):
    """
    Save the estimated MIDI notes to a MIDI file.
    
    Args:
        midi_notes: List of dictionaries containing MIDI note data
        sample_id: Sample ID for the filename
        output_dir: Directory to save the MIDI file
    """
    try:
        import mido
        
        # Create a new MIDI file with one track
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo (BPM)
        tempo = 120
        ms_per_beat = 60000 / tempo
        track.append(mido.MetaMessage('set_tempo', tempo=int(ms_per_beat * 1000), time=0))
        
        # Set time signature (4/4)
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        
        # Track name
        track.append(mido.MetaMessage('track_name', name=f"{sample_id} - Estimated Melody", time=0))
        
        # Convert note timings to ticks (assuming 480 ticks per quarter note)
        ticks_per_quarter = 480
        current_time = 0
        current_ticks = 0
        
        # Add note events to the track
        for note in midi_notes:
            if note['midi_note'] <= 0:
                continue  # Skip unvoiced notes
                
            # Calculate note start time in ticks
            start_ticks = int(note['start_time'] * 1000 * ticks_per_quarter / ms_per_beat)
            # Calculate note duration in ticks
            duration_ticks = int(note['duration'] * 1000 * ticks_per_quarter / ms_per_beat)
            
            # Calculate delta time (time since last event)
            delta_ticks = start_ticks - current_ticks
            
            # Add note on event
            track.append(mido.Message('note_on', note=int(note['midi_note']), 
                                     velocity=80 if note['is_vowel'] else 60, 
                                     time=max(0, delta_ticks)))
            
            # Add note off event
            track.append(mido.Message('note_off', note=int(note['midi_note']), 
                                     velocity=0, time=duration_ticks))
            
            # Update current time and ticks
            current_ticks = start_ticks + duration_ticks
        
        # Save the MIDI file
        midi_path = os.path.join(output_dir, f"{sample_id}_estimated.mid")
        mid.save(midi_path)
        print(f"MIDI file saved to {midi_path}")
        return midi_path
        
    except ImportError:
        print("Warning: mido library not found. MIDI file export skipped.")
        print("To export MIDI files, install mido with: pip install mido")
        return None
    except Exception as e:
        print(f"Error saving MIDI file: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate MIDI from phonemes and F0')
    parser.add_argument('--config', type=str, default="futurevox/config/default.yaml", 
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default="output/midi", 
                        help='Output directory for visualizations')
    parser.add_argument('--sample_id', type=str, 
                        help='Process a specific sample ID only')
    parser.add_argument('--method', type=str, default='median', 
                        choices=['median', 'mean', 'mode'],
                        help='Method to estimate MIDI notes')
    parser.add_argument('--max_samples', type=int, default=5, 
                        help='Maximum number of samples to process')
    parser.add_argument('--export_midi', action='store_true',
                        help='Export MIDI files')
    args = parser.parse_args()
    
    print("Starting MIDI estimation from phonemes and F0...")
    
    # Read config file
    config = read_config(args.config)
    
    # Get dataset path from config
    h5_path = config['datasets']['data_set']
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Check if we're processing a specific sample or multiple samples
    if args.sample_id:
        midi_notes = create_midi_visualization(args.sample_id, h5_path, args.output, config, args.method)
        if midi_notes and args.export_midi:
            save_midi_file(midi_notes, args.sample_id, args.output)
    else:
        # Process multiple samples
        try:
            with h5py.File(h5_path, 'r') as f:
                # Get list of samples (excluding metadata)
                samples = [k for k in f.keys() if k != 'metadata']
                
                if args.max_samples:
                    samples = samples[:min(args.max_samples, len(samples))]
                
                print(f"Processing {len(samples)} samples...")
                
                for i, sample_id in enumerate(samples):
                    print(f"Processing sample {i+1}/{len(samples)}: {sample_id}")
                    midi_notes = create_midi_visualization(sample_id, h5_path, args.output, config, args.method)
                    if midi_notes and args.export_midi:
                        save_midi_file(midi_notes, sample_id, args.output)
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
    
    print("MIDI estimation complete!")

if __name__ == '__main__':
    main()