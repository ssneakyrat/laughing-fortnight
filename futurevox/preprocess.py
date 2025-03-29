import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import h5py
import yaml
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # This must be done before importing pyplot
from matplotlib.patches import Rectangle

def read_config(config_path="futurevox/config/default.yaml"):
    """Read the configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_lab_file(lab_path):
    """Read a label file and return the phoneme segments."""
    phonemes = []
    with open(lab_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, phone = int(parts[0]), int(parts[1]), parts[2]
                phonemes.append({'start': start, 'end': end, 'phone': phone})
    return phonemes

def extract_f0(audio, sample_rate, min_f0=70, max_f0=400, frame_length=1024, hop_length=256):
    """
    Extract fundamental frequency (F0) using librosa's pyin algorithm.
    Returns times and f0 values.
    """
    # Use PYIN algorithm for F0 extraction
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=min_f0,
        fmax=max_f0,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # Get corresponding time values
    times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length)
    
    return times, f0

def extract_mel_spectrogram(audio, config):
    """Extract mel spectrogram based on config parameters."""
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=config['audio']['sample_rate'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        n_mels=config['audio']['n_mels'],
        fmin=config['audio']['fmin'],
        fmax=config['audio']['fmax']
    )
    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

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

def is_vowel(phoneme):
    """Check if a phoneme is a vowel."""
    vowels = ['iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'ao', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ey', 'ay', 'oy', 'aw', 'ow']
    return phoneme in vowels

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

def phonemes_to_frames(phonemes, lab_sample_rate, hop_length, sample_rate, scaling_factor=227.13):
    """Convert phoneme timings to frame indices for the mel spectrogram."""
    phoneme_frames = []
    for p in phonemes:
        # Convert from lab time units to seconds
        start_time = p['start'] / lab_sample_rate / scaling_factor
        end_time = p['end'] / lab_sample_rate / scaling_factor
        
        # Convert from seconds to frame indices
        start_frame = int(start_time * sample_rate / hop_length)
        end_frame = int(end_time * sample_rate / hop_length)
        
        duration = end_time - start_time
        
        phoneme_frames.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'phone': p['phone']
        })
    return phoneme_frames

def extract_midi_notes(f0_values, phoneme_frames, method='median'):
    """
    Extract MIDI notes from F0 values based on phoneme segmentation.
    
    Args:
        f0_values: Array of F0 values
        phoneme_frames: List of phoneme frame information
        method: Method to estimate notes ('median', 'mean', or 'mode')
        
    Returns:
        List of dictionaries containing MIDI note information
    """
    midi_notes = []
    
    for p in phoneme_frames:
        # Estimate MIDI note
        midi_note = estimate_midi_from_phoneme(
            f0_values, 
            p['start_frame'], 
            p['end_frame'], 
            method
        )
        
        # Only include valid notes
        if midi_note > 0:
            note_data = {
                'phone': p['phone'],
                'start_time': p['start_time'],
                'end_time': p['end_time'],
                'duration': p['duration'],
                'start_frame': p['start_frame'],
                'end_frame': p['end_frame'],
                'midi_note': midi_note,
                'is_vowel': is_vowel(p['phone'])
            }
            midi_notes.append(note_data)
    
    # Sort by start time to ensure sequential processing
    if midi_notes:
        midi_notes.sort(key=lambda x: x['start_time'])
        
        # Adjust end times to prevent overlaps
        for i in range(len(midi_notes) - 1):
            current_note = midi_notes[i]
            next_note = midi_notes[i + 1]
            
            # If there's overlap, adjust the current note's end time
            if current_note['end_time'] != next_note['start_time']:
                current_note['end_time'] = next_note['start_time']
                current_note['duration'] = current_note['end_time'] - current_note['start_time']
    
    return midi_notes

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
        
        # Get MIDI data if available
        has_midi = 'midi' in sample_group
        if has_midi:
            midi_notes = []
            midi_notes_data = sample_group['midi']['notes'][:]
            midi_start_times = sample_group['midi']['start_times'][:]
            midi_end_times = sample_group['midi']['end_times'][:]
            midi_durations = sample_group['midi']['durations'][:]
            midi_phones_bytes = sample_group['midi']['phones'][:]
            midi_phones = [p.decode('utf-8') for p in midi_phones_bytes]
            
            for i in range(len(midi_notes_data)):
                midi_notes.append({
                    'midi_note': midi_notes_data[i],
                    'start_time': midi_start_times[i],
                    'end_time': midi_end_times[i],
                    'duration': midi_durations[i],
                    'phone': midi_phones[i]
                })
        
        # Recreate phoneme frames for visualization
        phoneme_frames = []
        for i, phone in enumerate(phones):
            phoneme_data = {
                'phone': phone,
                'start_time': start_times[i],
                'end_time': end_times[i],
                'duration': durations[i],
                'has_midi': False
            }
            
            # Check if this phoneme has a corresponding MIDI note
            if has_midi:
                for note in midi_notes:
                    if abs(note['start_time'] - phoneme_data['start_time']) < 0.001:
                        phoneme_data['has_midi'] = True
                        phoneme_data['midi_note'] = note['midi_note']
                        break
                    
            phoneme_frames.append(phoneme_data)

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
    
    # Second subplot: F0 contour with MIDI notes if available
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(f0_times, f0_values, 'r-', linewidth=1.5, alpha=0.8)
    
    # Overlay MIDI notes if available
    if has_midi:
        for note in midi_notes:
            # Convert MIDI note to Hz for plotting
            note_freq = midi_to_hz(note['midi_note'])
            ax2.plot(
                [note['start_time'], note['end_time']], 
                [note_freq, note_freq], 
                'b-', linewidth=2, alpha=0.7
            )
        ax2.legend(['F0 Contour', 'MIDI Notes'])
    
    ax2.set_title('F0 Contour' + (' with MIDI Notes' if has_midi else ''))
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlim(0, f0_times[-1] if len(f0_times) > 0 else 0)
    ax2.set_ylim(config['audio']['f0_min'], config['audio']['f0_max'])
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
        # Set alpha based on whether it has MIDI note
        alpha = 0.8 if p.get('has_midi', False) else 0.5
        
        rect = Rectangle(
            (p['start_time'], 0), 
            p['duration'], 
            1, 
            facecolor=get_phone_color(p['phone']), 
            edgecolor='black', 
            alpha=alpha
        )
        ax3.add_patch(rect)
        
        # Add phoneme text
        text_x = p['start_time'] + p['duration'] / 2
        
        # Add MIDI note if available
        if p.get('has_midi', False):
            text = f"{p['phone']}\n{midi_to_note_name(p['midi_note'])}"
        else:
            text = p['phone']
            
        ax3.text(
            text_x, 
            0.5, 
            text, 
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
    table_text += "Phone | Start  | End    | Duration"
    if has_midi:
        table_text += " | MIDI Note"
    table_text += "\n"
    table_text += "------+--------+--------+----------"
    if has_midi:
        table_text += "+----------"
    table_text += "\n"
    
    for p in phoneme_frames:
        line = f"{p['phone']:<6}| {p['start_time']:.3f} | {p['end_time']:.3f} | {p['duration']:.3f}"
        if has_midi and p.get('has_midi', False):
            line += f" | {p['midi_note']} ({midi_to_note_name(p['midi_note'])})"
        elif has_midi:
            line += " | -"
        line += "\n"
        table_text += line
    
    plt.figtext(0.5, 0.01, table_text, fontsize=7, family='monospace')
    
    # Add sample ID
    plt.figtext(0.02, 0.01, f"Sample ID: {sample_id}" + (" (with MIDI)" if has_midi else ""), fontsize=8)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    return plt.gcf()

def process_files(config_path="futurevox/config/default.yaml", lab_sample_rate=44100, scaling_factor=227.13, midi_method='median'):
    """Process WAV and LAB files, extract features, and save to a single HDF5 file."""
    # Read config
    config = read_config(config_path)
    
    # Get paths
    data_raw_path = config['datasets']['data_raw']
    wav_dir = os.path.join(data_raw_path, "wav")
    lab_dir = os.path.join(data_raw_path, "lab")
    binary_dir = os.path.join(data_raw_path, "binary")
    
    # Create binary directory if it doesn't exist
    os.makedirs(binary_dir, exist_ok=True)
    
    # Define the path for the single HDF5 file
    h5_path = os.path.join(binary_dir, "dataset.h5")
    
    # Get list of WAV files
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    
    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return None
    
    # Create the HDF5 file
    with h5py.File(h5_path, 'w') as f:
        # Store metadata and configuration
        metadata_group = f.create_group('metadata')
        for key, value in config['audio'].items():
            metadata_group.attrs[key] = value
        metadata_group.attrs['lab_sample_rate'] = lab_sample_rate
        metadata_group.attrs['scaling_factor'] = scaling_factor
        metadata_group.attrs['midi_method'] = midi_method
        
        # Store file list
        file_list = np.array([os.path.splitext(wav)[0] for wav in wav_files], dtype='S100')
        metadata_group.create_dataset('file_list', data=file_list)
        
        # Process each WAV file
        sample_id_for_visualization = None
        
        for i, wav_file in enumerate(wav_files):
            # Get corresponding lab file
            base_name = os.path.splitext(wav_file)[0]
            lab_file = base_name + '.lab'
            lab_path = os.path.join(lab_dir, lab_file)
            
            # Skip if lab file doesn't exist
            if not os.path.exists(lab_path):
                print(f"Warning: No matching lab file for {wav_file}")
                continue
                
            wav_path = os.path.join(wav_dir, wav_file)
            
            try:
                print(f"Processing {wav_file} ({i+1}/{len(wav_files)})")
                
                # Store the first valid sample ID for visualization
                if sample_id_for_visualization is None:
                    sample_id_for_visualization = base_name
                
                # Create a group for this sample
                sample_group = f.create_group(base_name)
                
                # Load audio
                audio, sample_rate = librosa.load(wav_path, sr=config['audio']['sample_rate'])
                
                # Extract features
                mel_spec = extract_mel_spectrogram(audio, config)
                f0_times, f0_values = extract_f0(
                    audio, 
                    sample_rate, 
                    min_f0=config['audio']['f0_min'],
                    max_f0=config['audio']['f0_max'],
                    frame_length=config['audio']['n_fft'], 
                    hop_length=config['audio']['hop_length']
                )
                
                # Load phonemes
                phonemes = read_lab_file(lab_path)
                
                # Convert phoneme timings to frame indices
                phoneme_frames = phonemes_to_frames(
                    phonemes, 
                    lab_sample_rate, 
                    config['audio']['hop_length'],
                    sample_rate,
                    scaling_factor
                )
                
                # Extract MIDI notes from phonemes and F0
                midi_notes = extract_midi_notes(f0_values, phoneme_frames, midi_method)
                
                # Create subgroups
                audio_group = sample_group.create_group('audio')
                feature_group = sample_group.create_group('features')
                phoneme_group = sample_group.create_group('phonemes')
                midi_group = sample_group.create_group('midi')
                
                # Store audio data
                audio_group.create_dataset('waveform', data=audio)
                
                # Store features
                feature_group.create_dataset('mel_spectrogram', data=mel_spec)
                feature_group.create_dataset('f0_times', data=f0_times)
                feature_group.create_dataset('f0_values', data=f0_values)
                
                # Store phoneme data
                phones = np.array([p['phone'] for p in phoneme_frames], dtype='S10')
                start_frames = np.array([p['start_frame'] for p in phoneme_frames])
                end_frames = np.array([p['end_frame'] for p in phoneme_frames])
                start_times = np.array([p['start_time'] for p in phoneme_frames])
                end_times = np.array([p['end_time'] for p in phoneme_frames])
                durations = np.array([p['duration'] for p in phoneme_frames])
                
                phoneme_group.create_dataset('phones', data=phones)
                phoneme_group.create_dataset('start_frames', data=start_frames)
                phoneme_group.create_dataset('end_frames', data=end_frames)
                phoneme_group.create_dataset('start_times', data=start_times)
                phoneme_group.create_dataset('end_times', data=end_times)
                phoneme_group.create_dataset('durations', data=durations)
                
                # Store MIDI data
                if midi_notes:
                    midi_notes_data = np.array([note['midi_note'] for note in midi_notes])
                    midi_start_times = np.array([note['start_time'] for note in midi_notes])
                    midi_end_times = np.array([note['end_time'] for note in midi_notes])
                    midi_durations = np.array([note['duration'] for note in midi_notes])
                    midi_start_frames = np.array([note['start_frame'] for note in midi_notes])
                    midi_end_frames = np.array([note['end_frame'] for note in midi_notes])
                    midi_phones = np.array([note['phone'] for note in midi_notes], dtype='S10')
                    midi_is_vowel = np.array([1 if note['is_vowel'] else 0 for note in midi_notes], dtype=bool)
                    
                    midi_group.create_dataset('notes', data=midi_notes_data)
                    midi_group.create_dataset('start_times', data=midi_start_times)
                    midi_group.create_dataset('end_times', data=midi_end_times)
                    midi_group.create_dataset('durations', data=midi_durations)
                    midi_group.create_dataset('start_frames', data=midi_start_frames)
                    midi_group.create_dataset('end_frames', data=midi_end_frames)
                    midi_group.create_dataset('phones', data=midi_phones)
                    midi_group.create_dataset('is_vowel', data=midi_is_vowel)
                    
                    # Store number of MIDI notes
                    midi_group.attrs['num_notes'] = len(midi_notes)
                    
                    # Also store note names for easy reference
                    note_names = [midi_to_note_name(note) for note in midi_notes_data]
                    note_names_bytes = np.array([name.encode('utf-8') for name in note_names], dtype='S10')
                    midi_group.create_dataset('note_names', data=note_names_bytes)
                else:
                    # Create empty datasets for consistency
                    midi_group.create_dataset('notes', data=np.array([]))
                    midi_group.create_dataset('start_times', data=np.array([]))
                    midi_group.create_dataset('end_times', data=np.array([]))
                    midi_group.create_dataset('durations', data=np.array([]))
                    midi_group.create_dataset('phones', data=np.array([], dtype='S10'))
                    midi_group.attrs['num_notes'] = 0
                
                # Store sample metadata
                sample_group.attrs['filename'] = wav_file
                sample_group.attrs['duration'] = len(audio) / sample_rate
                sample_group.attrs['num_frames'] = mel_spec.shape[1]
                sample_group.attrs['num_phonemes'] = len(phones)
                sample_group.attrs['num_midi_notes'] = len(midi_notes) if midi_notes else 0
                
            except Exception as e:
                print(f"Error processing {wav_file}: {str(e)}")
    
    print(f"All samples processed and saved to {h5_path}")
    return h5_path, sample_id_for_visualization

def validate_dataset(h5_path):
    """Validate the entire dataset by checking sample integrity."""
    print(f"Validating dataset: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # Get dataset metadata
        metadata = f['metadata']
        file_list_bytes = metadata['file_list'][:]
        file_list = [name.decode('utf-8') for name in file_list_bytes]
        
        print(f"\nFound {len(file_list)} samples in the dataset")
        print(f"Audio configuration: {dict(metadata.attrs.items())}")
        
        # Validate each sample
        valid_samples = 0
        issues = 0
        
        print("\nSample validation:")
        for i, sample_id in enumerate(file_list):
            if sample_id not in f:
                print(f"  - Warning: Sample '{sample_id}' in file list but not in dataset")
                issues += 1
                continue
            
            sample = f[sample_id]
            
            # Check required groups and datasets
            required_groups = ['audio', 'features', 'phonemes', 'midi']
            missing_groups = [g for g in required_groups if g not in sample]
            
            if missing_groups:
                print(f"  - Warning: Sample '{sample_id}' missing groups: {missing_groups}")
                issues += 1
                continue
            
            # Check mel spectrogram and F0 alignment
            mel_spec = sample['features']['mel_spectrogram'][:]
            f0_values = sample['features']['f0_values'][:]
            
            if len(f0_values) != mel_spec.shape[1]:
                print(f"  - Warning: Sample '{sample_id}' has misaligned F0 ({len(f0_values)}) and mel spectrogram ({mel_spec.shape[1]})")
                issues += 1
            
            # Check phoneme frame ranges
            phoneme_end_frames = sample['phonemes']['end_frames'][:]
            if len(phoneme_end_frames) > 0 and max(phoneme_end_frames) > mel_spec.shape[1]:
                print(f"  - Warning: Sample '{sample_id}' has phoneme frames exceeding spectrogram length")
                issues += 1
            
            # Check MIDI data
            if sample['midi'].attrs['num_notes'] > 0:
                midi_notes = sample['midi']['notes'][:]
                midi_end_frames = sample['midi']['end_frames'][:]
                
                # Check for valid MIDI note range
                if any(note < 0 or note > 127 for note in midi_notes):
                    print(f"  - Warning: Sample '{sample_id}' has invalid MIDI note values")
                    issues += 1
                
                # Check for MIDI frame alignment
                if max(midi_end_frames) > mel_spec.shape[1]:
                    print(f"  - Warning: Sample '{sample_id}' has MIDI frames exceeding spectrogram length")
                    issues += 1
            
            valid_samples += 1
        
        # Print summary statistics
        print(f"\nValidation Summary:")
        print(f"  - Valid samples: {valid_samples}/{len(file_list)} ({valid_samples/len(file_list)*100:.1f}%)")
        print(f"  - Issues found: {issues}")
        
        # Calculate dataset statistics
        print("\nDataset Statistics:")
        total_duration = 0
        total_phonemes = 0
        total_midi_notes = 0
        phoneme_counts = {}
        note_counts = {}
        
        for sample_id in file_list:
            if sample_id in f:
                sample = f[sample_id]
                total_duration += sample.attrs.get('duration', 0)
                
                # Phoneme statistics
                if 'phonemes' in sample and 'phones' in sample['phonemes']:
                    phones = sample['phonemes']['phones'][:]
                    total_phonemes += len(phones)
                    
                    # Count phoneme occurrences
                    for phone_bytes in phones:
                        phone = phone_bytes.decode('utf-8')
                        phoneme_counts[phone] = phoneme_counts.get(phone, 0) + 1
                
                # MIDI statistics
                if 'midi' in sample:
                    num_notes = sample['midi'].attrs.get('num_notes', 0)
                    total_midi_notes += num_notes
                    
                    if num_notes > 0:
                        midi_notes = sample['midi']['notes'][:]
                        
                        # Count note occurrences
                        for note in midi_notes:
                            note_name = midi_to_note_name(note)
                            note_counts[note_name] = note_counts.get(note_name, 0) + 1
        
        print(f"  - Total audio duration: {total_duration:.2f} seconds")
        print(f"  - Total phonemes: {total_phonemes}")
        print(f"  - Total MIDI notes: {total_midi_notes}")
        print(f"  - Average phonemes per second: {total_phonemes/total_duration:.2f}")
        print(f"  - Average MIDI notes per second: {total_midi_notes/total_duration:.2f}")
        print(f"  - Percentage of phonemes with MIDI notes: {total_midi_notes/total_phonemes*100:.1f}%")
        
        # Print top 10 most common phonemes
        top_phonemes = sorted(phoneme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 most common phonemes:")
        for phone, count in top_phonemes:
            print(f"  - {phone}: {count} occurrences ({count/total_phonemes*100:.1f}%)")
        
        # Print top 10 most common MIDI notes
        if note_counts:
            top_notes = sorted(note_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            print("\nTop 10 most common MIDI notes:")
            for note, count in top_notes:
                print(f"  - {note}: {count} occurrences ({count/total_midi_notes*100:.1f}%)")
        
        return valid_samples, issues

def main():
    print("Starting audio data processing...")
    
    config_path = "config/default.yaml"
    config = read_config(config_path)
    data_raw_path = config['datasets']['data_raw']
    binary_dir = os.path.join(data_raw_path, "binary")
    
    # Process files and create single HDF5 file with MIDI information
    h5_path, sample_id = process_files(config_path, midi_method='median')
    
    if h5_path and sample_id:
        # Validate the dataset
        validate_dataset(h5_path)
        
        # Create visualization for the first sample
        vis_output_path = os.path.join(binary_dir, f"{sample_id}_visualization.png")
        create_visualization(sample_id, h5_path, vis_output_path, config)
    
    print("Processing complete!")

if __name__ == '__main__':
    main()