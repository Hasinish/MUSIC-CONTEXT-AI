"""
audio_features.py - Audio feature extraction and segmentation pipeline.
Part of the GNN-BERT Music Context Understanding project (CSE425).

Implements:
- Audio resampling to 22,050 Hz
- Log-mel spectrogram extraction (128 bins)
- Chroma feature extraction (12 pitch classes)
- Fixed-duration window segmentation (5-second segments)
- Per-track and per-segment normalization
- Synthetic audio generator for zero-dependency testing and verification
"""

import os
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

# Attempt librosa import, provide pure numpy/scipy fallback for robust execution
try:
    import librosa  # type: ignore
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    from scipy.signal import spectrogram  # type: ignore


def load_audio(
    filepath: str,
    target_sr: int = 22050,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to the target sampling rate (default 22,050 Hz).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found at: {filepath}")

    if LIBROSA_AVAILABLE:
        y, sr = librosa.load(filepath, sr=target_sr, mono=mono)
        return y, sr
    else:
        # Fallback using scipy.io.wavfile if available
        import scipy.io.wavfile as wavfile
        from scipy.signal import resample
        sr, data = wavfile.read(filepath)
        if data.ndim > 1 and mono:
            data = data.mean(axis=1)
        if sr != target_sr:
            num_samples = int(len(data) * target_sr / sr)
            data = resample(data, num_samples)
        # Normalize to [-1.0, 1.0]
        data = data.astype(np.float32) / (np.max(np.abs(data)) + 1e-8)
        return data, target_sr


def extract_log_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract log-mel spectrogram (n_mels x T frames) and normalize to [0, 1].
    """
    if LIBROSA_AVAILABLE:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
    else:
        # Pure numpy/scipy spectrogram fallback
        f, t, Sxx = spectrogram(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        # Bin frequencies into n_mels buckets
        mel_bins = np.linspace(0, Sxx.shape[0] - 1, n_mels + 1, dtype=int)
        log_mel = np.zeros((n_mels, Sxx.shape[1]), dtype=np.float32)
        for i in range(n_mels):
            log_mel[i, :] = Sxx[mel_bins[i]:mel_bins[i + 1] + 1, :].mean(axis=0)
        log_mel = 10 * np.log10(np.maximum(log_mel, 1e-8))
        log_mel = log_mel - np.max(log_mel)

    # Per-track min-max normalization
    min_val = np.min(log_mel)
    max_val = np.max(log_mel)
    norm_mel = (log_mel - min_val) / (max_val - min_val + 1e-8)
    return norm_mel.astype(np.float32)


def extract_chroma(
    y: np.ndarray,
    sr: int = 22050,
    n_chroma: int = 12,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract 12-dimensional chroma pitch-class features (12 x T frames).
    """
    if LIBROSA_AVAILABLE:
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma
        )
    else:
        # Fallback chroma computation via modular 12-bin pitch wrapping
        f, t, Sxx = spectrogram(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        chroma = np.zeros((n_chroma, Sxx.shape[1]), dtype=np.float32)
        # Approximate frequency to pitch class
        for idx, freq in enumerate(f):
            if freq > 65.4: # C2 frequency floor
                midi_note = int(round(12 * math.log2(freq / 440.0) + 69))
                pitch_class = midi_note % 12
                chroma[pitch_class, :] += Sxx[idx, :]
        # Normalize per frame
        norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8
        chroma = chroma / norm

    return chroma.astype(np.float32)


def segment_audio(
    y: np.ndarray,
    sr: int = 22050,
    segment_duration: float = 5.0,
    hop_duration: float = 5.0
) -> List[np.ndarray]:
    """
    Slice an audio waveform into fixed-duration window segments (e.g. 5.0 seconds each).
    """
    segment_len = int(segment_duration * sr)
    hop_len = int(hop_duration * sr)
    total_len = len(y)

    segments = []
    start = 0
    while start + segment_len <= total_len:
        seg = y[start : start + segment_len]
        segments.append(seg)
        start += hop_len

    # If remaining tail has at least 50% segment length, pad to full segment
    if total_len - start >= segment_len // 2:
        tail = y[start:]
        padded_seg = np.pad(tail, (0, segment_len - len(tail)), mode='constant')
        segments.append(padded_seg)

    # In case audio is shorter than 1 segment, pad it
    if len(segments) == 0:
        padded = np.pad(y, (0, max(0, segment_len - len(y))), mode='constant')
        segments.append(padded[:segment_len])

    return segments


def extract_segment_features(
    segments: List[np.ndarray],
    sr: int = 22050,
    n_mels: int = 128,
    n_chroma: int = 12
) -> np.ndarray:
    """
    Extract pooled summary feature vectors for each segment.
    Output shape: (num_segments, n_mels + n_chroma) = (num_segments, 140)
    Each row represents initial node feature vector h_i^(0) for graph node i.
    """
    node_features = []

    for seg in segments:
        # Mel summary (mean across time frames)
        mel = extract_log_mel_spectrogram(seg, sr=sr, n_mels=n_mels)
        mel_mean = np.mean(mel, axis=1) # (128,)

        # Chroma summary (mean across time frames)
        chroma = extract_chroma(seg, sr=sr, n_chroma=n_chroma)
        chroma_mean = np.mean(chroma, axis=1) # (12,)

        # Concatenate into unified 140-dim segment embedding
        feat = np.concatenate([mel_mean, chroma_mean], axis=0)
        node_features.append(feat)

    return np.array(node_features, dtype=np.float32)


def generate_synthetic_music_track(
    duration: float = 30.0,
    sr: int = 22050,
    genre_seed: int = 0
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Generate a realistic synthetic 30-second music track with harmonic chord progressions
    (C -> G -> Am -> F), rhythmic percussion pulses, and mock metadata tags.
    Used for instant unit testing, pipeline verification, and sample graph generation.
    """
    np.random.seed(genre_seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = np.zeros_like(t)

    # Chord progression frequencies: C major (261.6, 329.6, 392.0), G major (196.0, 246.9, 293.7),
    # A minor (220.0, 261.6, 330.0), F major (174.6, 220.0, 261.6)
    chords = [
        [261.63, 329.63, 392.00], # C
        [196.00, 246.94, 293.66], # G
        [220.00, 261.63, 329.63], # Am
        [174.61, 220.00, 261.63], # F
    ]

    chord_duration = 5.0 # Each chord lasts 5 seconds (aligns with 5s graph nodes!)
    num_chords = int(duration / chord_duration)

    for i in range(num_chords):
        chord = chords[i % len(chords)]
        start_idx = int(i * chord_duration * sr)
        end_idx = int(min((i + 1) * chord_duration * sr, len(t)))
        t_slice = t[start_idx:end_idx]

        chord_wave = np.zeros(len(t_slice))
        for freq in chord:
            # Fundamental + 2nd harmonic
            chord_wave += 0.4 * np.sin(2 * np.pi * freq * t_slice)
            chord_wave += 0.15 * np.sin(2 * np.pi * 2 * freq * t_slice)

        # Apply smooth attack and decay envelope to each chord window
        envelope = np.hanning(len(t_slice))
        waveform[start_idx:end_idx] += chord_wave * envelope

    # Add rhythmic beat pulses (120 BPM = 2 beats per second)
    beat_interval = int(sr * 0.5)
    for b in range(0, len(waveform), beat_interval):
        hit_len = min(int(sr * 0.08), len(waveform) - b)
        waveform[b : b + hit_len] += 0.3 * np.random.randn(hit_len) * np.exp(-np.linspace(0, 5, hit_len))

    # Add subtle colored background ambiance
    waveform += 0.02 * np.random.randn(len(waveform))

    # Normalize audio to [-0.95, 0.95]
    waveform = 0.95 * waveform / (np.max(np.abs(waveform)) + 1e-8)

    genres = ["Rock", "Electronic", "Jazz", "Classical", "Pop", "Hip-Hop", "Folk", "Metal"]
    assigned_genre = genres[genre_seed % len(genres)]
    
    metadata = {
        "track_id": f"synthetic_track_{genre_seed:03d}",
        "duration": duration,
        "sample_rate": sr,
        "genre": assigned_genre,
        "tags": [assigned_genre.lower(), "melodic", "progression", "synthesized", "rhythmic"],
        "caption": f"A dynamic {assigned_genre.lower()} instrumental piece featuring repeating chord progressions, rhythmic percussion, and rich harmonic texture.",
        "valence": float(np.clip(0.3 + 0.1 * (genre_seed % 5), 0.1, 0.9)),
        "arousal": float(np.clip(0.4 + 0.12 * (genre_seed % 5), 0.1, 0.9)),
    }

    return waveform.astype(np.float32), metadata


if __name__ == "__main__":
    print("=== Testing audio_features.py Pipeline ===")
    
    # Generate a synthetic track
    wave, meta = generate_synthetic_music_track(duration=30.0, sr=22050, genre_seed=1)
    print(f"Generated track: {meta['track_id']} | Genre: {meta['genre']} | Duration: {meta['duration']}s")
    print(f"Waveform shape: {wave.shape}, Min: {wave.min():.2f}, Max: {wave.max():.2f}")

    # Segment audio
    segments = segment_audio(wave, sr=22050, segment_duration=5.0)
    print(f"Extracted {len(segments)} segments of 5 seconds each.")

    # Extract features
    features = extract_segment_features(segments, sr=22050, n_mels=128, n_chroma=12)
    print(f"Segment node features shape: {features.shape} (Segments x Features)")
    assert features.shape == (6, 140), f"Expected shape (6, 140), got {features.shape}"

    print("SUCCESS: audio_features.py unit test passed flawlessly!")
