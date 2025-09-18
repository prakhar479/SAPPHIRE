"""
music_multifeature_extract.py

Single-file, modular extractor for audio+lyrics perceptual features.
Outputs a Python dict (JSON-serializable) called `features`.

Author: ChatGPT (adapted for user's feature list)
"""

import os
import json
import math
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import scipy
import librosa
import soundfile as sf
import pyloudnorm as pyln
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import hashlib

# Optional libs with graceful fallback
try:
    import crepe
    HAS_CREPE = True
except Exception:
    HAS_CREPE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT = SentenceTransformer('all-mpnet-base-v2')
except Exception:
    SBERT = None

# Phonetics: CMU pronouncing dictionary via 'pronouncing'
try:
    import pronouncing
except Exception:
    pronouncing = None

# Sentiment: use NLTK VADER as a fallback
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk_vader = SentimentIntensityAnalyzer()
except Exception:
    nltk_vader = None

# --------------------------
# Utilities
# --------------------------
def load_audio(path: str, sr: Optional[int]=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def save_json(obj, path):
    with open(path,'w',encoding='utf8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# --------------------------
# Acoustic features
# --------------------------
def perceptual_mfcc(y, sr, n_mfcc=13, apply_a_weight=True):
    """
    Compute MFCCs, and optionally apply A-weighting approximation to the power spectrum
    to emphasize perceptual loudness before mel filter bank.
    """
    # compute STFT power
    S = np.abs(librosa.stft(y, n_fft=2048))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    if apply_a_weight:
        # approximate A-weighting filter (not ISO226 exact but perceptual)
        # A-weight curve approximation:
        def a_weight(f):
            f = np.asarray(f, dtype=float)
            ra = (12194**2 * f**4) / ((f**2 + 20.6**2) * np.sqrt((f**2 + 107.7**2)*(f**2 + 737.9**2))*(f**2+12194**2))
            with np.errstate(divide='ignore'):
                aw = 20*np.log10(ra) + 2.00
            return aw
        aw = a_weight(freqs)
        # convert dB to linear scale and multiply each bin
        aw_lin = 10**(aw/20.0)
        S = S * aw_lin[:,None]

    # mel spectrogram then mfcc
    S_mel = librosa.feature.melspectrogram(sr=sr, S=S)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S_mel), n_mfcc=n_mfcc)
    # return mean & std as summary plus full matrix
    return {'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
            'mfcc_std': np.std(mfcc, axis=1).tolist(),
            'mfcc_matrix': mfcc.tolist()}

def chroma_features(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    # estimate key by matching to templates (simple approach)
    # templates for major/minor (Krumhansl-Schmuckler style)
    maj_template = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    min_template = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    def correlate_to_template(chroma_vec, template):
        scores = []
        for shift in range(12):
            rolled = np.roll(template, shift)
            scores.append(np.corrcoef(chroma_vec, rolled)[0,1])
        best = int(np.nanargmax(scores))
        return best, max(scores)
    # find best major or minor
    k_maj, s_maj = correlate_to_template(chroma_mean, maj_template)
    k_min, s_min = correlate_to_template(chroma_mean, min_template)
    if s_maj >= s_min:
        key = {'tonic': k_maj, 'mode': 'major', 'score': float(s_maj)}
    else:
        key = {'tonic': k_min, 'mode': 'minor', 'score': float(s_min)}
    return {'chroma_mean': chroma_mean.tolist(), 'chroma_std': chroma_std.tolist(), 'estimated_key': key}

def loudness_curve(y, sr, frame_length=2048, hop_length=512):
    """
    Returns short-term loudness (approx LUFS per frame).
    Uses pyloudnorm to measure integrated LUFS for short windows.
    """
    meter = pyln.Meter(sr)
    # frame-wise loudness (compute RMS then convert)
    times = librosa.times_like(y, sr=sr, hop_length=hop_length)
    loudness_curve = []
    for i in range(0, len(y), hop_length):
        frame = y[i:i+frame_length]
        if len(frame) < frame_length//4:
            continue
        try:
            l = meter.integrated_loudness(frame)
        except Exception:
            # fallback to dBFS
            rms = np.sqrt(np.mean(frame**2)+1e-12)
            l = 20*np.log10(rms+1e-12)
        loudness_curve.append(float(l))
    return {'times': times[:len(loudness_curve)].tolist(), 'loudness_db': loudness_curve}

def spectral_descriptors(y, sr):
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    return {'centroid_mean': float(np.mean(spec_centroid)),
            'bandwidth_mean': float(np.mean(spec_bw)),
            'zcr_mean': float(np.mean(zcr))}

# --------------------------
# Rhythm & groove
# --------------------------
def rhythm_features(y, sr):
    # tempo + beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='time')
    # onset envelope and onset times for density
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    duration = len(y)/sr
    density = len(onset_times) / duration if duration>0 else 0.0
    # micro-timing: compute actual beat times and compare to ideal grid (16th)
    tempo_bpm = tempo
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) if len(beat_frames)>0 else []
    # create a 16th-note grid from first beat
    micro_dev = None
    if len(beat_times) >= 1:
        first = beat_times[0]
        beat_period = 60.0 / max(tempo_bpm,1e-6)
        # sample beat subdivisions around the audio to compute average deviation
        grid_positions = []
        for bt in beat_times:
            for s in range(4):  # 16th subdivisions
                grid_positions.append(bt + (s * beat_period / 4.0))
        # find nearest grid to each onset and compute deviation
        onsets = onset_times
        deviations = []
        if len(grid_positions)>0:
            gp = np.array(grid_positions)
            for o in onsets:
                idx = np.argmin(np.abs(gp - o))
                deviations.append(float(o - gp[idx]))
        micro_dev = {'median_dev_sec': float(np.median(deviations)) if deviations else 0.0,
                     'mean_dev_sec': float(np.mean(deviations)) if deviations else 0.0,
                     'std_dev_sec': float(np.std(deviations)) if deviations else 0.0}
    # simple swing estimate: compare durations between subdivisions
    swing = None
    if len(onset_times) > 3:
        inter = np.diff(onset_times)
        # swing ratio = median of odd inter-onset / even inter-onset (heuristic)
        odd = inter[::2]
        even = inter[1::2]
        if len(even)>0 and np.median(even)>0:
            swing = float(np.median(odd)/np.median(even))
    return {'tempo_bpm': float(tempo_bpm),
            'onset_count': int(len(onset_times)),
            'rhythmic_density_onsets_per_sec': float(density),
            'microtiming': micro_dev,
            'swing_ratio_est': swing}

# --------------------------
# Melody & pitch
# --------------------------
def pitch_contour_crepe(y_path: str, sr: int = 22050, hop_length=256):
    """
    Use CREPE to estimate pitch contour. CREPE expects a file path.
    If CREPE unavailable, use librosa.piptrack fallback (function below).
    Returns times & fundamental frequency (Hz).
    """
    if not HAS_CREPE:
        raise RuntimeError("CREPE not available; use pitch_contour_librosa instead.")
    # crepe.predict returns (f0, confidence) arrays
    import soundfile as sf
    audio, sr_read = sf.read(y_path, dtype='float32')
    if audio.ndim>1:
        audio = np.mean(audio, axis=1)
    # run crepe (uses default model step_size=10ms)
    time, frequency, confidence, activation = crepe.predict(audio, sr_read, viterbi=True)
    return {'times': time.tolist(), 'frequency_hz': frequency.tolist(), 'confidence': confidence.tolist()}

def pitch_contour_librosa(y, sr, hop_length=256):
    # use harmonic extraction via librosa.effects.harmonic then piptrack
    y_h = librosa.effects.harmonic(y)
    S = np.abs(librosa.stft(y_h, n_fft=2048, hop_length=hop_length))
    pitches, mags = librosa.piptrack(S=S, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr, hop_length=hop_length)
    f0 = []
    for i in range(pitches.shape[1]):
        col = pitches[:, i]
        magcol = mags[:, i]
        if np.max(magcol) < 1e-6:
            f0.append(0.0)
        else:
            idx = magcol.argmax()
            f0.append(float(col[idx]))
    return {'times': times.tolist(), 'frequency_hz': f0}

def vibrato_stats(pitch_freqs, times):
    # compute instantaneous frequency deviations in cents around smoothed contour
    freqs = np.array(pitch_freqs)
    mask = freqs > 0
    if mask.sum() < 3:
        return {'vibrato_rate_hz': None, 'vibrato_extent_cents': None}
    # smooth contour
    from scipy.signal import savgol_filter
    smooth = savgol_filter(freqs[mask], 9, 3) if len(freqs[mask])>=9 else freqs[mask]
    residual = freqs[mask] - smooth
    # convert to cents relative to smooth
    cents = 1200.0 * np.log2((residual + smooth + 1e-12) / (smooth + 1e-12))
    # estimate rate via FFT
    if len(times) >= 2:
        dt = np.mean(np.diff(np.array(times)[mask]))
        # compute main frequency in residual
        import numpy.fft as fft
        A = np.abs(fft.rfft(cents - np.mean(cents)))
        freqs_fft = fft.rfftfreq(len(cents), d=dt)
        idx = np.argmax(A[1:]) + 1 if len(A)>1 else 0
        rate = float(freqs_fft[idx]) if len(freqs_fft)>0 else None
    else:
        rate = None
    extent = float(np.percentile(np.abs(cents), 95))
    return {'vibrato_rate_hz': rate, 'vibrato_extent_cents': extent}

def find_melodic_motifs(freqs, times, min_len_frames=6, n_best=10):
    """
    Quantize to semitones, make interval sequence, find repeating subsequences by hashing.
    Returns top repeating motifs (simple).
    """
    freqs = np.array(freqs)
    mask = freqs > 0
    if mask.sum() < min_len_frames:
        return []
    f = freqs[mask]
    # convert to MIDI numbers
    midi = 69 + 12 * np.log2(f/440.0 + 1e-12)
    midi_q = np.round(midi).astype(int)
    # intervals
    intervals = np.diff(midi_q)
    # collect n-grams of intervals
    ngrams = defaultdict(list)
    for n in range(3, 8):  # motif lengths (in intervals)
        for i in range(len(intervals)-n+1):
            key = tuple(intervals[i:i+n])
            ngrams[key].append(i)
    # select repeated ones
    repeated = [(k,v) for k,v in ngrams.items() if len(v) > 1]
    # sort by (repetitions * length)
    repeated_sorted = sorted(repeated, key=lambda kv: (len(kv[1])*len(kv[0])), reverse=True)
    motifs = []
    for k,v in repeated_sorted[:n_best]:
        motifs.append({'intervals': list(k), 'occurrences': len(v)})
    return motifs

# --------------------------
# Lyrical features
# --------------------------
def load_lyrics(path: str) -> str:
    with open(path, 'r', encoding='utf8') as f:
        text = f.read()
    return text

def lyric_semantic_vector(text: str):
    if SBERT is not None:
        vec = SBERT.encode([text])[0]
        return {'embedding': vec.tolist(), 'dim': len(vec)}
    else:
        # fallback: TF-IDF would be used; for brevity return empty
        return {'embedding': None, 'note': 'sentence-transformers not available'}

def lyric_sentiment_arc(text: str, window_size=50):
    # split into tokens (words) and compute rolling sentiment score using VADER if available
    tokens = text.split()
    if nltk_vader is None:
        return {'note': 'VADER not available', 'sentiment_arc': None}
    scores = []
    for i in range(0, max(1, len(tokens)), window_size):
        span = " ".join(tokens[i:i+window_size])
        s = nltk_vader.polarity_scores(span)['compound']
        scores.append(float(s))
    return {'sentiment_arc': scores, 'window_size_words': window_size}

def phonetic_patterns(text: str):
    """
    Use CMU dict via pronouncing to get phoneme sequences for English words.
    Compute phoneme rate (phonemes per word, phonemes per second if audio duration given).
    """
    if pronouncing is None:
        return {'note': 'pronouncing not available', 'phoneme_rate': None}
    words = [w.strip(".,;:!?()[]\"'").lower() for w in text.split()]
    phonemes_seq = []
    missing = 0
    for w in words:
        try:
            p = pronouncing.phones_for_word(w)
            if p:
                # choose first pronunciation
                phonemes_seq.extend(p[0].split())
            else:
                missing += 1
        except Exception:
            missing += 1
    total_phonemes = len(phonemes_seq)
    avg_phonemes_per_word = total_phonemes / max(1, len(words))
    # phoneme type diversity
    unique_phonemes = len(set(phonemes_seq))
    return {'total_phonemes': int(total_phonemes),
            'avg_phonemes_per_word': float(avg_phonemes_per_word),
            'unique_phonemes': int(unique_phonemes),
            'words_missing_pron': int(missing)}

# --------------------------
# Structural: segmentation + chords + repetition
# --------------------------
def structure_segmentation(y, sr, hop_length=512, n_segments=6):
    # compute timbre features across frames and compute self-similarity
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S**2), n_mfcc=13)
    # compute recurrence / self-similarity
    R = librosa.segment.recurrence_matrix(mfcc, backtrack=False, width=3, metric='cosine')
    # create feature for clustering by using aggregated mfcc per frame
    features = mfcc.T  # frames x features
    # Agglomerative clustering into n_segments
    if features.shape[0] < n_segments:
        labels = np.zeros(features.shape[0], dtype=int).tolist()
    else:
        cluster = AgglomerativeClustering(n_clusters=n_segments, affinity='cosine', linkage='average')
        labels = cluster.fit_predict(features).tolist()
    # compress adjacent identical labels into segments with times
    times = librosa.frames_to_time(np.arange(len(labels)), sr=sr, hop_length=hop_length)
    segments = []
    if len(labels)>0:
        cur_label = labels[0]
        start_t = times[0]
        for i,l in enumerate(labels[1:], start=1):
            if l != cur_label:
                end_t = times[i]
                segments.append({'label': int(cur_label), 'start': float(start_t), 'end': float(end_t)})
                cur_label = l
                start_t = times[i]
        segments.append({'label': int(cur_label), 'start': float(start_t), 'end': float(times[-1])})
    return {'labels': labels, 'segments': segments}

def chord_estimation(y, sr, hop_length=512):
    # compute chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    # simple chord templates (major/minor triads)
    chord_templates = {}
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    def triad_template(root, kind):
        t = np.zeros(12)
        r = root
        if kind=='maj':
            t[r]=1; t[(r+4)%12]=1; t[(r+7)%12]=1
        else:
            t[r]=1; t[(r+3)%12]=1; t[(r+7)%12]=1
        return t
    for i,nm in enumerate(notes):
        chord_templates[f"{nm}:maj"] = triad_template(i,'maj')
        chord_templates[f"{nm}:min"] = triad_template(i,'min')
    chord_seq = []
    for f in range(chroma.shape[1]):
        vec = chroma[:,f]
        best = None
        best_score = -1
        for chname, templ in chord_templates.items():
            s = np.dot(vec, templ) / (np.linalg.norm(vec)+1e-9) # cosine-like
            if s > best_score:
                best_score = float(s); best = chname
        chord_seq.append({'time': float(times[f]), 'chord': best, 'score': best_score})
    return {'chord_sequence': chord_seq}

def repetition_detection_segments(segments):
    # hash each segment's label and duration to find repeated segments
    seg_hashes = defaultdict(list)
    for s in segments:
        h = hashlib.sha1(f"{s['label']}-{round(s['end']-s['start'],3)}".encode()).hexdigest()[:8]
        seg_hashes[h].append(s)
    repeats = []
    for h, occur in seg_hashes.items():
        if len(occur)>1:
            repeats.append({'hash':h, 'occurrences': len(occur), 'instances': occur})
    return {'repeats': repeats}

# --------------------------
# Main orchestrator
# --------------------------
def analyze(audio_path: Optional[str]=None, lyrics_path: Optional[str]=None, out_json: Optional[str]=None) -> Dict[str,Any]:
    features = {}
    y = None; sr=None
    if audio_path and os.path.exists(audio_path):
        y, sr = load_audio(audio_path, sr=22050)
        features['acoustic'] = {}
        features['acoustic'].update(perceptual_mfcc(y, sr))
        features['acoustic'].update(chroma_features(y, sr))
        features['acoustic']['loudness'] = loudness_curve(y, sr)
        features['acoustic']['spectral'] = spectral_descriptors(y, sr)
        features['rhythm'] = rhythm_features(y, sr)
        # pitch (use crepe if available)
        try:
            if HAS_CREPE:
                pc = pitch_contour_crepe(audio_path, sr)
            else:
                pc = pitch_contour_librosa(y, sr)
        except Exception:
            pc = pitch_contour_librosa(y, sr)
        features['melody'] = {}
        features['melody']['pitch_contour'] = pc
        features['melody']['vibrato'] = vibrato_stats(pc['frequency_hz'], pc['times'])
        features['melody']['motifs'] = find_melodic_motifs(pc['frequency_hz'], pc['times'])
        # structure & harmony
        struct = structure_segmentation(y, sr)
        features['structure'] = struct
        features['structure']['chords'] = chord_estimation(y, sr)
        features['structure']['repetition'] = repetition_detection_segments(struct['segments'])
    # lyrics only
    if lyrics_path and os.path.exists(lyrics_path):
        text = load_lyrics(lyrics_path)
        features['lyrics'] = {}
        features['lyrics']['raw'] = None  # don't serialize full raw by default
        features['lyrics']['semantic'] = lyric_semantic_vector(text)
        features['lyrics']['sentiment_arc'] = lyric_sentiment_arc(text)
        features['lyrics']['phonetics'] = phonetic_patterns(text)
    # save JSON optionally
    if out_json:
        save_json(features, out_json)
    return features

# CLI usage
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--audio', type=str, help='path to audio file (wav, mp3)')
    p.add_argument('--lyrics', type=str, help='path to lyrics text file')
    p.add_argument('--out', type=str, default='features.json')
    args = p.parse_args()
    feats = analyze(audio_path=args.audio, lyrics_path=args.lyrics, out_json=args.out)
    print("Saved features ->", args.out)
