#!/usr/bin/env python3
"""
PyFL Studio - an FL Studio-inspired DAW written in Python.

Install:   ./install.sh          (installs every dependency, creates run.sh)
Run:       ./run.sh              (or: python pyfl_studio.py)

Features
  * Channel Rack with step sequencer (instrument, sample and audio-clip tracks)
  * Piano Roll (click = add note, drag = resize/move, right-click = delete,
    Ctrl+click = slur a note into the next one, Alt+click = set choir vowel)
  * Built-in instruments, all in  Add > track menu:
      Piano, Electric Piano, Organ, Marimba, Acoustic Guitar, Electric Guitar,
      Violin, Cello, String Ensemble, Flute, Brass, Choir (sings A E I O U),
      Synth, Synth Bass, Synth Pad
  * Slurs on every instrument (legato + glide, no re-trigger)
  * Audio clips: drag .mp3 / .wav / .flac / .ogg files into the song; they are
    placed at the play bar and can be dragged around the timeline
  * Audio-clip effects (select an audio track): Auto-Tune, Auto-Harmonizer and a Vocoder.  Press "Notes for
    effects" in the piano roll and draw notes/chords on the track like on any synth: Auto-Tune sings the notes
    you draw, the Harmonizer turns a chord into harmony voices, and the Vocoder plays the notes through a synth
    sound of your choice (or a chord drone / noise / another track).  Without notes they follow a key and scale.
  * Hum notes: pick an instrument track, hum into the microphone and the notes
    (with their lengths) are written into the piano roll
  * Play bar: click or drag the ruler (or the white line in the piano roll) to
    move it; Play starts from the play bar, Restart jumps to the start
  * Mixer: pan, low-pass filter, drive, reverb send, delay send, master
  * MIDI file import/export, live MIDI keyboard input, typing keyboard as piano
  * WAV export, project save/load (.pyfl)
Shortcuts: Space = play/pause, Home = play bar to start, Z-M / Q-P rows = piano keys
"""
import sys, os, json, math, threading, traceback, time, shutil, subprocess, copy
from math import gcd
import numpy as np
from scipy.signal import lfilter, resample_poly, sosfilt, sosfiltfilt, butter
from scipy.ndimage import uniform_filter1d
from scipy.io import wavfile

try:
    import sounddevice as sd
except Exception:
    sd = None
try:
    import mido
except Exception:
    mido = None
try:
    import soundfile as sf
except Exception:
    sf = None
try:
    import miniaudio
except Exception:
    miniaudio = None

SR, BLOCK, STREAM_BLOCK = 44100, 512, 1024
ROOT, P_LO, P_HI, MAX_LEN = 60, 24, 108, 2048
LEN_CHOICES = [16, 32, 64, 128, 256, 512, 1024, 2048]
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff")
MIDI_EXT = (".mid", ".midi")
PALETTE = ["#ff6b4a", "#ffb13b", "#9ed84b", "#4bd1c5", "#5aa9ff", "#b57bff", "#ff74b8", "#e6e66a"]
VOWEL_NAMES = ["A  (ah)", "E  (eh)", "I  (ee)", "O  (oh)", "U  (oo)"]
VOWEL_LETTERS = "AEIOU"
TWO_PI = 2.0 * np.pi
LEVEL = 0.577  # rms of a full-scale saw: every instrument is normalised to this loudness
_RNG = np.random.default_rng(12345)


# ============================================================ FILE LOADING
def _resample(data, sr):
    if int(sr) == SR:
        return data
    g = gcd(int(sr), SR)
    return resample_poly(data, SR // g, int(sr) // g, axis=1).astype(np.float32)


def _decode_ffmpeg(path):
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found")
    ch = 2
    probe = shutil.which("ffprobe")
    if probe:  # keep mono files at full level (ffmpeg's mono->stereo upmix is -3 dB)
        try:
            r = subprocess.run([probe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                                "stream=channels", "-of", "csv=p=0", path], capture_output=True, text=True)
            ch = max(1, min(2, int(r.stdout.strip().split()[0])))
        except Exception:
            ch = 2
    out = subprocess.run([exe, "-v", "error", "-i", path, "-f", "f32le", "-ac", str(ch), "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).reshape(-1, ch).T.copy()


def load_sample(path):
    """Load wav / mp3 / flac / ogg / aiff as float32 (2, N) at SR.
    Tries soundfile, then miniaudio, then scipy (wav), then the ffmpeg command line."""
    data, sr, errs = None, SR, []
    if sf is not None:
        try:
            d, sr = sf.read(path, always_2d=True, dtype="float32")
            data = d.T
        except Exception as ex:
            errs.append(f"soundfile: {ex}")
    if data is None and miniaudio is not None:
        try:
            dec = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.FLOAT32,
                                        nchannels=2, sample_rate=SR)
            data, sr = np.asarray(dec.samples, dtype=np.float32).reshape(-1, 2).T, SR
        except Exception as ex:
            errs.append(f"miniaudio: {ex}")
    if data is None and path.lower().endswith(".wav"):
        try:
            sr, d = wavfile.read(path)
            if d.dtype.kind == "i":
                d = d / float(np.iinfo(d.dtype).max)
            elif d.dtype.kind == "u":
                d = (d.astype(np.float32) - 128.0) / 128.0
            d = d.astype(np.float32)
            data = (d[:, None] if d.ndim == 1 else d).T
        except Exception as ex:
            errs.append(f"wav: {ex}")
    if data is None:
        try:
            data, sr = _decode_ffmpeg(path), SR
        except Exception as ex:
            errs.append(f"ffmpeg: {ex}")
    if data is None:
        raise RuntimeError("Could not decode this file.\n" + "\n".join(errs) +
                           "\n\nTip: run install.sh (it installs soundfile, miniaudio and ffmpeg).")
    data = np.asarray(data, dtype=np.float32)
    if data.shape[0] == 1:
        data = np.vstack([data, data])
    data = data[:2]
    data = _resample(data, sr)
    if data.shape[1] < 2:
        data = np.zeros((2, 2), np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


def make_peaks(data, nb=None):
    """Waveform overview (max abs per bucket, ~100 buckets per second) used to draw audio clips."""
    mono = np.abs(data).max(axis=0)
    if nb is None:
        nb = int(min(40000, max(2000, len(mono) / SR * 100)))
    chunk = max(1, int(math.ceil(len(mono) / nb)))
    pad = (-len(mono)) % chunk
    if pad:
        mono = np.concatenate([mono, np.zeros(pad, mono.dtype)])
    pk = mono.reshape(-1, chunk).max(axis=1)
    return (pk / max(float(pk.max()), 1e-9)).astype(np.float32)


_DRUMS = {}


def make_drums():
    if _DRUMS:
        return _DRUMS
    rng = np.random.default_rng(1)
    T = lambda d: np.arange(int(d * SR)) / SR
    fin = lambda a: np.vstack([a, a]).astype(np.float32) * (0.9 / max(1e-6, np.abs(a).max()))
    x = T(0.45)
    ph = 2 * np.pi * np.cumsum(45 + 130 * np.exp(-x * 28)) / SR
    _DRUMS["kick"] = fin(np.sin(ph) * np.exp(-x * 7))
    x = T(0.25)
    _DRUMS["snare"] = fin(rng.standard_normal(len(x)) * np.exp(-x * 22) * 0.6 + np.sin(2 * np.pi * 185 * x) * np.exp(-x * 28) * 0.5)
    x = T(0.09)
    _DRUMS["hat"] = fin(np.diff(rng.standard_normal(len(x)), prepend=0) * np.exp(-x * 70))
    x = T(0.3)
    env = np.exp(-x * 14) * 0.6
    for k in (0.0, 0.011, 0.022):
        env += np.where(x >= k, np.exp(-np.clip(x - k, 0, None) * 70), 0)
    _DRUMS["clap"] = fin(np.diff(rng.standard_normal(len(x)), prepend=0) * env)
    return _DRUMS


# ============================================================ INSTRUMENT DSP
# Every instrument is additive / FM synthesis evaluated block by block.  The voice keeps one
# "fundamental phase" accumulator, so vibrato, glides and slurs are all phase-continuous.
def _rms(a):
    return math.sqrt(float(np.sum(a * a)) * 0.5)


def midi_hz(p):
    return 440.0 * 2.0 ** ((np.asarray(p, dtype=np.float64) - 69.0) / 12.0)


def _hcount(fm, hmax):
    return int(max(1, min(hmax, 0.45 * SR / max(float(fm), 20.0))))


def _noise(n):
    x = _RNG.standard_normal(n + 3)
    return (x[:-3] + x[1:-2] + x[2:-1] + x[3:]) * 0.5


_RAMPS32 = {}


def _ramp32(n):
    r = _RAMPS32.get(n)
    if r is None:
        r = _RAMPS32[n] = np.linspace(0.0, 1.0, n).astype(np.float32)
    return r


def _period(hs):
    """Phase period (in cycles) that makes every partial exactly periodic, or None if inharmonic."""
    for per in (1.0, 2.0):
        if np.allclose(hs * per, np.round(hs * per), atol=1e-9):
            return per
    return None


def additive(ph, ratios, A0, A1, poff=None, period=1.0):
    """sum_h A_h(t) sin(2*pi*ratio_h*phase(t)).  A_h ramps linearly from A0 to A1 over the block
    (the ramp is factored out of the sum, so it costs two mat-vec products).  Inaudible partials
    are skipped, and the sines run in float32 on a phase that was reduced in float64 first."""
    n = len(ph)
    same = A0 is A1
    top = max(float(A0.max()), float(A1.max()))
    keep = np.maximum(A0, A1) > top * 2e-3
    if not keep.all():
        ratios, A0, A1 = ratios[keep], A0[keep], A1[keep]
        if poff is not None:
            poff = poff[keep]
    if period:  # exactly periodic partials: wrap the phase once, everything then fits float32
        phr = ((ph % period) * TWO_PI).astype(np.float32)
        arg = np.outer(ratios.astype(np.float32), phr)
    else:
        arg = np.outer(ratios, ph)
        arg -= np.floor(arg)
        arg *= TWO_PI
        arg = arg.astype(np.float32)
    if poff is not None:
        arg += (poff * TWO_PI).astype(np.float32)[:, None]
    s = np.sin(arg)
    y = A0.astype(np.float32) @ s
    if not same:
        y += _ramp32(n) * ((A1 - A0).astype(np.float32) @ s)
    return y.astype(np.float64)


def _osc(w, ph):
    if w == 0:
        return np.sin(2 * np.pi * ph)
    if w == 1:
        return 2 * ph - 1
    if w == 2:
        return np.where(ph < 0.5, 1.0, -1.0)
    return 4 * np.abs(ph - 0.5) - 1


# ---- vowel formants (Csound formant tables: freq Hz, level dB, bandwidth Hz) for a, e, i, o, u
def _tab(rows):
    return np.array(rows, dtype=np.float64)


_TENOR = [
    _tab([(650, 0, 80), (1080, -6, 90), (2650, -7, 120), (2900, -8, 130), (3250, -22, 140)]),
    _tab([(400, 0, 70), (1700, -14, 80), (2600, -12, 100), (3200, -14, 120), (3580, -20, 120)]),
    _tab([(290, 0, 40), (1870, -15, 90), (2800, -18, 100), (3250, -20, 120), (3540, -30, 120)]),
    _tab([(400, 0, 40), (800, -10, 80), (2600, -12, 100), (2800, -12, 120), (3000, -26, 120)]),
    _tab([(350, 0, 40), (600, -20, 60), (2700, -17, 100), (2900, -14, 120), (3300, -26, 120)]),
]
_ALTO = [
    _tab([(800, 0, 80), (1150, -4, 90), (2800, -20, 120), (3500, -36, 130), (4950, -60, 140)]),
    _tab([(400, 0, 60), (1600, -24, 80), (2700, -30, 120), (3300, -35, 150), (4950, -60, 200)]),
    _tab([(350, 0, 50), (1700, -20, 100), (2700, -30, 120), (3700, -36, 150), (4950, -60, 200)]),
    _tab([(450, 0, 70), (800, -9, 80), (2830, -16, 100), (3500, -28, 130), (4950, -55, 135)]),
    _tab([(325, 0, 50), (700, -12, 60), (2530, -30, 170), (3500, -40, 180), (4950, -64, 200)]),
]
_SOPRANO = [
    _tab([(800, 0, 80), (1150, -6, 90), (2900, -32, 120), (3900, -20, 130), (4950, -50, 140)]),
    _tab([(350, 0, 60), (2000, -20, 100), (2800, -15, 120), (3600, -40, 150), (4950, -56, 200)]),
    _tab([(270, 0, 60), (2140, -12, 90), (2950, -26, 100), (3900, -26, 120), (4950, -44, 120)]),
    _tab([(450, 0, 40), (800, -11, 80), (2830, -22, 100), (3800, -22, 120), (4950, -50, 120)]),
    _tab([(325, 0, 50), (700, -16, 60), (2700, -35, 170), (3800, -40, 180), (4950, -60, 200)]),
]


def _vowel(tab, w):
    w = min(max(float(w), 0.0), 4.0)
    i = int(w)
    j = min(i + 1, 4)
    t = w - i
    return tab[i] * (1 - t) + tab[j] * t


def formants_for(p, w, f0):
    """Formant (freq, gain, bandwidth) arrays for a (possibly morphing) vowel w at pitch p."""
    if p <= 52:
        a = _vowel(_TENOR, w)
    elif p <= 64:
        t = (p - 52) / 12.0
        a = (1 - t) * _vowel(_TENOR, w) + t * _vowel(_ALTO, w)
    elif p <= 76:
        t = (p - 64) / 12.0
        a = (1 - t) * _vowel(_ALTO, w) + t * _vowel(_SOPRANO, w)
    else:
        a = _vowel(_SOPRANO, w)
    F, L, B = a[:, 0].copy(), 10.0 ** (a[:, 1] / 20.0), a[:, 2]
    if f0 > F[0]:  # singers tune the first formant up to follow a high fundamental
        F[0] = min(f0, F[0] * 1.8)
    return F, L, B


# ---- plucked / struck models: return (ratios, base amps, rate1, rate2, w1, w2, noise, noise decay)
def _keep(ratios, fm, *arrs):
    ok = ratios * fm < 0.45 * SR
    if not ok.any():
        ok[0] = True
    return (ratios[ok],) + tuple(a[ok] for a in arrs)


def _m_piano(midi, vel, fm, tone):
    f0 = float(midi_hz(midi))
    B = 3e-4 * 2.0 ** (max(midi - 60, 0) / 10.0)
    hs = np.arange(1, 49, dtype=np.float64)
    ratios = hs * np.sqrt(1 + B * hs * hs)
    fh = ratios * f0
    base = (0.35 + 0.65 * np.abs(np.sin(np.pi * hs * 0.12))) * hs ** -0.8 / (1 + (fh / (2200.0 + 5000.0 * vel)) ** 2)
    base = base * hs ** ((tone - 0.5) * 1.6)
    kp = 2.0 ** ((midi - 60) / 16.0)
    r1, r2 = (1.3 + 0.55 * hs) * kp, (0.22 + 0.05 * hs) * kp
    ratios, base, r1, r2 = _keep(ratios, fm, base, r1, r2)
    return ratios, base, r1, r2, 0.65, 0.35, 0.03 * vel, 80.0


def _m_guitar(midi, vel, fm, tone):
    f0 = float(midi_hz(midi))
    hs = np.arange(1, 41, dtype=np.float64)
    ratios = hs * np.sqrt(1 + 1.0e-4 * hs * hs)
    fh = ratios * f0
    body = (1 + 1.6 * np.exp(-((fh - 105) / 50) ** 2) + 1.2 * np.exp(-((fh - 210) / 80) ** 2)
            + 0.6 * np.exp(-((fh - 420) / 150) ** 2))
    base = (0.25 + 0.75 * np.abs(np.sin(np.pi * hs * 0.17))) * hs ** -0.95 * body / (1 + (fh / (3000 + 2500 * vel)) ** 2)
    base = base * hs ** ((tone - 0.5) * 1.6)
    kp = 2.0 ** ((midi - 52) / 20.0)
    r1, r2 = (1.6 + 0.8 * hs) * kp, (0.7 + 0.06 * hs) * kp
    ratios, base, r1, r2 = _keep(ratios, fm, base, r1, r2)
    return ratios, base, r1, r2, 0.75, 0.25, 0.10 * vel, 140.0


def _m_eguitar(midi, vel, fm, tone):
    f0 = float(midi_hz(midi))
    hs = np.arange(1, 41, dtype=np.float64)
    ratios = hs * np.sqrt(1 + 0.6e-4 * hs * hs)
    fh = ratios * f0
    base = (0.3 + 0.7 * np.abs(np.sin(np.pi * hs * 0.22))) * hs ** -0.7 / (1 + (fh / (2800 + 1500 * vel)) ** 2)
    base = base * hs ** ((tone - 0.5) * 1.6)
    kp = 2.0 ** ((midi - 52) / 28.0)
    r1, r2 = (0.7 + 0.5 * hs) * kp, (0.12 + 0.02 * hs) * kp
    ratios, base, r1, r2 = _keep(ratios, fm, base, r1, r2)
    return ratios, base, r1, r2, 0.45, 0.55, 0.05 * vel, 150.0


def _m_marimba(midi, vel, fm, tone):
    ratios = np.array([1.0, 3.99, 9.93, 17.3])
    amps = np.array([1.0, 0.42 * (0.6 + vel), 0.14 * vel, 0.05 * vel]) * ratios ** ((tone - 0.5) * 1.2)
    kp = 2.0 ** ((midi - 60) / 12.0)
    r1, r2 = np.array([3.5, 11, 22, 36]) * kp, np.array([0.8, 2.5, 5, 8]) * kp
    ratios, base, r1, r2 = _keep(ratios, fm, amps, r1, r2)
    return ratios, base, r1, r2, 0.7, 0.3, 0.04 * vel, 120.0


def fam_pluck(v, ch, sp, ph, f, ts, e, vow, pitch):
    n = len(ph)
    midi = float(np.mean(pitch))
    fm = float(f.max())
    ratios, base, r1, r2, w1, w2, nz, nd = sp["model"](midi, v.vel, fm, ch.tone)
    base = base * (LEVEL / max(_rms(base), 1e-9))
    t0, t1 = float(ts[0]), float(ts[-1])
    A0 = base * (w1 * np.exp(-r1 * t0) + w2 * np.exp(-r2 * t0))
    A1 = base * (w1 * np.exp(-r1 * t1) + w2 * np.exp(-r2 * t1))
    sig = additive(ph, ratios, A0, A1, period=None)
    if nz:
        sig = sig + _noise(n) * nz * LEVEL * np.exp(-ts * nd)
    if t0 > 0.1 and float(A0.max()) < 1.5e-4:
        v.dead = True
    return sig


def fam_ep(v, ch, sp, ph, f, ts, e, vow, pitch):
    """Rhodes-like electric piano: FM tine with a decaying 'bark'."""
    midi = float(np.mean(pitch))
    kp = 2.0 ** ((midi - 60) / 24.0)
    cyc = TWO_PI * ph
    idx = (1.0 + 2.2 * v.vel) * (0.35 + ch.tone) * np.exp(-ts * 3.0 * kp)
    sig = np.sin(cyc + idx * np.sin(cyc))
    if float(f.max()) * 14 < 0.45 * SR:
        sig = sig + 0.5 * v.vel * np.exp(-ts * 18.0) * 0.25 * np.sin(cyc * 14.0)
    amp = 0.6 * np.exp(-ts * 0.9 * kp) + 0.4 * np.exp(-ts * 0.25 * kp)
    if ts[0] > 0.1 and float(amp[0]) < 1.5e-4:
        v.dead = True
    return sig * amp * (LEVEL / 0.75)


def fam_osc(v, ch, sp, ph, f, ts, e, vow, pitch):
    w = ch.wave
    if sp.get("ens"):
        d = ch.detune
        sig = 0.0
        for c, off in ((-d, 0.0), (0.0, 0.33), (d, 0.67)):
            sig = sig + _osc(w, (ph * 2.0 ** (c / 1200.0) + off) % 1.0)
        sig = sig / 3.0
    else:
        sig = _osc(w, ph % 1.0)
        if ch.detune > 0.01:
            sig = (sig + _osc(w, (ph * 2.0 ** (ch.detune / 1200.0)) % 1.0)) * 0.5
    sub = sp.get("sub", 0.0)
    if sub:
        sig = sig * (1 - 0.25 * sub) + sub * np.sin(np.pi * ph)
    return sig


def fam_sustain(v, ch, sp, ph, f, ts, e, vow, pitch):
    """Bowed strings, flute, brass, organ: harmonic spectrum + body formants + breath/bow noise."""
    n = len(ph)
    fm = float(f.max())
    if "partials" in sp:
        hs = np.array(sp["partials"][0], dtype=np.float64)
        base = np.array(sp["partials"][1], dtype=np.float64)
        ok = hs * fm < 0.45 * SR
        if not ok.any():
            return np.zeros(n)
        hs, base = hs[ok], base[ok] * hs[ok] ** ((ch.tone - 0.5) * 0.8)
        A0 = A1 = base * (LEVEL / max(_rms(base), 1e-9))
        per = _period(hs)
    else:
        f0 = float(midi_hz(np.asarray(pitch).reshape(-1)[-1]))
        hs = np.arange(1, _hcount(fm, sp.get("hmax", 32)) + 1, dtype=np.float64)
        fh = hs * f0
        base = hs ** -sp["roll"] * hs ** ((ch.tone - 0.5) * 1.6)
        for fk, g, wd in sp.get("formants", ()):
            base = base * (1 + g * np.exp(-(np.log2(fh / fk) / wd) ** 2))
        base = base * (LEVEL / max(_rms(base), 1e-9))
        kb = sp.get("kb", 0.0) * (1.7 - v.vel)
        A0 = base * np.exp(-kb * (hs - 1) * (1 - min(1.0, float(e[0]))))
        A1 = base * np.exp(-kb * (hs - 1) * (1 - min(1.0, float(e[-1]))))
        per = 1.0
    H = len(hs)
    ens = sp.get("ens", 0)
    if ens:
        sig = np.zeros(n)
        for j, c in enumerate(np.linspace(-1.0, 1.0, ens) * ch.detune):
            sig += additive(ph * 2.0 ** (c / 1200.0) + v.poff2[j, 0], hs, A0, A1, period=per)
        sig /= math.sqrt(ens)
    else:
        sig = additive(ph, hs, A0, A1, v.poff[:H] if sp.get("rphase") else None, period=per)
    if sp.get("noise"):
        sig = sig + _noise(n) * sp["noise"] * LEVEL * 0.5
    if sp.get("chiff"):
        sig = sig + _noise(n) * sp["chiff"] * LEVEL * np.exp(-ts * sp.get("chiffd", 16.0))
    return sig


def fam_choir(v, ch, sp, ph, f, ts, e, vow, pitch):
    """Formant choir: 3 detuned voices whose harmonics are shaped by vowel formants."""
    n = len(ph)
    fm = float(f.max())
    hs = np.arange(1, _hcount(fm, 24) + 1, dtype=np.float64)
    H = len(hs)
    pa = np.asarray(pitch).reshape(-1)
    wa = np.asarray(vow).reshape(-1)
    tone = hs ** ((ch.tone - 0.5) * 1.6)

    def spectrum(p, w):
        f0 = float(midi_hz(p))
        F, L, B = formants_for(float(p), float(w), f0)
        fh = hs * f0
        lor = (L[None, :] / (1 + ((fh[:, None] - F[None, :]) / (1.5 * B[None, :])) ** 2)).sum(axis=1)
        A = hs ** -1.2 * (lor + 0.015) * tone
        return A * (LEVEL / max(_rms(A), 1e-9))

    A0 = spectrum(pa[0], wa[0])
    A1 = spectrum(pa[-1], wa[-1])
    sig = np.zeros(n)
    for j, c in enumerate((-1.0, 0.0, 1.0)):
        pj = ph * 2.0 ** (c * ch.detune / 1200.0) + v.poff2[j, 0]
        shimmer = 1.0 + 0.12 * np.sin(TWO_PI * (0.35 + 0.11 * j) * ts + 2.1 * j)
        sig += additive(pj, hs, A0, A1, v.poff2[j, :H]) * shimmer
    sig /= math.sqrt(3.0)
    return sig + _noise(n) * 0.03 * LEVEL


FAMILIES = {"osc": fam_osc, "pluck": fam_pluck, "ep": fam_ep, "sustain": fam_sustain, "choir": fam_choir}

# ---- instrument registry (name -> category, DSP family, default knob settings)
INSTRUMENTS = {}
INSTRUMENT_CATS = ["Keys", "Guitars", "Strings", "Winds & Brass", "Voice", "Synths"]
_BASE_TIMBRE = dict(wave=1, a=0.01, d=0.25, s=0.55, r=0.2, detune=10.0, cut=1.0, drive=0.0,
                    rev=0.0, dly=0.0, glide=0.06, vib=0.0, tone=0.5)


def _reg(name, cat, fam, short, preset, **spec):
    INSTRUMENTS[name] = dict(cat=cat, fam=fam, short=short, preset=preset, **spec)


_reg("Piano", "Keys", "pluck", "Piano", dict(a=0.001, d=0.002, s=1.0, r=0.28, rev=0.15), model=_m_piano, life=25.0, gain=1.5)
_reg("Electric Piano", "Keys", "ep", "E-Piano", dict(a=0.002, d=0.002, s=1.0, r=0.25, rev=0.12, dly=0.05), life=20.0, gain=1.5)
_reg("Organ", "Keys", "sustain", "Organ", dict(a=0.008, d=0.01, s=1.0, r=0.07, vib=0.45, rev=0.1),
     partials=([0.5, 1, 2, 3, 4, 5, 6, 8], [0.6, 1.0, 0.8, 0.45, 0.5, 0.18, 0.2, 0.1]),
     vibc=15.0, vrate=6.5, vdelay=0.0, vramp=0.01, chiff=0.05, chiffd=60.0)
_reg("Marimba", "Keys", "pluck", "Marimba", dict(a=0.001, d=0.002, s=1.0, r=0.12, rev=0.1), model=_m_marimba, life=8.0, gain=2.0)
_reg("Acoustic Guitar", "Guitars", "pluck", "Guitar", dict(a=0.001, d=0.002, s=1.0, r=0.2, rev=0.12),
     model=_m_guitar, life=12.0, gain=2.0)
_reg("Electric Guitar", "Guitars", "pluck", "E-Guitar",
     dict(a=0.001, d=0.002, s=1.0, r=0.25, drive=0.3, cut=0.8, rev=0.12, dly=0.15), model=_m_eguitar, life=20.0, gain=0.3)
_reg("Violin", "Strings", "sustain", "Violin", dict(a=0.09, d=0.1, s=1.0, r=0.22, vib=0.7, glide=0.07, rev=0.2),
     roll=0.9, hmax=36, formants=[(300, 1.2, 0.35), (1000, 1.5, 0.5), (2800, 1.8, 0.45)], kb=0.05, noise=0.03,
     vibc=28.0, vrate=5.5, vdelay=0.18, vramp=0.3)
_reg("Cello", "Strings", "sustain", "Cello", dict(a=0.12, d=0.1, s=1.0, r=0.3, vib=0.6, glide=0.08, rev=0.2),
     roll=1.0, hmax=40, formants=[(110, 1.0, 0.4), (220, 1.2, 0.4), (550, 1.0, 0.5), (1100, 1.0, 0.5)], kb=0.04,
     noise=0.04, vibc=22.0, vrate=5.0, vdelay=0.2, vramp=0.3)
_reg("String Ensemble", "Strings", "sustain", "Strings",
     dict(a=0.35, d=0.2, s=1.0, r=0.6, vib=0.35, detune=14.0, glide=0.1, rev=0.3),
     roll=0.9, hmax=30, formants=[(300, 1.0, 0.4), (1000, 1.2, 0.5), (2800, 1.4, 0.5)], kb=0.04, noise=0.02,
     ens=3, vibc=22.0, vrate=5.2, vdelay=0.25, vramp=0.4)
_reg("Flute", "Winds & Brass", "sustain", "Flute", dict(a=0.08, d=0.05, s=1.0, r=0.15, vib=0.5, rev=0.2),
     roll=2.0, hmax=14, kb=0.1, noise=0.07, chiff=0.25, chiffd=18.0, rphase=True,
     vibc=18.0, vrate=5.0, vdelay=0.3, vramp=0.5)
_reg("Brass", "Winds & Brass", "sustain", "Brass", dict(a=0.06, d=0.1, s=0.9, r=0.12, vib=0.25, rev=0.12),
     roll=0.55, hmax=24, formants=[(1000, 1.0, 0.7), (2400, 0.8, 0.6)], kb=0.5, noise=0.01,
     vibc=18.0, vrate=5.3, vdelay=0.35, vramp=0.4, scoop=35.0)
_reg("Choir", "Voice", "choir", "Choir", dict(a=0.22, d=0.2, s=1.0, r=0.5, vib=0.55, detune=14.0, glide=0.09, rev=0.35),
     vibc=30.0, vrate=5.4, vdelay=0.35, vramp=0.4)
_reg("Synth", "Synths", "osc", "Synth", dict())
_reg("Synth Bass", "Synths", "osc", "Synth Bass", dict(a=0.005, d=0.25, s=0.6, r=0.1, detune=6.0, cut=0.55, drive=0.1),
     sub=0.7, gain=0.9)
_reg("Synth Pad", "Synths", "osc", "Synth Pad", dict(a=0.7, d=0.5, s=0.8, r=1.2, detune=14.0, cut=0.6, rev=0.3, glide=0.15),
     ens=3, gain=1.6)


def apply_preset(ch, inst):
    if inst not in INSTRUMENTS:
        inst = "Synth"
    ch.inst = inst
    for k, val in {**_BASE_TIMBRE, **INSTRUMENTS[inst]["preset"]}.items():
        setattr(ch, k, val)


def gm_to_inst(prog):
    """Map a General-MIDI program number to one of our instruments."""
    if prog < 8: return "Piano"
    if prog < 16: return "Marimba"
    if prog < 24: return "Organ"
    if prog < 26: return "Acoustic Guitar"
    if prog < 32: return "Electric Guitar"
    if prog < 40: return "Synth Bass"
    if prog < 42: return "Violin"
    if prog < 44: return "Cello"
    if prog < 52: return "String Ensemble"
    if prog < 56: return "Choir"
    if prog < 72: return "Brass"
    if prog < 80: return "Flute"
    if prog < 88: return "Synth"
    if prog < 96: return "Synth Pad"
    return "Synth"


# ============================================================ MODEL
_NOTE_ID = [1]


def new_id():
    _NOTE_ID[0] += 1
    return _NOTE_ID[0]


def _bump_ids(m):
    _NOTE_ID[0] = max(_NOTE_ID[0], int(m))


def mknote(st, ln, pi, ve=0.9, slur=0, vowel=-1):
    """[start, length, pitch, velocity, slur_target_id (0 = none), vowel (-1 = channel default), id, 0]
    A slur links this note to the note whose id is stored in slot 4 (any later note, any distance)."""
    return [float(st), float(ln), int(pi), float(ve), int(slur), int(vowel), new_id(), 0.0]


def mkclip(start, length_steps, offset_sec=0.0, dur_sec=0.0, ve=1.0):
    """Audio clip: [start, length_steps, ROOT, vel, 0, source_offset_sec, id, duration_sec (0 = whole file)]"""
    return [float(start), float(length_steps), ROOT, float(ve), 0, float(offset_sec), new_id(), float(dur_sec)]


def fixnote(n):
    n = list(n)
    while len(n) < 8:
        n.append([0, -1, 0, 0.0][len(n) - 4])
    return n


def build_chains(notes):
    """Group notes into slur chains.  A note whose slot 4 holds the id of a later note is joined to it:
    the chain shares ONE voice, so the envelope is not re-triggered; the pitch glides across the gap."""
    ns = [fixnote(n) for n in notes]
    byid = {n[6]: n for n in ns if n[6]}
    link, claimed = {}, set()
    for n in sorted(ns, key=lambda n: (n[0], n[2])):
        t = n[4]
        if t >= 2 and t in byid and byid[t][0] > n[0] and t not in claimed and byid[t] is not n:
            link[id(n)] = byid[t]
            claimed.add(t)
    chains = []
    for n in sorted(ns, key=lambda n: (n[0], n[2])):
        if n[6] in claimed and n[6]:
            continue
        chain = [n]
        while id(chain[-1]) in link:
            chain.append(link[id(chain[-1])])
        chains.append(chain)
    return chains


def legacy_slur_links(ns):
    """Old projects stored slur as a 0/1 flag meaning 'join the next note'.  Convert to explicit links."""
    ns = sorted(ns, key=lambda n: (n[0], n[2]))
    claimed = set()
    for i, n in enumerate(ns):
        if n[4] != 1:
            continue
        n[4] = 0
        end, j = n[0] + n[1], i + 1
        while j < len(ns) and ns[j][0] <= n[0] + 1e-9:
            j += 1
        if j >= len(ns) or ns[j][0] > end + 1.0 + 1e-9:
            continue
        first, best = ns[j][0], None
        k = j
        while k < len(ns) and ns[k][0] <= first + 1e-9:
            if ns[k][6] not in claimed and (best is None or abs(ns[k][2] - n[2]) < abs(ns[best][2] - n[2])):
                best = k
            k += 1
        if best is not None:
            n[4] = ns[best][6]
            claimed.add(ns[best][6])


class Channel:
    PARAMS = ["name", "kind", "path", "builtin", "color", "wave", "a", "d", "s", "r", "detune",
              "vol", "pan", "cut", "drive", "rev", "dly", "mute", "inst", "glide", "vib", "tone", "vowel", "gate", "uid", "fx"]  # + notes, pnotes
    WAVES = ["Sine", "Saw", "Square", "Triangle"]

    def __init__(self, name, kind="synth", sample=None, path=None, color=None, inst="Synth"):
        self.name, self.kind, self.sample, self.path, self.builtin = name, kind, sample, path, None
        self.color = color or PALETTE[0]
        self.notes = []  # [start_step, length_steps, midi_pitch, velocity, slur, vowel]
        self.vowel = 0
        self.gate = True  # samplers: stop the sample when the note ends (built-in drums are one-shots)
        apply_preset(self, inst)
        self.vol, self.pan, self.mute = 0.8, 0.0, False
        self.zi = np.zeros((2, 1))
        self.peaks = make_peaks(sample) if (kind == "audio" and sample is not None) else None
        self._pk, self._chains, self._starts = None, [], np.zeros(0)
        self.uid, self.fx = new_uid(), default_fx()  # fx: audio-clip effects (Auto-Tune / Harmonizer / Vocoder)
        # audio-clip tracks: "notes for effects" (drawn in the piano roll like a synth) steer Auto-Tune, the
        # Harmonizer and the Vocoder carrier; fx_clips caches the effect-processed audio of every clip.
        self.pnotes, self.notes_mode = [], False
        self.fx_clips, self.fx_status, self.fx_refresh_ui = {}, "", False
        self._voc, self._carrier, self._carrier_prev = None, None, None

    @property
    def fx_busy(self):
        return any(e["busy"] for e in self.fx_clips.values())

    @property
    def spec(self):
        return INSTRUMENTS.get(self.inst, INSTRUMENTS["Synth"])

    def set_audio(self, data, path):
        self.sample, self.path, self.builtin = data, path, None
        self.peaks = make_peaks(data) if self.kind == "audio" else None
        self.fx_clips.clear()

    def clip_steps(self, bpm):
        """Length of the whole audio file in steps."""
        if self.sample is None:
            return 1.0
        return self.sample.shape[1] / SR / (60.0 / bpm / 4)

    def clip_offset(self, n):
        """Seconds skipped at the start of the file (left-trim) for audio clip note n."""
        return max(0.0, float(n[5]))

    def clip_dur(self, n):
        """Playing length in seconds of audio clip note n (after trimming)."""
        total = self.sample.shape[1] / SR if self.sample is not None else 0.0
        off = min(self.clip_offset(n), max(0.0, total - 0.01))
        d = n[7] if n[7] > 0 else total - off
        return max(0.01, min(d, total - off)) if total else 0.01

    def clip_steps_of(self, n, bpm):
        return self.clip_dur(n) / (60.0 / bpm / 4)

    def to_dict(self):
        d = {k: getattr(self, k) for k in self.PARAMS}
        d["notes"] = self.notes
        d["pnotes"] = self.pnotes
        return d

    @staticmethod
    def from_dict(d):
        ch = Channel(d.get("name", "Channel"), d.get("kind", "synth"))
        for k in Channel.PARAMS:
            if k in d:
                setattr(ch, k, d[k])
        if ch.inst not in INSTRUMENTS:
            ch.inst = "Synth"
        if "gate" not in d and ch.builtin:
            ch.gate = False
        ch.fx = merge_fx(d.get("fx"))
        _UID[0] = max(_UID[0], int(ch.uid))
        ch.notes = [fixnote(n) for n in d.get("notes", [])]
        ch.pnotes = [fixnote(n) for n in d.get("pnotes", [])]
        _bump_ids(max([n[6] for n in ch.notes + ch.pnotes] + [1]))
        for n in ch.notes + ch.pnotes:
            if not n[6]:
                n[6] = new_id()
        if ch.kind == "synth" and any(n[4] == 1 for n in ch.notes):
            legacy_slur_links(ch.notes)
        if ch.kind in ("sample", "audio"):
            if ch.builtin and ch.builtin in make_drums():
                ch.sample = make_drums()[ch.builtin]
            elif ch.path and os.path.exists(ch.path):
                try:
                    ch.set_audio(load_sample(ch.path), ch.path)
                except Exception:
                    traceback.print_exc()
        return ch


class Project:
    def __init__(self):
        self.channels, self.bpm, self.length, self.master = [], 130.0, 32, 0.8

    def next_color(self):
        return PALETTE[len(self.channels) % len(PALETTE)]

    def add_instrument(self, inst="Synth", name=None):
        spec = INSTRUMENTS.get(inst, INSTRUMENTS["Synth"])
        label = name or (spec["short"] if inst != "Synth" else f"Synth {len(self.channels) + 1}")
        ch = Channel(label, "synth", color=self.next_color(), inst=inst)
        self.channels.append(ch)
        return ch

    def add_synth(self, name=None):
        return self.add_instrument("Synth", name)

    def add_sample(self, path, data=None):
        data = load_sample(path) if data is None else data
        ch = Channel(os.path.splitext(os.path.basename(path))[0][:14], "sample", data, path, self.next_color())
        self.channels.append(ch)
        return ch

    def add_audio(self, path, data, start_step=0.0):
        """Audio clip track: the file plays at its original speed from start_step."""
        ch = Channel(os.path.splitext(os.path.basename(path))[0][:14], "audio", data, path, self.next_color())
        ch.notes = [mkclip(start_step, ch.clip_steps(self.bpm))]
        self.channels.append(ch)
        return ch

    def to_json(self):
        return {"bpm": self.bpm, "length": self.length, "master": self.master,
                "channels": [c.to_dict() for c in self.channels]}

    def load_json(self, d):
        self.bpm, self.length, self.master = d.get("bpm", 130), d.get("length", 32), d.get("master", 0.8)
        self.channels = [Channel.from_dict(c) for c in d.get("channels", [])]


def default_project(p):
    dr = make_drums()
    for nm, key, vol, rev in [("Kick", "kick", 0.9, 0.0), ("Clap", "clap", 0.6, 0.25),
                              ("Hat", "hat", 0.45, 0.05), ("Snare", "snare", 0.7, 0.15)]:
        c = Channel(nm, "sample", dr[key], color=p.next_color())
        c.builtin, c.vol, c.rev, c.gate = key, vol, rev, False
        p.channels.append(c)
    N = lambda st, ln, pi, ve=0.9: mknote(st, ln, pi, ve)
    p.channels[0].notes = [N(i, 1, ROOT) for i in range(0, 32, 4)]
    p.channels[1].notes = [N(i, 1, ROOT) for i in range(4, 32, 8)]
    p.channels[2].notes = [N(i, 1, ROOT, 0.6) for i in range(2, 32, 4)]
    p.channels[3].notes = [N(i, 1, ROOT, 0.5) for i in (7, 15, 23, 31)]
    bass = p.add_instrument("Synth Bass", "Bass")
    bass.cut, bass.r, bass.vol, bass.detune = 0.55, 0.1, 0.85, 6.0
    bass.notes = [N(s, l, pi) for s, l, pi in [(0, 2, 45), (3, 1, 45), (4, 2, 45), (7, 1, 48), (8, 2, 43), (11, 1, 43),
                                              (12, 2, 43), (14, 2, 48), (16, 2, 41), (19, 1, 41), (20, 2, 41), (23, 1, 45),
                                              (24, 2, 43), (27, 1, 43), (28, 2, 43), (30, 2, 50)]]
    lead = p.add_synth("Lead")
    lead.wave, lead.cut, lead.rev, lead.dly, lead.vol, lead.d, lead.s = 2, 0.8, 0.35, 0.3, 0.5, 0.15, 0.3
    lead.notes = [N(s, 1, pi, 0.8) for s, pi in zip(range(0, 32, 2), [69, 72, 76, 72, 67, 71, 74, 71, 65, 69, 72, 69, 67, 71, 74, 79])]


def import_midi(path, project):
    mid = mido.MidiFile(path)
    spt = 4.0 / mid.ticks_per_beat
    bpm, found, maxend = None, [], 0.0
    for ti, tr in enumerate(mid.tracks):
        t, on, notes, name, prog = 0, {}, [], None, None
        for m in tr:
            t += m.time
            if m.is_meta:
                if m.type == "set_tempo" and bpm is None:
                    bpm = mido.tempo2bpm(m.tempo)
                elif m.type == "track_name" and not name:
                    name = m.name
            elif getattr(m, "channel", -1) == 9:
                continue  # GM drums are skipped (they'd sound wrong on a synth)
            elif m.type == "program_change" and prog is None:
                prog = m.program
            elif m.type == "note_on" and m.velocity > 0:
                on.setdefault((m.channel, m.note), []).append((t, m.velocity))
            elif m.type in ("note_off", "note_on"):
                lst = on.get((m.channel, m.note))
                if lst:
                    s, v = lst.pop(0)
                    notes.append(mknote(s * spt, max(0.25, (t - s) * spt), m.note, v / 127.0))
        if notes:
            found.append((name, prog, ti, notes))
            maxend = max(maxend, max(n[0] + n[1] for n in notes))
    if bpm:
        project.bpm = round(bpm, 2)
    project.length = int(min(MAX_LEN, max(16, math.ceil(maxend / 16) * 16)))
    for name, prog, ti, notes in found[:16]:
        inst = gm_to_inst(prog) if prog is not None else "Synth"
        ch = project.add_instrument(inst, (name or f"Track {ti}")[:14])
        ch.vol = 0.55
        ch.notes = [n for n in notes if n[0] < project.length]
    return len(found)


def export_midi(path, project):
    mid = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(project.bpm)))
    mid.tracks.append(meta)
    for i, ch in enumerate(project.channels):
        if not ch.notes or ch.kind == "audio":
            continue
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=ch.name))
        ev = []
        for n in ch.notes:
            st, ln, pi, ve = n[:4]
            pi = int(min(127, max(0, pi)))
            ev.append((int(st * 120), 1, pi, ve))
            ev.append((int((st + ln) * 120), 0, pi, ve))
        ev.sort(key=lambda e: (e[0], e[1]))
        last = 0
        for t, on, pi, ve in ev:
            tr.append(mido.Message("note_on" if on else "note_off", note=pi, channel=i % 9,
                                   velocity=max(1, int(ve * 127)) if on else 0, time=t - last))
            last = t
        mid.tracks.append(tr)
    mid.save(path)


# ============================================================ ENGINE
class Ring:
    def __init__(self, d):
        self.d, self.buf, self.i = d, np.zeros(d), 0

    def idx(self, n):
        ix = (self.i + np.arange(n)) % self.d
        self.i = (self.i + n) % self.d
        return ix


class Voice:
    def __init__(self, ch, pitch, vel, delay=0, hold=float("inf"), segs=None, live=False, vowel=None):
        self.ch, self.pitch, self.vel, self.age, self.hold, self.dead = ch, pitch, vel, -int(delay), hold, False
        self.live, self.ph = live, 0.0
        self.off, self.dur, self.data = 0, 1 << 60, None  # audio clips: samples skipped / samples to play / processed audio
        w = ch.vowel if (vowel is None or vowel < 0) else vowel
        # segs: (start_sample, target_pitch, pitch_at_start, target_vowel, vowel_at_start, ramp_samples)
        # ramp_samples > 0: smooth glide over exactly that span (a bent slur band); 0: quick exponential glide
        self.segs = segs or [(0.0, float(pitch), float(pitch), float(w), float(w), 0.0)]
        if len(self.segs) > 1:
            a = np.array(self.segs, dtype=np.float64)
            self.ts, self.P, self.S, self.W, self.U, self.D = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
        if ch.kind == "synth":
            self.poff = _RNG.random(64)
            self.poff2 = _RNG.random((3, 64))

    def curve(self, tt, tau):
        if len(self.segs) == 1:
            s = self.segs[0]
            return s[1], s[3]
        idx = np.clip(np.searchsorted(self.ts, tt, side="right") - 1, 0, len(self.ts) - 1)
        dt = np.maximum(tt - self.ts[idx], 0.0)
        u = np.clip(dt / np.maximum(self.D[idx], 1.0), 0.0, 1.0)
        x = np.where(self.D[idx] > 0, u * u * (3 - 2 * u), 1 - np.exp(-dt / tau))
        return self.S[idx] + (self.P[idx] - self.S[idx]) * x, self.U[idx] + (self.W[idx] - self.U[idx]) * x


class Engine:
    MAXV = 64

    def __init__(self, project):
        self.p, self.playing, self.pos, self.voices, self.peak = project, False, 0, [], 0.0
        self.lock = threading.Lock()
        self.combs = [(Ring(d), 0.84) for d in (1557, 1617, 1491, 1422)]
        self.aps = [Ring(1051), Ring(773)]
        self.echo = Ring(20000)

    def step_samples(self):
        return 60.0 / self.p.bpm / 4 * SR

    def loop_len(self):
        return max(1, int(self.p.length * self.step_samples()))

    # ---- control (GUI thread)
    def note_on(self, ch, pitch, vel=0.9, hold=float("inf")):
        with self.lock:
            self.voices.append(Voice(ch, pitch, vel, 0, hold, live=True))

    def note_off(self, ch, pitch):
        with self.lock:
            for v in self.voices:
                if v.ch is ch and v.pitch == pitch and v.hold == float("inf"):
                    v.hold = max(v.age, 0)

    def _cut(self):
        """Silence everything the sequencer started (live keyboard notes keep ringing)."""
        keep = []
        for v in self.voices:
            if v.live:
                keep.append(v)
            elif (v.ch.kind == "synth" or (v.ch.kind == "sample" and v.ch.gate)) and v.age >= 0:
                v.hold = min(v.hold, v.age)
                keep.append(v)
        self.voices = keep

    def silence(self):
        """Stop everything that is sounding, including notes started from the keyboard and previews."""
        with self.lock:
            keep = []
            for v in self.voices:
                if v.ch.kind == "synth" or (v.ch.kind == "sample" and v.ch.gate):
                    v.hold = min(v.hold, max(v.age, 0))
                    keep.append(v)
            self.voices = keep

    def stop_previews(self):
        with self.lock:
            self.voices = [v for v in self.voices if not getattr(v.ch, "is_preview", False)]

    def _audio_voice(self, ch, n, delay):
        v = Voice(ch, ROOT, n[3], delay)
        ent = ch.fx_clips.get(n[6])
        if ent is not None and ent["data"] is not None and ent["key"] == ent["want"]:
            v.data, v.off, v.dur = ent["data"], 0, ent["data"].shape[1]  # the effect-processed copy of exactly this clip
        else:
            v.off, v.dur = int(ch.clip_offset(n) * SR), int(ch.clip_dur(n) * SR)
        return v

    # ---- hidden synth that the vocoder uses as carrier when it is played by the track's own notes
    def carrier(self, ch, preview=False):
        attr = "_carrier_prev" if preview else "_carrier"
        cc, inst = getattr(ch, attr), ch.fx["voc"].get("inst", "Synth Pad")
        if cc is None:
            cc = Channel("carrier", "synth", inst=inst)
            cc.rev = cc.dly = 0.0
            setattr(ch, attr, cc)
        if cc.inst != inst:
            apply_preset(cc, inst)
            cc.rev = cc.dly = 0.0
        cc.notes = ch.pnotes
        return cc

    def _sched_channels(self):
        out = list(self.p.channels)
        for c in self.p.channels:
            v = c.fx["voc"]
            if c.kind == "audio" and not c.mute and v["on"] and v["carrier"] == "notes" and c.pnotes:
                out.append(self.carrier(c))
        return out

    def _prime_audio(self):
        """Start audio clips that are already sounding at the play bar (so Play can begin mid-clip)."""
        ss = self.step_samples()
        for ch in self.p.channels:
            if ch.kind != "audio" or ch.mute or ch.sample is None:
                continue
            for n in ch.notes:
                s = int(n[0] * ss)
                if s < self.pos < s + int(ch.clip_dur(n) * SR):
                    v = self._audio_voice(ch, n, 0)
                    v.age = self.pos - s
                    self.voices.append(v)

    def play(self):
        with self.lock:
            if self.pos >= self.loop_len():
                self.pos = 0
            self._cut()
            self._prime_audio()
            self.playing = True

    def pause(self):
        with self.lock:
            self.playing = False
            self._cut()

    def seek(self, pos):
        with self.lock:
            self.pos = int(min(max(pos, 0), self.loop_len() - 1))
            self._cut()
            if self.playing:
                self._prime_audio()

    # ---- DSP (audio thread)
    def _plan(self, ch):
        key = hash(tuple(tuple(n) for n in ch.notes))
        if ch._pk != key:
            if ch.kind == "synth":
                chains = build_chains(ch.notes)
            else:
                chains = [[fixnote(n)] for n in sorted(ch.notes)]
            ch._chains = chains
            ch._starts = np.array([c[0][0] for c in chains], dtype=np.float64)
            ch._pk = key
        return ch._starts, ch._chains

    @staticmethod
    def _seg_at(seg, t, tau):
        """(pitch, vowel) of a segment at sample time t."""
        t0, p, sp_, w, sw, d = seg
        dt = max(0.0, t - t0)
        if d > 0:
            u = min(dt / d, 1.0)
            x = u * u * (3 - 2 * u)
        else:
            x = 1 - math.exp(-dt / tau)
        return sp_ + (p - sp_) * x, sw + (w - sw) * x

    def _segs(self, ch, chain, ss):
        tau = max(ch.glide, 0.003) * SR / 3.0
        base = chain[0][0]
        segs, prev = [], None
        for k, n in enumerate(chain):
            tb = (n[0] - base) * ss
            p = float(n[2])
            w = float(n[5] if n[5] >= 0 else ch.vowel)
            if prev is None:
                seg = (tb, p, p, w, w, 0.0)
            else:
                a = chain[k - 1]
                t_end = (a[0] + a[1] - base) * ss
                if tb - t_end > 1.0:  # a gap: glide across it, landing exactly on the next note (the bent band)
                    t0 = max(t_end, prev[0])
                    sp_, sw = self._seg_at(prev, t0, tau)
                    seg = (t0, p, sp_, w, sw, tb - t0)
                else:  # notes touch: quick glide as the next note starts
                    sp_, sw = self._seg_at(prev, tb, tau)
                    seg = (tb, p, sp_, w, sw, 0.0)
            segs.append(seg)
            prev = seg
        return segs

    def _schedule(self, pos, m, off, ss):
        for ch in self._sched_channels():
            if ch.mute or not ch.notes:
                continue
            starts, chains = self._plan(ch)
            if not len(starts):
                continue
            si = (starts * ss).astype(np.int64)
            for i in np.nonzero((si >= pos) & (si < pos + m))[0]:
                chain = chains[i]
                first = chain[0]
                delay = off + int(si[i]) - pos
                if ch.kind == "audio":
                    v = self._audio_voice(ch, first, delay)
                elif len(chain) == 1:
                    v = Voice(ch, first[2], first[3], delay, first[1] * ss, vowel=first[5])
                else:
                    hold = (chain[-1][0] + chain[-1][1] - first[0]) * ss
                    v = Voice(ch, first[2], first[3], delay, hold, segs=self._segs(ch, chain, ss))
                self.voices.append(v)

    def _synth(self, v, buf, n):
        ch = v.ch
        sp = ch.spec
        tt0 = v.age + np.arange(n, dtype=np.float64)
        live = tt0 >= 0
        if not live.any():
            return
        tt = np.maximum(tt0, 0)
        A, D, S, R = max(ch.a, 0.002) * SR, max(ch.d, 0.002) * SR, ch.s, max(ch.r, 0.01) * SR
        held = lambda x: np.where(x < A, x / A, np.where(x < A + D, 1 - (1 - S) * (x - A) / D, S))
        e = held(tt)
        if v.hold != float("inf"):
            lvl = float(held(np.array([float(v.hold)]))[0])
            e = np.where(tt < v.hold, e, lvl * np.clip(1 - (tt - v.hold) / R, 0, 1))
            if v.age + n >= v.hold + R:
                v.dead = True
        if sp.get("life") and v.age > sp["life"] * SR:
            v.dead = True
        ts = tt / SR
        pitch, vow = v.curve(tt, max(ch.glide, 0.003) * SR / 3.0)
        f = midi_hz(pitch)
        vd = ch.vib * sp.get("vibc", 25.0)
        if vd > 0.01:
            ramp = np.clip((ts - sp.get("vdelay", 0.15)) / max(sp.get("vramp", 0.3), 0.01), 0.0, 1.0)
            f = f * np.exp2(vd * ramp * np.sin(TWO_PI * sp.get("vrate", 5.5) * ts) / 1200.0)
        if sp.get("scoop"):
            f = f * np.exp2(-sp["scoop"] * np.exp(-ts / 0.05) / 1200.0)
        f = f * live
        ph = v.ph + np.cumsum(f) / SR
        v.ph = float(ph[-1])
        pitch_a = np.broadcast_to(np.asarray(pitch, dtype=np.float64), (n,))
        sig = FAMILIES[sp["fam"]](v, ch, sp, ph, f, ts, e, vow, pitch_a)
        sig = sig * e * live * v.vel * (0.4 * sp.get("gain", 1.0))
        buf[0] += sig
        buf[1] += sig

    def _sample(self, v, buf, n):
        d = v.data if v.data is not None else v.ch.sample
        if d is None:
            v.dead = True
            return
        audio = v.ch.kind == "audio"
        gated = (not audio) and v.ch.gate  # sampler note lasts exactly as long as the note
        rate = 1.0 if audio else 2.0 ** ((v.pitch - ROOT) / 12)
        rel = v.age + np.arange(n, dtype=np.float64)  # samples since the note/clip started
        idx = rel * rate + (v.off if audio else 0)
        N = d.shape[1]
        ok = (rel >= 0) & (idx < N - 1)
        if audio:
            ok &= rel < v.dur
        if ok.any():
            i = idx[ok]
            i0 = i.astype(np.int64)
            fr = (i - i0).astype(np.float32)
            out = (d[:, i0] * (1 - fr) + d[:, i0 + 1] * fr) * v.vel
            if audio:  # ~4 ms fades so trimmed clips never click
                r = rel[ok]
                fade = np.minimum(1.0, np.minimum(r / 176.0, (v.dur - r) / 176.0)).clip(0.0, 1.0).astype(np.float32)
                out = out * fade
            if gated and v.hold != float("inf"):  # ~10 ms fade-out starting where the note ends
                out = out * np.clip(1.0 - (rel[ok] - v.hold) / 441.0, 0.0, 1.0).astype(np.float32)
            buf[:, ok] += out
        if audio:
            if v.age + n >= v.dur or idx[-1] >= N - 1:
                v.dead = True
        else:
            if (v.age + n) * rate >= N - 1:
                v.dead = True
            if gated and v.age + n >= v.hold + 441:
                v.dead = True

    # ---- vocoder (audio-clip track): the clip is the modulator, the carrier is a chord drone, noise, or another track
    def _voc_init(self, key):
        bands, smooth = key
        edges = np.geomspace(110.0, 9500.0, bands + 1)
        sos = [butter(2, [edges[i], edges[i + 1]], btype="bandpass", fs=SR, output="sos") for i in range(bands)]
        a = 1.0 - math.exp(-1.0 / (SR * max(smooth, 0.0005)))
        hp = butter(2, 5500.0, btype="highpass", fs=SR, output="sos")
        return dict(key=key, sos=sos, zm=[np.zeros((s.shape[0], 2)) for s in sos], zc=[np.zeros((s.shape[0], 2)) for s in sos],
                    ze=[np.zeros((2, 1)) for _ in sos], eb=np.array([a]), ea=np.array([1.0, -(1.0 - a)]),
                    hp=hp, zh=np.zeros((hp.shape[0], 2)), ph=np.zeros(0))

    def _drone(self, st, cfg, n):
        notes = [48 + int(cfg["root"]) + i for i in CHORDS.get(cfg["chord"], CHORDS["Major"])]
        f = np.array([midi_hz(m) * 2.0 ** (d / 1200.0) for m in notes for d in (-6.0, 6.0)])
        if len(st["ph"]) != len(f):
            st["ph"] = np.zeros(len(f))
        ph = st["ph"][:, None] + f[:, None] * (np.arange(1, n + 1) / SR)[None, :]
        st["ph"] = ph[:, -1] % 1.0
        return (2.0 * (ph % 1.0) - 1.0).sum(axis=0) * (0.3 / len(f))

    def _vocode(self, ch, buf, car):
        cfg, n = ch.fx["voc"], buf.shape[1]
        key = (int(cfg["bands"]), round(float(cfg["smooth"]), 4))
        st = ch._voc
        if st is None or st["key"] != key:
            st = ch._voc = self._voc_init(key)
        mod = buf.mean(axis=0)
        if cfg["carrier"] in ("track", "notes"):  # another track / this track's own notes: silence when it plays nothing
            c = car.mean(axis=0) if car is not None else np.zeros(n)
        elif cfg["carrier"] == "noise":
            c = _RNG.standard_normal(n) * 0.3
        else:
            c = self._drone(st, cfg, n)
        out = np.zeros(n)
        for b, sos in enumerate(st["sos"]):
            fm, st["zm"][b] = sosfilt(sos, mod, zi=st["zm"][b])
            fc, st["zc"][b] = sosfilt(sos, c, zi=st["zc"][b])
            env, st["ze"][b] = lfilter(st["eb"], st["ea"], np.abs(np.vstack([fm, fc])), axis=1, zi=st["ze"][b])
            out += fc * env[0] / (env[1] + 0.002)  # carrier band, normalised, shaped by the voice's band energy
        hf, st["zh"] = sosfilt(st["hp"], mod, zi=st["zh"])
        out = (out * 2.0 + hf * float(cfg["sib"])) * float(cfg["gain"])  # x2 makeup: gain 1.0 ~ the voice's own level
        mix = float(cfg["mix"])
        return buf * (1.0 - mix) + out[None, :] * mix

    def _reverb(self, x):
        n, y = len(x), np.zeros(len(x))
        for r, g in self.combs:
            ix = r.idx(n)
            d = r.buf[ix]
            r.buf[ix] = x + g * d
            y += d
        y *= 0.12
        for r in self.aps:
            ix = r.idx(n)
            d = r.buf[ix]
            w = y + 0.5 * d
            r.buf[ix] = w
            y = -0.5 * w + d
        return y

    def _echo(self, x):
        d = max(BLOCK + 1, int(60.0 / self.p.bpm * 0.75 * SR))
        if d != self.echo.d:
            self.echo = Ring(d)
        ix = self.echo.idx(len(x))
        y = self.echo.buf[ix]
        self.echo.buf[ix] = x + 0.45 * y
        return y

    def render(self, n):
        p = self.p
        ss = self.step_samples()
        loop = max(1, int(p.length * ss))
        bufs = {}
        with self.lock:
            if self.playing:
                if self.pos >= loop:
                    self.pos = 0
                pos, off, left = self.pos, 0, n
                while left > 0:
                    m = min(left, loop - pos)
                    self._schedule(pos, m, off, ss)
                    pos, off, left = pos + m, off + m, left - m
                    if pos >= loop:
                        pos = 0
                self.pos = pos
            if len(self.voices) > self.MAXV:
                del self.voices[:len(self.voices) - self.MAXV]
            live = []
            for v in self.voices:
                buf = bufs.get(v.ch)
                if buf is None:
                    buf = bufs[v.ch] = np.zeros((2, n))
                (self._synth if v.ch.kind == "synth" else self._sample)(v, buf, n)
                v.age += n
                if not v.dead:
                    live.append(v)
            self.voices = live
        hidden, by_uid = set(), {c.uid: c for c in p.channels}
        for ch in p.channels:  # the vocoder's carrier is not heard directly (unless "hide" is off)
            vc = ch.fx["voc"]
            if ch.kind != "audio" or ch.mute or not vc["on"] or not vc["hide"]:
                continue
            if vc["carrier"] == "track" and vc["carrier_uid"] in by_uid:
                hidden.add(by_uid[vc["carrier_uid"]])
            elif vc["carrier"] == "notes":
                hidden.add(self.carrier(ch))
        for ch in list(bufs):
            if ch.kind == "audio" and not ch.mute and ch.fx["voc"]["on"]:
                cfg = ch.fx["voc"]
                car = by_uid.get(cfg["carrier_uid"]) if cfg["carrier"] == "track" else (
                    self.carrier(ch) if cfg["carrier"] == "notes" else None)
                bufs[ch] = self._vocode(ch, bufs[ch], bufs.get(car) if car is not None else None)
        out, rev_in, dly_in = np.zeros((2, n)), np.zeros(n), np.zeros(n)
        for ch, buf in bufs.items():
            if ch.mute or ch in hidden:
                continue
            if ch.cut < 0.995:
                a = 1 - math.exp(-2 * math.pi * 20.0 * 1000.0 ** ch.cut / SR)
                buf, ch.zi = lfilter([a], [1, -(1 - a)], buf, axis=1, zi=ch.zi)
            if ch.drive > 0.01:
                g = 1 + ch.drive * 25
                buf = np.tanh(buf * g) / math.tanh(g)
            ang = (ch.pan + 1) * math.pi / 4
            g = ch.vol ** 2 * 1.414
            buf = buf * np.array([[math.cos(ang) * g], [math.sin(ang) * g]])
            mono = (buf[0] + buf[1]) * 0.5
            rev_in += mono * ch.rev
            dly_in += mono * ch.dly
            out += buf
        out += self._reverb(rev_in) + self._echo(dly_in)
        out *= p.master
        np.tanh(out, out)
        self.peak = max(float(np.abs(out).max()), self.peak * 0.9)
        return out

    def render_offline(self):
        """Render one loop plus a reverb/release tail (call on a fresh Engine)."""
        self.voices, self.pos, self.playing = [], 0, True
        for c in self.p.channels:
            c._voc = None  # vocoder filters start from silence
        ss = self.step_samples()
        total, chunks = int(self.p.length * ss), []
        done = 0
        while done < total:
            m = min(BLOCK, total - done)
            chunks.append(self.render(m))
            done += m
        self.playing = False
        for _ in range(int(2.5 * SR / BLOCK)):
            chunks.append(self.render(BLOCK))
        return np.concatenate(chunks, axis=1)


class Audio:
    def __init__(self, engine):
        self.eng, self.stream = engine, None
        if sd is None:
            print("sounddevice not available - no audio output.")
            return
        try:
            self.stream = sd.OutputStream(samplerate=SR, channels=2, blocksize=STREAM_BLOCK, dtype="float32", callback=self.cb)
            self.stream.start()
        except Exception as e:
            self.stream = None
            print("Could not open audio device:", e)

    def cb(self, outdata, frames, t, status):
        try:
            pos = 0
            while pos < frames:
                m = min(BLOCK, frames - pos)
                outdata[pos:pos + m] = self.eng.render(m).T
                pos += m
        except Exception:
            outdata.fill(0)
            traceback.print_exc()

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


# ============================================================ HUM -> NOTES (pitch tracking)
def yin_frame(x, sr, fmin=65.0, fmax=1100.0, thr=0.15):
    """YIN pitch estimate for one frame (len = 2W).  Returns (f0_hz, confidence) or (0, 0)."""
    n = len(x)
    W = n // 2
    tmax = min(int(sr / fmin), W - 1)
    tmin = max(2, int(sr / fmax))
    x = x - x.mean()
    N = 1 << (n + W).bit_length()
    cc = np.fft.irfft(np.conj(np.fft.rfft(x[:W], N)) * np.fft.rfft(x, N), N)[:tmax + 1]
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    taus = np.arange(tmax + 1)
    d = (c2[W] - c2[0]) + (c2[W + taus] - c2[taus]) - 2 * cc
    d[0] = 0.0
    cm = np.ones(tmax + 1)
    cm[1:] = d[1:] * np.arange(1, tmax + 1) / np.maximum(np.cumsum(d[1:]), 1e-12)
    below = np.nonzero(cm[tmin:tmax] < thr)[0]
    if len(below):
        t = tmin + int(below[0])
        while t + 1 < tmax and cm[t + 1] < cm[t]:
            t += 1
    else:
        t = tmin + int(np.argmin(cm[tmin:tmax]))
        if cm[t] > 0.35:
            return 0.0, 0.0
    tf = float(t)
    if 1 <= t < tmax:
        s0, s1, s2 = cm[t - 1], cm[t], cm[t + 1]
        den = s0 - 2 * s1 + s2
        if abs(den) > 1e-12:
            tf = t + 0.5 * (s0 - s2) / den
    return sr / tf, float(1.0 - cm[t])


def hz_to_midi(f):
    return 69.0 + 12.0 * math.log2(f / 440.0)


def midi_name(m):
    names = "C C# D D# E F F# G G# A A# B".split()
    mi = int(round(m))
    return f"{names[mi % 12]}{mi // 12 - 1}"


def hum_to_events(audio, sr, min_dur=0.06):
    """Turn a mono recording into [(t_start, t_end, midi_pitch, loudness 0..1)] in seconds."""
    audio = np.asarray(audio, dtype=np.float64)
    n = 2048 if sr <= 50000 else 4096
    hop = n // 4
    if len(audio) < n:
        return []
    nfr = 1 + (len(audio) - n) // hop
    frames = np.lib.stride_tricks.sliding_window_view(audio, n)[::hop][:nfr]
    rms = np.sqrt(np.einsum("ij,ij->i", frames, frames) / n)
    srt = np.sort(rms)
    floor, peak = srt[int(0.1 * (nfr - 1))], srt[int(0.95 * (nfr - 1))]
    if peak < 0.004:
        return []
    gate = max(0.003, floor * 2.5, peak * 0.12)
    track = np.full(nfr, np.nan)
    for i in range(nfr):
        if rms[i] < gate:
            continue
        f0, conf = yin_frame(frames[i], sr)
        if f0 > 0 and conf >= 0.65:
            track[i] = hz_to_midi(f0)
    # octave-jump repair against the local median
    fixed = track.copy()
    for i in np.nonzero(~np.isnan(track))[0]:
        win = track[max(0, i - 4):i + 5]
        med = float(np.nanmedian(win))
        d = track[i] - med
        if abs(abs(d) - 12) < 1.0:
            fixed[i] = track[i] - 12 * np.sign(d)
    # light median smoothing
    smooth = fixed.copy()
    for i in np.nonzero(~np.isnan(fixed))[0]:
        smooth[i] = np.nanmedian(fixed[max(0, i - 2):i + 3])
    # segmentation into notes
    half = n / 2.0
    notes, cur, pend, gap = [], None, [], 0

    def close(c, end_i):
        notes.append((c["i0"], end_i, float(np.median(c["p"])), float(np.median(rms[c["i0"]:end_i]))))

    for i in range(nfr):
        m = smooth[i]
        if np.isnan(m):
            gap += 1
            if cur is not None and gap >= 3:
                close(cur, cur["last"] + 1)
                cur, pend = None, []
            continue
        gap = 0
        if cur is None:
            cur, pend = {"i0": i, "p": [m], "r": [rms[i]], "last": i, "dip": None}, []
            continue
        ref = float(np.median(cur["p"]))
        if abs(m - ref) <= 0.75:
            r = rms[i]
            if len(cur["r"]) >= 3:  # loudness dip then recovery = the same pitch sung again
                rr = float(np.median(cur["r"]))
                if r < 0.5 * rr:
                    if cur["dip"] is None:
                        cur["dip"] = i
                    cur["last"] = i
                    continue
                if cur["dip"] is not None and i - cur["dip"] >= 2 and r > 0.7 * rr:
                    close(cur, cur["dip"])
                    cur, pend = {"i0": i, "p": [m], "r": [r], "last": i, "dip": None}, []
                    continue
                cur["dip"] = None
            cur["p"].append(m)
            cur["r"].append(r)
            cur["last"] = i
            pend = []
        else:
            pend.append((i, m))
            vals = [q[1] for q in pend]
            if len(pend) >= 4 and max(vals) - min(vals) < 1.0:
                close(cur, pend[0][0])
                cur, pend = {"i0": pend[0][0], "p": vals, "r": [rms[i]], "last": i, "dip": None}, []
    if cur is not None:
        close(cur, cur["last"] + 1)
    ref_loud = max(peak, 1e-9)
    ev = []
    for i0, i1, p, loud in notes:
        ta = (i0 * hop + half - hop / 2.0) / sr
        tb = ((i1 - 1) * hop + half + hop / 2.0) / sr
        ev.append([max(0.0, ta), tb, int(round(p)), min(1.0, loud / ref_loud)])
    merged = []
    for e in ev:  # glue repeated same-pitch fragments separated by a tiny gap
        if merged and merged[-1][2] == e[2] and e[0] - merged[-1][1] < 0.06:
            merged[-1][1] = e[1]
            merged[-1][3] = max(merged[-1][3], e[3])
        else:
            merged.append(e)
    return [tuple(e) for e in merged if e[1] - e[0] >= min_dur]


# ============================================================ AUDIO-CLIP EFFECTS
# Auto-Tune and Auto-Harmonizer re-synthesise the clip with PSOLA (pitch-synchronous overlap-add), which
# changes the pitch but keeps the timing and the vowel/formant character.  They run in a background thread
# and the result is cached on the channel.  The Vocoder runs live inside the engine.
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES = {
    "Chromatic (all notes)": list(range(12)), "Major": [0, 2, 4, 5, 7, 9, 11], "Natural minor": [0, 2, 3, 5, 7, 8, 10],
    "Harmonic minor": [0, 2, 3, 5, 7, 8, 11], "Dorian": [0, 2, 3, 5, 7, 9, 10], "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "Pentatonic major": [0, 2, 4, 7, 9], "Pentatonic minor": [0, 3, 5, 7, 10], "Blues": [0, 3, 5, 6, 7, 10]}
# name -> (scale degrees, semitones used when the scale is chromatic); degrees None = one octave
HARM_CHOICES = {"Off": None, "3rd above": (2, 4), "3rd below": (-2, -3), "4th above": (3, 5), "5th above": (4, 7),
                "5th below": (-4, -7), "6th above": (5, 9), "Octave above": (None, 12), "Octave below": (None, -12)}
CHORDS = {"Major": [0, 4, 7, 12], "Minor": [0, 3, 7, 12], "Power (5th)": [0, 7, 12, 19], "Major 7th": [0, 4, 7, 11, 12],
          "Minor 7th": [0, 3, 7, 10, 12], "Sus4": [0, 5, 7, 12]}
VOC_BANDS = [8, 12, 16, 20, 24, 32]
FX_HOP, FX_WIN = 256, 2048
_UID = [0]


def new_uid():
    _UID[0] += 1
    return _UID[0]


def default_fx():
    return {"key": 0, "scale": "Major",
            "tune": {"on": False, "speed": 0.02, "amount": 1.0, "notes": False, "fold": True},
            "harm": {"on": False, "v1": "3rd above", "v2": "5th above", "v3": "Off", "l1": 0.6, "l2": 0.5, "l3": 0.5,
                     "dry": 1.0, "spread": 0.6, "human": 0.4, "notes": False, "lc": 0.6, "fold": True},
            "voc": {"on": False, "carrier": "drone", "carrier_uid": 0, "root": 0, "chord": "Major", "bands": 20,
                    "smooth": 0.008, "sib": 0.5, "mix": 1.0, "gain": 1.0, "hide": True, "inst": "Synth Pad"}}


def merge_fx(d):
    """Defaults + whatever a saved project contains (so new settings never break old projects)."""
    out = default_fx()
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k].update(v)
            elif k in out:
                out[k] = v
    return out


# ---- pitch tracking over a whole clip
def _median_nan(track, k):
    out, h = track.copy(), k // 2
    for i in np.nonzero(~np.isnan(track))[0]:
        out[i] = np.nanmedian(track[max(0, i - h):i + h + 1])
    return out


def _octave_fix(track):
    out = track.copy()
    for i in np.nonzero(~np.isnan(track))[0]:
        d = track[i] - np.nanmedian(track[max(0, i - 4):i + 5])
        if abs(abs(d) - 12) < 1.0:
            out[i] = track[i] - 12 * np.sign(d)
    return out


def _fill_gaps(track, maxgap=3):
    out, idx = track.copy(), np.nonzero(~np.isnan(track))[0]
    for a, b in zip(idx[:-1], idx[1:]):
        g = b - a - 1
        if 0 < g <= maxgap:
            out[a + 1:b] = np.linspace(out[a], out[b], g + 2)[1:-1]
    return out


def _fill_nearest(track):
    idx = np.nonzero(~np.isnan(track))[0]
    return np.interp(np.arange(len(track)), idx, track[idx])


def track_pitch(x, sr, hop=FX_HOP, n=FX_WIN):
    """Per-frame pitch of a mono signal in MIDI units (NaN = unvoiced), frame RMS and frame centre samples."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    nfr = 1 + (len(x) - n) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop][:nfr]
    rms = np.sqrt(np.einsum("ij,ij->i", frames, frames) / n)
    track = np.full(nfr, np.nan)
    if nfr and rms.max() > 0.002:
        srt = np.sort(rms)
        gate = max(0.002, srt[int(0.1 * (nfr - 1))] * 2.5, srt[int(0.95 * (nfr - 1))] * 0.06)
        for i in np.nonzero(rms >= gate)[0]:
            f0, conf = yin_frame(frames[i], sr, 70.0, 1000.0)
            if f0 > 0 and conf >= 0.72:
                track[i] = hz_to_midi(f0)
        track = _fill_gaps(_median_nan(_octave_fix(track), 5), 3)
    return track, rms, np.arange(nfr) * hop + n // 2


def _allowed_notes(key, pcs):
    return np.array([m for m in range(0, 128) if (m - key) % 12 in pcs], dtype=np.float64)


def _snap(m, allowed):
    i = int(np.clip(np.searchsorted(allowed, m), 1, len(allowed) - 1))
    lo, hi = allowed[i - 1], allowed[i]
    return lo if abs(m - lo) <= abs(hi - m) else hi


def scale_targets(midi, allowed, hyst=0.15):
    """Nearest scale note for every voiced frame, with hysteresis so vibrato does not flip the target."""
    out, tgt = np.full(len(midi), np.nan), None
    for i, m in enumerate(midi):
        if np.isnan(m):
            tgt = None
            continue
        if tgt is None or abs(m - tgt) > 0.5 + hyst:
            tgt = _snap(m, allowed)
        out[i] = tgt
    return out


def tune_correction(midi, targets, speed, amount, hop, sr):
    """Semitones to add to every frame.  speed = retune time in seconds (0 = instant / robotic)."""
    a = 1.0 if speed <= 1e-3 else 1.0 - math.exp(-hop / (sr * speed))
    corr, cs = np.zeros(len(midi)), 0.0
    for i in range(len(midi)):
        if np.isnan(midi[i]):
            cs *= 0.85
        else:
            raw = 0.0 if np.isnan(targets[i]) else targets[i] - midi[i]  # no target note here: leave the voice alone
            cs += a * (raw - cs)
        corr[i] = cs * amount
    return corr


def _scale_shift(note, deg, semis, key, pcs):
    if np.isnan(note):
        return np.nan
    if len(pcs) == 12:
        return note + semis
    if deg is None:
        deg = len(pcs) if semis > 0 else -len(pcs)
    octv, pc = divmod(int(round(note)) - key, 12)
    d = pcs.index(pc) if pc in pcs else int(np.argmin([abs(pc - p) for p in pcs]))
    o2, d2 = divmod(octv * len(pcs) + d + deg, len(pcs))
    return float(key + 12 * o2 + pcs[d2])


# ---- PSOLA
_HANN = {}


def _hann(L):
    w = _HANN.get(L)
    if w is None:
        w = _HANN[L] = np.hanning(L)
    return w


def _segments(mask, min_len):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [(int(a), int(b)) for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]) if b - a >= min_len]


class Psola:
    """Pitch-synchronous overlap-add: re-places one-period grains of the original at a new spacing, so the pitch
    changes while the timing and the vowel colour stay.  Analysis (pitch marks) is done once and reused."""

    def __init__(self, x2, mono, track, centers, sr):
        self.x, self.sr, self.N = x2, sr, x2.shape[1]
        idx, vf = np.arange(self.N), ~np.isnan(track)
        if vf.any():
            f0f = 440.0 * 2.0 ** ((_fill_nearest(track) - 69.0) / 12.0)
            self.f0 = np.interp(idx, centers, f0f)
            self.voiced = np.interp(idx, centers, vf.astype(np.float64)) >= 0.5
        else:
            self.f0, self.voiced = np.full(self.N, 200.0), np.zeros(self.N, bool)
        self.segs = _segments(self.voiced, int(0.02 * sr))
        lp = sosfiltfilt(butter(2, 900.0, "low", fs=sr, output="sos"), mono) if self.N > 64 else mono
        marks = []
        for a, b in self.segs:
            marks.extend(self._marks(lp, a, b))
        self.marks = np.array(marks, dtype=np.int64)
        w = int(0.008 * sr) | 1
        self.mask = uniform_filter1d(uniform_filter1d(self.voiced.astype(np.float64), w), w)

    def _marks(self, lp, a, b):
        sr, N = self.sr, self.N
        T = sr / self.f0[a]
        cur = a + int(np.argmax(lp[a:min(b, int(a + T) + 1)]))
        out = [cur]
        while True:
            T = sr / self.f0[min(cur, N - 1)]
            nxt = cur + T
            if nxt >= b:
                break
            lo, hi = max(int(nxt - 0.25 * T), cur + 1), min(int(nxt + 0.25 * T) + 1, N)
            if hi <= lo:
                break
            j = lo + int(np.argmax(lp[lo:hi]))
            out.append(j if j > cur else int(nxt))
            cur = out[-1]
        return out

    def shift(self, ratio):
        """Pitch-shifted copy (voiced parts only, zero elsewhere).  ratio = per-sample pitch ratio (1 = unchanged)."""
        sr, N, marks = self.sr, self.N, self.marks
        P = int(sr / 60.0) * 2 + 8
        xp, out = np.pad(self.x, ((0, 0), (P, P))), np.zeros((self.x.shape[0], N + 2 * P))
        for k, (a, b) in enumerate(self.segs):
            mk = marks[np.searchsorted(marks, a):np.searchsorted(marks, b)]
            if len(mk) < 2:
                continue
            s = float(mk[0])
            while s < b:
                si = int(s)
                j = int(np.searchsorted(mk, si))
                if j >= len(mk):
                    j = len(mk) - 1
                elif j > 0 and si - mk[j - 1] < mk[j] - si:
                    j -= 1
                m = int(mk[j])
                Ta = sr / self.f0[m]
                Tout = (sr / self.f0[si]) / ratio[si]
                half = max(2, int(Ta))  # a grain is two analysis periods wide (ratio >= 0.5, so grains never leave gaps)
                out[:, si + P - half:si + P + half + 1] += xp[:, m + P - half:m + P + half + 1] * (_hann(2 * half + 1) * (Tout / half))
                s += Tout
        return out[:, P:P + N] * self.mask


def _active_notes(ctl, centers, track):
    """For every voiced frame: the sorted pitches of the notes drawn on the track at that moment (None = none)."""
    st, en, pi = ctl["notes"][:, 0], ctl["notes"][:, 1], ctl["notes"][:, 2]
    steps = ctl["start"] + (centers / SR) / ctl["sec"]
    out = [None] * len(track)
    for i in np.nonzero(~np.isnan(track))[0]:
        act = pi[(st <= steps[i]) & (steps[i] < en)]
        if len(act):
            out[i] = np.sort(act)
    return out


def _near_octave(m, p, fold):
    return p + 12.0 * round((m - p) / 12.0) if fold else p


def notes_targets(track, active, fold):
    """Auto-Tune target = the nearest of the notes held at that moment (so chords act like 'allowed notes')."""
    out = np.full(len(track), np.nan)
    for i, act in enumerate(active):
        if act is not None:
            c = np.array([_near_octave(track[i], p, fold) for p in act])
            out[i] = c[np.argmin(np.abs(c - track[i]))]
    return out


def fx_autotune(y, mono, fx, sr, ctl=None):
    track, rms, centers = track_pitch(mono, sr)
    t = fx["tune"]
    if t.get("notes"):  # follow the notes drawn on the track
        targets = notes_targets(track, _active_notes(ctl, centers, track), bool(t.get("fold", True))) \
            if ctl is not None and len(ctl["notes"]) else np.full(len(track), np.nan)
    else:
        targets = scale_targets(track, _allowed_notes(fx["key"], SCALES.get(fx["scale"], SCALES["Major"])))
    corr = tune_correction(track, targets, t["speed"], t["amount"], FX_HOP, sr)
    ratio = np.clip(np.interp(np.arange(y.shape[1]), centers, 2.0 ** (corr / 12.0)), 0.5, 2.0)
    ps = Psola(y, mono, track, centers, sr)
    return ps.shift(ratio) + y * (1.0 - ps.mask)


def fx_harmonize(y, fx, sr, ctl=None):
    """Harmony voices that follow the melody: fixed scale intervals, or one voice per note of the chord drawn on the track."""
    h, key = fx["harm"], fx["key"]
    pcs = SCALES.get(fx["scale"], SCALES["Major"])
    mono = y.mean(axis=0)
    track, rms, centers = track_pitch(mono, sr)
    ps, N = Psola(y, mono, track, centers, sr), y.shape[1]
    out = y * h["dry"]
    voices = []  # (target pitch per frame, level, pan, index)
    if h.get("notes"):
        if ctl is not None and len(ctl["notes"]):
            active = _active_notes(ctl, centers, track)
            kmax = min(4, max([len(a) for a in active if a is not None] + [0]))
            pans = np.linspace(-h["spread"], h["spread"], kmax) if kmax > 1 else [0.0]
            for j in range(kmax):
                tgt = np.full(len(track), np.nan)
                for i, act in enumerate(active):
                    if act is not None and len(act) > j:
                        tgt[i] = _near_octave(track[i], act[j], bool(h.get("fold", True)))
                voices.append((tgt, h["lc"], pans[j], j))
    else:
        base = scale_targets(track, _allowed_notes(key, pcs))
        pans = [-h["spread"], h["spread"], 0.0]
        for vi, (name, level) in enumerate(((h["v1"], h["l1"]), (h["v2"], h["l2"]), (h["v3"], h["l3"]))):
            spec = HARM_CHOICES.get(name)
            if spec is not None:
                voices.append((np.array([_scale_shift(b, spec[0], spec[1], key, pcs) for b in base]), level, pans[vi], vi))
    idx = np.arange(N)
    for tgt, level, pan, vi in voices:
        on = ~np.isnan(tgt)
        if level <= 0 or not on.any() or not ps.segs:
            continue
        rng = np.random.default_rng(vi + 1)  # a slow random drift per voice makes the stack sound like different singers
        drift = np.interp(np.arange(len(track)), np.linspace(0, len(track), 24), rng.standard_normal(24)) * 12.0 * h["human"]
        ratio = np.clip(np.interp(idx, centers, 2.0 ** ((np.nan_to_num(tgt - track) + drift / 100.0) / 12.0)), 0.5, 2.0)
        w = int(0.012 * sr) | 1
        res = ps.shift(ratio) * uniform_filter1d((np.interp(idx, centers, on.astype(np.float64)) >= 0.5).astype(np.float64), w)
        d = int(sr * (0.010 + 0.007 * vi) * h["human"])
        if d:
            res = np.concatenate([np.zeros((res.shape[0], d)), res[:, :N - d]], axis=1)
        ang = (pan + 1) * math.pi / 4
        out += res * np.array([[math.cos(ang) * 1.414], [math.sin(ang) * 1.414]]) * level
    pk = float(np.abs(out).max())
    return out / pk if pk > 1.0 else out  # stacking voices must not clip


def process_fx(sample, fx, ctl=None):
    """Offline part of the clip effects (Auto-Tune, then Harmonizer) for one clip.  Same length as the input.
    ctl = notes drawn on the track mapped onto this clip: {"notes": array(start, end, pitch), "start": step, "sec": s}"""
    y = np.asarray(sample, dtype=np.float64)
    if y.shape[1] < 4096:
        return np.asarray(sample, dtype=np.float32)
    if fx["tune"]["on"]:
        y = fx_autotune(y, y.mean(axis=0), fx, SR, ctl)
    if fx["harm"]["on"]:
        y = fx_harmonize(y, fx, SR, ctl)
    return y.astype(np.float32)


def fx_needs_processing(ch):
    return ch.kind == "audio" and ch.sample is not None and bool(ch.fx["tune"]["on"] or ch.fx["harm"]["on"])


def _uses_notes(ch):
    f = ch.fx
    return bool((f["tune"]["on"] and f["tune"].get("notes")) or (f["harm"]["on"] and f["harm"].get("notes")))


def clip_fx_key(ch, n, bpm):
    """Everything that influences the processed audio of one clip; when it changes the clip is re-rendered."""
    f = ch.fx
    cfg = json.dumps({"k": f["key"], "s": f["scale"], "t": f["tune"], "h": f["harm"]}, sort_keys=True)
    ctl = hash(tuple((round(m[0], 4), round(m[1], 4), int(m[2])) for m in ch.pnotes)) if _uses_notes(ch) else 0
    return hash((cfg, round(n[0], 4), round(ch.clip_offset(n), 4), round(ch.clip_dur(n), 4), round(bpm, 3), ctl, id(ch.sample)))


def _start_clip_job(project, ch, n, ent):
    ent["busy"] = True
    job, key = ent["job"], ent["want"]
    a = int(ch.clip_offset(n) * SR)
    region = ch.sample[:, a:a + int(ch.clip_dur(n) * SR)].copy()
    cfg, sec = copy.deepcopy(ch.fx), 60.0 / project.bpm / 4
    ctl = None
    if _uses_notes(ch) and ch.pnotes:
        ctl = {"notes": np.array([[m[0], m[0] + m[1], m[2]] for m in ch.pnotes], dtype=np.float64), "start": float(n[0]), "sec": sec}

    def work():
        out = None
        try:
            out = process_fx(region, cfg, ctl)
        except Exception as ex:
            traceback.print_exc()
            ch.fx_status = f"Error: {ex}"
        if ent["job"] == job:  # (a failed render stores None: the original audio plays until something changes)
            ent["data"], ent["key"] = out, key
        ent["busy"] = False

    threading.Thread(target=work, daemon=True).start()


def ensure_fx(project, debounce=0.4):
    """Call regularly (the GUI does, twice a second): starts background renders for every audio clip whose effect
    inputs changed - settings, notes drawn for the effects, clip position/trim, tempo.  Returns True while work remains.
    Until a clip's render is ready the original audio plays."""
    now, pending = time.time(), False
    for ch in project.channels:
        if ch.kind != "audio" or ch.sample is None:
            continue
        if not fx_needs_processing(ch):
            if ch.fx_clips:
                ch.fx_clips.clear()
                ch.fx_status = ""
            continue
        ids = {n[6] for n in ch.notes}
        for cid in [c for c in ch.fx_clips if c not in ids]:
            del ch.fx_clips[cid]
        for n in ch.notes:
            key = clip_fx_key(ch, n, project.bpm)
            ent = ch.fx_clips.get(n[6])
            if ent is None:
                ent = ch.fx_clips[n[6]] = dict(key=None, data=None, want=None, since=now, busy=False, job=0)
            if ent["want"] != key:
                ent["want"], ent["since"], ent["job"] = key, now, ent["job"] + 1  # newer request: an older render is discarded
            if ent["key"] != key:
                pending = True
                if not ent["busy"] and now - ent["since"] >= debounce:
                    _start_clip_job(project, ch, n, ent)
        busy = sum(1 for e in ch.fx_clips.values() if e["busy"])
        waiting = sum(1 for e in ch.fx_clips.values() if e["key"] != e["want"])
        if busy:
            ch.fx_status = f"Processing {busy} clip{'s' if busy != 1 else ''}..."
        elif waiting:
            ch.fx_status = "Waiting for changes to settle..."
        elif not ch.fx_status.startswith(("Detect", "Error")):
            ch.fx_status = "Ready"
    return pending


def detect_key(sample):
    """Krumhansl-Schmuckler key estimate from the clip's pitches -> (key 0-11, 'Major' | 'Natural minor') or None."""
    track, _, _ = track_pitch(np.asarray(sample, dtype=np.float64).mean(axis=0), SR)
    v = track[~np.isnan(track)]
    if len(v) < 10:
        return None
    hist = np.bincount(np.round(v).astype(int) % 12, minlength=12).astype(float)
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    mnr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best = max(((np.corrcoef(hist, np.roll(p, k))[0, 1], k, nm) for k in range(12)
                for nm, p in (("Major", maj), ("Natural minor", mnr))), key=lambda t: t[0])
    return best[1], best[2]


def detect_key_async(ch):
    ch.fx_status = "Detecting key..."

    def work():
        try:
            r = detect_key(ch.sample)
            if r is None:
                ch.fx_status = "Could not hear a clear pitch to detect the key"
            else:
                ch.fx["key"], ch.fx["scale"] = r
                ch.fx_status = f"Detected key: {KEY_NAMES[r[0]]} {r[1]}"
                ch.fx_refresh_ui = True
        except Exception as ex:
            ch.fx_status = f"Error: {ex}"

    threading.Thread(target=work, daemon=True).start()


def fx_wait(project, timeout=600.0, tick=None):
    """Block until every clip effect has finished rendering (used before exporting)."""
    t0 = time.time()
    while ensure_fx(project, debounce=0.0) and time.time() - t0 < timeout:
        if tick:
            tick()
        time.sleep(0.05)


# ==== GUI ====
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QRectF, QPointF, QRect, QTimer, pyqtSignal, QLineF
from PyQt6.QtGui import QColor, QPainter, QPen, QPainterPath

BG, PANEL, DARK, ACC, TXT, DIM = "#2b3238", "#37414a", "#1f262b", "#ff9a2e", "#d5dde2", "#8a98a3"

QSS = f"""
* {{ color:{TXT}; font-family:'Segoe UI','Helvetica Neue',Arial; font-size:12px; }}
QMainWindow, QDialog {{ background:{BG}; }}
QWidget#strip {{ background:{PANEL}; border:1px solid {DARK}; border-radius:6px; }}
QWidget#rollbar {{ background:{PANEL}; border-bottom:2px solid {DARK}; }}
QMenuBar {{ background:{DARK}; }} QMenuBar::item:selected {{ background:{ACC}; color:#111; }}
QMenu {{ background:{PANEL}; border:1px solid {DARK}; }} QMenu::item:selected {{ background:{ACC}; color:#111; }}
QMenu::item:disabled {{ color:{ACC}; font-weight:bold; }}
QToolBar {{ background:{PANEL}; border:none; border-bottom:2px solid {DARK}; spacing:8px; padding:5px; }}
QDockWidget {{ titlebar-close-icon:none; }}
QDockWidget::title {{ background:{DARK}; padding:5px 8px; color:{ACC}; font-weight:bold; }}
QPushButton {{ background:#47525b; border:1px solid {DARK}; border-radius:4px; padding:4px 10px; }}
QPushButton:hover {{ background:#56626c; }} QPushButton:checked {{ background:{ACC}; color:#111; }}
QPushButton:disabled {{ color:#6d7a84; }}
QPushButton#play {{ font-size:18px; min-width:40px; color:#7be07b; }}
QPushButton#play:checked {{ background:#3f8f3f; color:white; }}
QPushButton#stop {{ font-size:18px; min-width:40px; color:#ff7b6b; }}
QPushButton#restart {{ font-size:18px; min-width:40px; color:#6bb5ff; }}
QPushButton#rec {{ font-size:14px; font-weight:bold; color:#ff7b6b; padding:8px 16px; }}
QPushButton#led {{ min-width:14px; max-width:14px; min-height:14px; max-height:14px; padding:0; border-radius:7px; background:#2c3a2c; }}
QPushButton#led:checked {{ background:#6bff6b; }}
QDoubleSpinBox, QSpinBox, QComboBox {{ background:{DARK}; border:1px solid #11161a; border-radius:4px; padding:3px 6px; }}
QDoubleSpinBox#bpm {{ color:{ACC}; font:bold 20px 'Consolas','Courier New',monospace; min-width:105px; }}
QLabel#time {{ color:{ACC}; background:{DARK}; border-radius:4px; padding:3px 10px; font:bold 20px 'Consolas','Courier New',monospace; }}
QLabel#bignote {{ color:{ACC}; background:{DARK}; border-radius:6px; padding:6px; font:bold 34px 'Consolas','Courier New',monospace; }}
QCheckBox {{ spacing:6px; }}
QTreeView {{ background:{DARK}; border:none; }} QTreeView::item:selected {{ background:{ACC}; color:#111; }}
QScrollArea {{ border:none; background:{BG}; }}
QScrollBar:vertical {{ background:{DARK}; width:12px; }} QScrollBar:horizontal {{ background:{DARK}; height:12px; }}
QScrollBar::handle {{ background:#58646e; border-radius:5px; min-height:24px; min-width:24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
QSlider::groove:vertical {{ background:{DARK}; width:8px; border-radius:4px; }}
QSlider::sub-page:vertical {{ background:{ACC}; border-radius:4px; }}
QSlider::handle:vertical {{ background:#dfe5e8; height:14px; margin:0 -6px; border-radius:3px; }}
QProgressBar {{ background:{DARK}; border:none; border-radius:3px; max-height:8px; }}
QProgressBar::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6bff6b,stop:0.75 #ffd23f,stop:1 #ff5b4a); }}
QToolTip {{ background:{DARK}; border:1px solid {ACC}; }}
"""


def clear(layout):
    while layout.count():
        it = layout.takeAt(0)
        if it.widget():
            it.widget().deleteLater()
        elif it.layout():
            clear(it.layout())


def fit_length(end_step):
    """Smallest allowed song length (in steps) that holds end_step."""
    for c in LEN_CHOICES:
        if c >= end_step:
            return c
    return MAX_LEN


def build_add_menu(menu, main):
    """Fill a menu with every instrument (grouped), audio clips and samples."""
    for cat in INSTRUMENT_CATS:
        hdr = menu.addAction(f"— {cat} —")
        hdr.setEnabled(False)
        for name, sp in INSTRUMENTS.items():
            if sp["cat"] == cat:
                menu.addAction(name, lambda _=False, n=name: main.add_instrument(n))
    menu.addSeparator()
    menu.addAction("Audio clip (mp3 / wav / flac)...", main.add_audio_dialog)
    menu.addAction("Sample (one-shot, pitched)...", main.add_sample_dialog)


class Knob(QtWidgets.QWidget):
    changed = pyqtSignal(float)

    def __init__(self, name, value=0.0, lo=0.0, hi=1.0, color=ACC, size=34, show=True):
        super().__init__()
        self.name, self.lo, self.hi, self.v, self.default, self.show = name, lo, hi, value, value, show
        self.color = QColor(color)
        self.setFixedSize(size, size + (13 if show else 0))
        self._drag = None
        self._tip()

    def _tip(self):
        self.setToolTip(f"{self.name}: {self.v:.2f}")

    def setValue(self, v):
        self.v = min(max(v, self.lo), self.hi)
        self._tip()
        self.update()

    def _set(self, v):
        self.setValue(v)
        self.changed.emit(self.v)

    def mousePressEvent(self, e):
        self._drag = (e.position().y(), self.v)

    def mouseMoveEvent(self, e):
        if self._drag:
            self._set(self._drag[1] + (self._drag[0] - e.position().y()) / 140.0 * (self.hi - self.lo))

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self._set(self.default)

    def wheelEvent(self, e):
        self._set(self.v + (1 if e.angleDelta().y() > 0 else -1) * (self.hi - self.lo) / 40)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self.width()
        r = QRectF(4, 4, s - 8, s - 8)
        frac = (self.v - self.lo) / (self.hi - self.lo)
        p.setBrush(QColor("#2a3238"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(r.adjusted(3, 3, -3, -3))
        p.setPen(QPen(QColor("#151a1e"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(r, 225 * 16, -270 * 16)
        p.setPen(QPen(self.color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(r, 225 * 16, int(-270 * 16 * frac))
        ang, c = math.radians(225 - 270 * frac), r.center()
        p.setPen(QPen(QColor("#ffffff"), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(c, QPointF(c.x() + math.cos(ang) * (s / 2 - 9), c.y() - math.sin(ang) * (s / 2 - 9)))
        if self.show:
            p.setPen(QColor(DIM))
            p.drawText(QRectF(0, s, s, 13), Qt.AlignmentFlag.AlignCenter, self.name)


# ------------------------------------------------------------ play bar ruler
class Ruler(QtWidgets.QWidget):
    """Bar/beat ruler with the draggable play bar.  Click or drag anywhere to move the play bar."""
    H = 24

    def __init__(self, main, cw, off, label="", pad=0):
        super().__init__()
        self.main, self.cw, self.off, self.label, self.pad = main, cw, off, label, pad
        self.px, self.drag = -20.0, False
        self.setFixedHeight(self.H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click or drag to move the play bar (hold Alt for fine positioning)")
        self.fit()

    def fit(self):
        self.setFixedWidth(int(self.off + self.main.project.length * self.cw + self.pad))

    def set_play(self, stepf):
        old = self.px
        self.px = self.off + stepf * self.cw
        for x in (old, self.px):
            self.update(QRect(int(x) - 9, 0, 19, self.H))

    def paintEvent(self, ev):
        p = QPainter(self)
        r = ev.rect()
        p.fillRect(r, QColor(DARK))
        L, cw, off = self.main.project.length, self.cw, self.off
        i0, i1 = max(0, int((r.left() - off) // cw) - 1), min(L, int((r.right() - off) // cw) + 2)
        f = p.font()
        f.setPixelSize(10)
        p.setFont(f)
        for i in range(i0, i1 + 1):
            x = off + i * cw
            if i % 16 == 0:
                p.fillRect(QRectF(x, 4, 1, self.H - 4), QColor("#9aa5ab"))
                p.setPen(QColor(TXT))
                p.drawText(QRectF(x + 3, 0, 40, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, str(i // 16 + 1))
            elif i % 4 == 0:
                p.fillRect(QRectF(x, 12, 1, self.H - 12), QColor("#6d7a84"))
            else:
                p.fillRect(QRectF(x, 18, 1, self.H - 18), QColor("#3e4952"))
        if self.label and off > 30 and r.left() < off:
            p.setPen(QColor(DIM))
            p.drawText(QRectF(4, 0, off - 8, self.H), Qt.AlignmentFlag.AlignVCenter, self.label)
        if self.px >= 0:
            x = self.px
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(ACC))
            tri = QtGui.QPolygonF([QPointF(x - 6, 1), QPointF(x + 6, 1), QPointF(x, 12)])
            p.drawPolygon(tri)
            p.fillRect(QRectF(x - 1, 10, 2, self.H - 10), QColor(ACC))

    def _seek(self, e):
        step = (e.position().x() - self.off) / self.cw
        if not (e.modifiers() & Qt.KeyboardModifier.AltModifier):
            step = round(step)
        self.main.seek_step(step)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.drag = True
            self._seek(e)

    def mouseMoveEvent(self, e):
        if self.drag:
            self._seek(e)

    def mouseReleaseEvent(self, e):
        self.drag = False


# ------------------------------------------------------------ channel rack
class StepRow(QtWidgets.QWidget):
    CW, CH = 24, 30

    def __init__(self, main, ch):
        super().__init__()
        self.main, self.ch, self.play, self.state = main, ch, -1, None
        self.setFixedHeight(self.CH + 2)
        self.fit()

    def fit(self):
        self.setFixedWidth(self.main.project.length * self.CW + 2)

    def starts(self):
        return {int(n[0]) for n in self.ch.notes}

    def set_play(self, i):
        if i != self.play:
            self.play = i
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        on, col = self.starts(), QColor(self.ch.color)
        p.setPen(Qt.PenStyle.NoPen)
        r0 = e.rect()
        i0 = max(0, (r0.left() - 2) // self.CW)
        i1 = min(self.main.project.length, (r0.right() - 2) // self.CW + 1)
        for i in range(i0, i1):
            r = QRectF(2 + i * self.CW, 1, self.CW - 2, self.CH)
            p.setBrush(col if i in on else QColor("#56616a" if (i // 4) % 2 else "#46505a"))
            p.drawRoundedRect(r, 3, 3)
            if i in on:
                p.setBrush(QColor(255, 255, 255, 45))
                p.drawRoundedRect(QRectF(r.x(), r.y(), r.width(), r.height() / 2), 3, 3)
            if i == self.play:
                p.setBrush(QColor(255, 255, 255, 90))
                p.drawRoundedRect(r, 3, 3)

    def cell(self, e):
        return int((e.position().x() - 2) // self.CW)

    def mousePressEvent(self, e):
        i = self.cell(e)
        if not 0 <= i < self.main.project.length:
            return
        self.main.select_channel(self.ch)
        self.state = (i not in self.starts()) and e.button() != Qt.MouseButton.RightButton
        self.apply(i)

    def mouseMoveEvent(self, e):
        i = self.cell(e)
        if self.state is not None and 0 <= i < self.main.project.length:
            self.apply(i)

    def mouseReleaseEvent(self, e):
        self.state = None

    def apply(self, i):
        ch, has = self.ch, i in self.starts()
        if self.state and not has:
            vow = self.main.roll.new_vowel if ch.spec["fam"] == "choir" else -1
            ch.notes.append(mknote(i, 1.0, ROOT, 0.9, 0, vow))
            self.main.engine.note_on(ch, ROOT, 0.9, hold=int(0.25 * SR))
        elif not self.state and has:
            ch.notes[:] = [n for n in ch.notes if int(n[0]) != i]
        else:
            return
        self.update()
        self.main.roll.update()


def paint_clip(p, ch, n, x0, y, w, h, vis_l, vis_r):
    """Draw one audio clip: a rectangle exactly as long as the (trimmed) clip, with the waveform as
    vertical up/down lines inside it."""
    col = QColor(ch.color)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setBrush(col.darker(230))
    p.setPen(QPen(col, 1.5))
    p.drawRoundedRect(QRectF(x0, y, w, h), 4, 4)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    pk = ch.peaks
    if pk is not None and len(pk) and ch.sample is not None:
        total, off, dur = ch.sample.shape[1] / SR, ch.clip_offset(n), ch.clip_dur(n)
        a, b = int(max(x0 + 1, vis_l)), int(min(x0 + w - 1, vis_r + 1))
        if b > a:
            xs = np.arange(a, b)
            idx = np.clip(((off + (xs - x0) / w * dur) / total * len(pk)).astype(np.int64), 0, len(pk) - 1)
            amp = pk[idx] * (h / 2 - 4)
            mid = y + h / 2
            p.setPen(QPen(col.lighter(145), 1))
            p.drawLines([QLineF(float(x), mid - float(v), float(x), mid + float(v)) for x, v in zip(xs, amp)])
    if w > 60 and ch.path:
        f = p.font()
        f.setPixelSize(10)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255, 190))
        p.drawText(QRectF(x0 + 6, y + 2, w - 10, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   os.path.basename(ch.path))


class ClipEdit:
    """Mouse editing shared by the rack lane and the piano roll for audio-clip tracks:
    drag = move, drag an edge = trim (cut the length), Ctrl+drag = copy, right-click = menu
    (Copy / Cut / Duplicate / Split here / Delete / Paste), middle-click on empty space = paste."""
    cdrag = None

    def _sec(self):
        return 60.0 / self.main.project.bpm / 4

    def _same_clip_source(self, other):
        return other is self.ch or bool(other.path and other.path == self.ch.path and self.ch.sample is not None)

    def clip_hit(self, sx, tol):
        bpm = self.main.project.bpm
        for i in range(len(self.ch.notes) - 1, -1, -1):
            n = self.ch.notes[i]
            ln = self.ch.clip_steps_of(n, bpm)
            if n[0] <= sx < n[0] + ln:
                t = min(tol, ln / 3.0)
                return i, ("l" if sx - n[0] < t else ("r" if n[0] + ln - sx < t else "m"))
        return None, None

    def clip_hover(self, sx, tol):
        i, zone = self.clip_hit(sx, tol)
        self.setCursor(Qt.CursorShape.SplitHCursor if zone in ("l", "r") else
                       (Qt.CursorShape.SizeAllCursor if zone == "m" else Qt.CursorShape.ArrowCursor))

    def clip_press(self, btn, mods, sx, tol, gpos):
        ch = self.ch
        i, zone = self.clip_hit(sx, tol)
        if btn == Qt.MouseButton.MiddleButton:
            if i is None:
                self.clip_paste(sx)
            return
        if btn == Qt.MouseButton.RightButton:
            self.clip_menu(i, sx, gpos)
            return
        if i is None:
            return
        n = ch.notes[i]
        if zone == "m" and (mods & Qt.KeyboardModifier.ControlModifier):  # Ctrl+drag = drag out a copy
            c = list(n)
            c[6] = new_id()
            ch.notes.append(c)
            i, n = len(ch.notes) - 1, c
        self.cdrag = dict(i=i, zone=zone, off=sx - n[0], start=n[0], offs=ch.clip_offset(n), dur=ch.clip_dur(n))

    def clip_move(self, sx, mods):
        d = self.cdrag
        if not d or d["i"] >= len(self.ch.notes):
            return False
        ch, n, sec, L = self.ch, self.ch.notes[d["i"]], self._sec(), self.main.project.length
        snap = (lambda v: v) if (mods & Qt.KeyboardModifier.AltModifier) else round
        total = ch.sample.shape[1] / SR
        if d["zone"] == "m":
            n[0] = float(max(0, min(L - 1, snap(sx - d["off"]))))
        elif d["zone"] == "r":  # trim the end
            dur = min(max((snap(sx) - d["start"]) * sec, 0.03), total - d["offs"])
            n[7], n[1] = dur, dur / sec
        else:  # trim the start (the audio stays where it is, only its beginning is cut)
            delta = (snap(sx) - d["start"]) * sec
            delta = max(-d["offs"], -d["start"] * sec, min(delta, d["dur"] - 0.03))
            n[0], n[5], n[7] = d["start"] + delta / sec, d["offs"] + delta, d["dur"] - delta
            n[1] = n[7] / sec
        return True

    def clip_release(self):
        if self.cdrag and self.cdrag["i"] < len(self.ch.notes):
            n = self.ch.notes[self.cdrag["i"]]
            self.main.ensure_length(n[0] + n[1])
        self.cdrag = None

    def clip_copy(self, i):
        self.main.clip_clipboard = (self.ch, list(self.ch.notes[i]))
        self.main.flash("Copied the audio clip - middle-click an empty spot (or right-click > Paste here) to paste it")

    def clip_paste(self, sx):
        cb = self.main.clip_clipboard
        if not cb:
            return
        if not self._same_clip_source(cb[0]):
            return self.main.flash("That clip belongs to a different audio file - paste it into a track using the same file.")
        c = list(cb[1])
        c[0], c[6] = float(max(0, min(self.main.project.length - 1, round(sx)))), new_id()
        c[1] = self.ch.clip_steps_of(c, self.main.project.bpm)
        self.ch.notes.append(c)
        self.main.ensure_length(c[0] + c[1])

    def clip_menu(self, i, sx, gpos):
        ch, m, sec = self.ch, QtWidgets.QMenu(self), self._sec()
        acts = {}
        if i is not None:
            for label, key in (("Copy", "copy"), ("Cut", "cut"), ("Duplicate", "dup"), ("Split here", "split"), ("Delete", "del")):
                acts[m.addAction(label)] = key
        cb = self.main.clip_clipboard
        if cb and self._same_clip_source(cb[0]):
            acts[m.addAction("Paste here")] = "paste"
        if not acts:
            return
        key = acts.get(m.exec(gpos))
        if key is None:
            return
        if key == "paste":
            self.clip_paste(sx)
        elif key == "copy":
            self.clip_copy(i)
        elif key == "cut":
            self.clip_copy(i)
            del ch.notes[i]
        elif key == "del":
            del ch.notes[i]
        elif key == "dup":
            c = list(ch.notes[i])
            c[0], c[6] = c[0] + c[1], new_id()
            ch.notes.append(c)
            self.main.ensure_length(c[0] + c[1])
        elif key == "split":
            n = ch.notes[i]
            cut = float(round(sx))
            if n[0] + 0.2 < cut < n[0] + n[1] - 0.2:
                dur, off = ch.clip_dur(n), ch.clip_offset(n)
                d1 = (cut - n[0]) * sec
                c = list(n)
                c[0], c[5], c[7], c[6] = cut, off + d1, dur - d1, new_id()
                c[1] = c[7] / sec
                n[7], n[1] = d1, d1 / sec
                ch.notes.append(c)
        self.update()


class AudioLane(ClipEdit, StepRow):
    """Channel-rack lane of an audio-clip track."""
    CH = 46

    def __init__(self, main, ch):
        super().__init__(main, ch)
        self.cdrag = None
        self.setMouseTracking(True)

    def paintEvent(self, e):
        p = QPainter(self)
        r0 = e.rect()
        L, CW, CH = self.main.project.length, self.CW, self.CH
        i0, i1 = max(0, (r0.left() - 2) // CW), min(L, (r0.right() - 2) // CW + 1)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(i0, i1):
            p.setBrush(QColor("#56616a" if (i // 4) % 2 else "#46505a"))
            p.drawRect(QRectF(2 + i * CW, 1, CW, CH))
        for n in self.ch.notes:
            x0, w = 2 + n[0] * CW, max(self.ch.clip_steps_of(n, self.main.project.bpm) * CW, 4.0)
            if x0 + w >= r0.left() and x0 <= r0.right():
                paint_clip(p, self.ch, n, x0, 1, w, CH, r0.left(), r0.right())
        if 0 <= self.play < L:
            p.fillRect(QRectF(2 + self.play * CW, 1, CW, CH), QColor(255, 255, 255, 40))

    def mousePressEvent(self, e):
        self.main.select_channel(self.ch)
        self.clip_press(e.button(), e.modifiers(), (e.position().x() - 2) / self.CW, 6.0 / self.CW,
                        e.globalPosition().toPoint())
        self.update()
        self.main.roll.update()

    def mouseMoveEvent(self, e):
        sx = (e.position().x() - 2) / self.CW
        if self.cdrag:
            if self.clip_move(sx, e.modifiers()):
                self.update()
                self.main.roll.update()
        else:
            self.clip_hover(sx, 6.0 / self.CW)

    def mouseReleaseEvent(self, e):
        self.clip_release()
        self.update()
        self.main.roll.update()


class ChannelRow(QtWidgets.QWidget):
    def __init__(self, main, ch):
        super().__init__()
        self.main, self.ch = main, ch
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(6)
        self.led = QtWidgets.QPushButton()
        self.led.setObjectName("led")
        self.led.setCheckable(True)
        self.led.setChecked(not ch.mute)
        self.led.toggled.connect(lambda on: (setattr(ch, "mute", not on), main.sync(ch)))
        self.pank = Knob("Pan", ch.pan, -1, 1, size=28, show=False)
        self.volk = Knob("Volume", ch.vol, 0, 1, size=28, show=False)
        self.pank.changed.connect(lambda v: (setattr(ch, "pan", v), main.sync(ch)))
        self.volk.changed.connect(lambda v: (setattr(ch, "vol", v), main.sync(ch)))
        self.btn = QtWidgets.QPushButton(ch.name)
        self.btn.setCheckable(True)
        self.btn.setFixedSize(112, 30)
        self.btn.setToolTip(ch.inst if ch.kind == "synth" else ("Audio clip" if ch.kind == "audio" else "Sample"))
        self.btn.setStyleSheet(f"QPushButton{{text-align:left;padding-left:8px;border-left:6px solid {ch.color};background:#4a555f;}}"
                               f"QPushButton:checked{{background:#6a7782;color:white;}}")
        self.btn.clicked.connect(lambda: main.select_channel(ch))
        self.btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.btn.customContextMenuRequested.connect(self.menu)
        self.steps = AudioLane(main, ch) if ch.kind == "audio" else StepRow(main, ch)
        for w in (self.led, self.pank, self.volk, self.btn, self.steps):
            lay.addWidget(w)
        lay.addStretch()

    def menu(self, pos):
        m = QtWidgets.QMenu(self)
        a_ren, a_clr, a_del = m.addAction("Rename..."), m.addAction("Clear pattern"), m.addAction("Delete track")
        a_mv = m.addAction("Move first clip to play bar") if self.ch.kind == "audio" else None
        act = m.exec(self.btn.mapToGlobal(pos))
        if act is None:
            return
        if act == a_ren:
            t, ok = QtWidgets.QInputDialog.getText(self, "Rename", "Name:", text=self.ch.name)
            if ok and t:
                self.ch.name = t[:14]
                self.main.refresh_all()
        elif act == a_clr:
            self.ch.notes.clear()
            self.main.roll.update()
            self.steps.update()
        elif act == a_del:
            self.main.project.channels.remove(self.ch)
            self.main.refresh_all()
        elif a_mv is not None and act == a_mv:
            if self.ch.notes:
                self.ch.notes[0][0] = float(round(self.main.engine.pos / self.main.engine.step_samples()))
                self.steps.update()


class Rack(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main, self.rows, self.ruler = main, [], None
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.inner = QtWidgets.QWidget()
        self.col = QtWidgets.QVBoxLayout(self.inner)
        self.col.setContentsMargins(4, 4, 4, 4)
        self.col.setSpacing(2)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.inner)
        v.addWidget(self.scroll)

    def rebuild(self):
        clear(self.col)
        self.rows = []
        # header row: the play-bar ruler, aligned with the step cells (210 px = 4 + 200 + 6 spacing)
        hdr = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(hdr)
        hl.setContentsMargins(4, 0, 4, 0)
        hl.setSpacing(6)
        lab = QtWidgets.QLabel("Play bar  \u25B8  click / drag the ruler")
        lab.setFixedWidth(200)
        lab.setStyleSheet(f"color:{DIM};")
        self.ruler = Ruler(self.main, StepRow.CW, 2)
        hl.addWidget(lab)
        hl.addWidget(self.ruler)
        hl.addStretch()
        self.col.addWidget(hdr)
        for ch in self.main.project.channels:
            r = ChannelRow(self.main, ch)
            self.rows.append(r)
            self.col.addWidget(r)
        add = QtWidgets.QPushButton("+  Add track")
        menu = QtWidgets.QMenu(add)
        build_add_menu(menu, self.main)
        add.setMenu(menu)
        self.col.addWidget(add, alignment=Qt.AlignmentFlag.AlignLeft)
        self.col.addStretch()

    def refresh_steps(self):
        for r in self.rows:
            r.steps.update()


# ------------------------------------------------------------ piano roll
class PianoRoll(ClipEdit, QtWidgets.QWidget):
    CW, RH, KW = 26, 14, 58

    def __init__(self, main):
        super().__init__()
        self.main, self.ch, self.px = main, None, -10.0
        self.mode = self.note = self.key = None
        self.lastlen, self.off = 1.0, 0.0
        self.slur_mode, self.new_vowel = False, 0
        self.sel, self.slur_from, self.group = set(), None, None  # selected note ids / pending slur start
        self.rb, self.mid = None, None  # rubber-band rectangle / middle-button state
        self._pairs_key, self._pairs = None, []
        self.cdrag = None
        self.setMouseTracking(True)
        self.fit()

    def fit(self):
        self.setFixedSize(self.KW + self.main.project.length * self.CW + 1, (P_HI - P_LO + 1) * self.RH)

    # ---- an audio track has two layers: its clips, and the notes that play its effects (like a synth)
    def notes_mode(self):
        return bool(self.ch and self.ch.kind == "audio" and self.ch.notes_mode)

    def clip_mode(self):
        return bool(self.ch and self.ch.kind == "audio" and not self.ch.notes_mode)

    def synthlike(self):
        return bool(self.ch and (self.ch.kind == "synth" or self.notes_mode()))

    @property
    def notes(self):
        if not self.ch:
            return []
        return self.ch.pnotes if self.notes_mode() else self.ch.notes

    def fam(self):
        if not self.ch:
            return ""
        if self.notes_mode():
            return INSTRUMENTS.get(self.ch.fx["voc"].get("inst"), INSTRUMENTS["Synth"])["fam"]
        return self.ch.spec["fam"]

    def pch(self):
        """Channel that sounds when you click/play a note: the carrier synth while editing notes for effects."""
        return self.main.engine.carrier(self.ch, True) if self.notes_mode() else self.ch

    def set_channel(self, ch):
        self.ch = ch
        self.sel, self.slur_from, self.group, self.rb, self.mid = set(), None, None, None, None
        self.main.set_hint("")
        self.update()

    def set_play(self, stepf):
        old = self.px
        self.px = self.KW + stepf * self.CW if stepf >= 0 else -10.0
        for x in (old, self.px):
            self.update(QRect(int(x) - 3, 0, 8, self.height()))

    def row_pitch(self, y):
        return int(min(P_HI, max(P_LO, P_HI - int(y // self.RH))))

    def note_rect(self, n):
        return QRectF(self.KW + n[0] * self.CW + 1, (P_HI - int(n[2])) * self.RH + 1, max(n[1] * self.CW - 2, 3), self.RH - 2)

    def note_at(self, x, y):
        if not self.ch:
            return None
        pitch, sx = self.row_pitch(y), (x - self.KW) / self.CW
        for n in reversed(self.notes):
            if int(n[2]) == pitch and n[0] <= sx < n[0] + n[1]:
                return n
        return None

    def selected_notes(self):
        return [n for n in self.notes if n[6] in self.sel] if self.ch else []

    # ---- slurs: explicit links between two notes, drawn as a bent band
    def slur_pairs(self):
        if not self.ch or not self.synthlike():
            return []
        key = hash(tuple(tuple(n) for n in self.notes))
        if key != self._pairs_key:
            self._pairs_key = key
            self._pairs = []
            if any(len(n) > 4 and n[4] >= 2 for n in self.notes):
                for chain in build_chains(self.notes):
                    self._pairs += list(zip(chain[:-1], chain[1:]))
        return self._pairs

    def link_target(self, n):
        for a, b in self.slur_pairs():
            if a[6] == n[6]:
                return b
        return None

    def ribbon(self, a, b):
        """Bent rectangle connecting the end of note a to the start of note b."""
        KW, CW, RH = self.KW, self.CW, self.RH
        ya, yb, h = (P_HI - int(a[2])) * RH + 1, (P_HI - int(b[2])) * RH + 1, RH - 2
        x0, x1 = KW + (a[0] + a[1]) * CW - 1, KW + b[0] * CW + 1
        if x1 - x0 < 8:  # the notes touch or overlap: a short bent connector just before b
            x0 = x1 - 8
        mid = (x0 + x1) / 2
        path = QPainterPath(QPointF(x0, ya))
        path.cubicTo(QPointF(mid, ya), QPointF(mid, yb), QPointF(x1, yb))
        path.lineTo(QPointF(x1, yb + h))
        path.cubicTo(QPointF(mid, yb + h), QPointF(mid, ya + h), QPointF(x0, ya + h))
        path.closeSubpath()
        return path

    def slur_click(self, n):
        a = self.slur_from
        if a is None:
            if n is None:
                return
            if self.link_target(n) is not None:
                n[4] = 0
                self.main.set_hint("Slur removed")
            else:
                self.slur_from = n
                self.main.set_hint("Slur: now click the LAST note (any distance away)  -  Esc cancels")
        elif n is None or n is a:
            self.slur_from = None
            self.main.set_hint("")
        elif n[0] == a[0]:
            self.main.set_hint("Slur: pick a note that starts at a different time")
            return
        else:
            src, dst = (a, n) if n[0] > a[0] else (n, a)
            for m in self.notes:
                if m is not src and m[4] == dst[6]:
                    m[4] = 0  # a note can only have one incoming slur
            src[4] = dst[6]
            self.slur_from = None
            self.main.set_hint("Slurred")
        self.update()

    def cancel(self):
        if self.slur_from is not None:
            self.slur_from = None
            self.main.set_hint("")
        else:
            self.sel.clear()
        self.update()

    # ---- selection / clipboard
    def copy_selected(self):
        notes = [list(n) for n in self.selected_notes()]
        if notes:
            self.main.note_clipboard = notes
            self.main.flash(f"Copied {len(notes)} note(s) - middle-click an empty tile to paste")

    def delete_selected(self):
        if not self.ch:
            return
        ids = set(self.sel)
        self.notes[:] = [n for n in self.notes if n[6] not in ids]
        for m in self.notes:
            if m[4] in ids:
                m[4] = 0
        self.sel.clear()
        self.update()
        self.main.rack.refresh_steps()

    def cut_selected(self):
        self.copy_selected()
        self.delete_selected()

    def select_all(self):
        if self.ch and not self.clip_mode():
            self.sel = {n[6] for n in self.notes}
            self.update()

    def paste_at(self, step, pitch):
        cb = self.main.note_clipboard
        if not cb:
            return self.main.flash("Nothing copied yet - select notes and copy them first")
        if not self.ch or self.clip_mode():
            return self.main.flash("Select an instrument or sampler track to paste notes into")
        first = min(cb, key=lambda n: (n[0], n[2]))
        dx, dp = step - first[0], (0 if pitch is None else pitch - int(first[2]))
        idmap = {n[6]: new_id() for n in cb}
        new = []
        for n in cb:
            c = list(n)
            c[0] = float(n[0] + dx)
            c[2] = int(min(P_HI, max(P_LO, int(n[2]) + dp)))
            c[6] = idmap[n[6]]
            c[4] = idmap.get(n[4], 0) if n[4] >= 2 else 0
            new.append(c)
        self.notes.extend(new)
        self.sel = {c[6] for c in new}
        self.main.ensure_length(max(c[0] + c[1] for c in new))
        self.main.engine.note_on(self.pch(), int(new[0][2]), 0.9, hold=int(0.2 * SR))
        self.main.flash(f"Pasted {len(new)} note(s) into {self.ch.name}")
        self.update()
        self.main.rack.refresh_steps()

    def note_menu(self):
        k = len(self.sel)
        m = QtWidgets.QMenu(self)
        a_copy = m.addAction(f"Copy  ({k} note{'s' if k != 1 else ''})")
        a_cut = m.addAction("Cut")
        a_del = m.addAction("Delete")
        act = m.exec(QtGui.QCursor.pos())
        if act == a_copy:
            self.copy_selected()
        elif act == a_cut:
            self.cut_selected()
        elif act == a_del:
            self.delete_selected()

    # ---- painting
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rc = ev.rect()
        W, H, L, KW, RH, CW = self.width(), self.height(), self.main.project.length, self.KW, self.RH, self.CW
        rows = P_HI - P_LO + 1
        r0, r1 = max(0, rc.top() // RH), min(rows - 1, rc.bottom() // RH)
        for r in range(r0, r1 + 1):
            pitch, y = P_HI - r, r * RH
            black = pitch % 12 in (1, 3, 6, 8, 10)
            p.fillRect(QRectF(KW, y, W - KW, RH), QColor("#2b3339" if black else "#353f46"))
            if pitch % 12 == 0:
                p.fillRect(QRectF(KW, y + RH - 1, W - KW, 1), QColor("#1c2227"))
        i0, i1 = max(0, (rc.left() - KW) // CW), min(L, (rc.right() - KW) // CW + 1)
        for i in range(i0, i1 + 1):
            col = "#12171a" if i % 16 == 0 else ("#1f262b" if i % 4 == 0 else "#2a3238")
            p.fillRect(QRectF(KW + i * CW, 0, 1, H), QColor(col))
        for r in range(r0, r1 + 1):
            pitch, y = P_HI - r, r * RH
            black = pitch % 12 in (1, 3, 6, 8, 10)
            p.fillRect(QRectF(0, y, KW, RH), QColor("#dfe5e8"))
            p.fillRect(QRectF(0, y + RH - 1, KW, 1), QColor("#9aa5ab"))
            if black:
                p.fillRect(QRectF(0, y, KW * 0.62, RH), QColor("#21282d"))
            if pitch % 12 == 0:
                p.setPen(QColor("#333"))
                p.drawText(QRectF(KW * 0.62, y, KW * 0.38 - 3, RH), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                           f"C{pitch // 12 - 1}")
        if self.ch and self.clip_mode():
            y0 = H / 2 - 110
            f = p.font()
            f.setPixelSize(11)
            p.setFont(f)
            p.setPen(QColor(DIM))
            p.drawText(QRectF(KW + 8, y0 - 26, 900, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       "Audio clip:  drag = move   |   drag an edge = cut the length   |   Ctrl+drag = copy   |   "
                       "right-click = menu   |   middle-click empty = paste")
            for n in self.notes:
                x0 = KW + n[0] * CW
                w = max(self.ch.clip_steps_of(n, self.main.project.bpm) * CW, 4.0)
                if x0 + w >= rc.left() and x0 <= rc.right():
                    paint_clip(p, self.ch, n, x0, y0, w, 220, rc.left(), rc.right())
        elif self.ch:
            col = QColor(self.ch.color)
            choir = self.fam() == "choir"
            if self.ch.kind == "audio":  # the clip(s) as a faint guide so notes can be lined up with the voice
                p.setOpacity(0.3)
                for n in self.ch.notes:
                    x0, w = KW + n[0] * CW, max(self.ch.clip_steps_of(n, self.main.project.bpm) * CW, 4.0)
                    if x0 + w >= rc.left() and x0 <= rc.right():
                        paint_clip(p, self.ch, n, x0, H / 2 - 110, w, 220, rc.left(), rc.right())
                p.setOpacity(1.0)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            for a, b in self.slur_pairs():  # bent bands under the notes
                path = self.ribbon(a, b)
                if path.boundingRect().right() < rc.left() - 40 or path.boundingRect().left() > rc.right() + 40:
                    continue
                c2 = QColor(col)
                c2.setAlpha(190)
                p.setBrush(c2)
                p.setPen(QPen(col.darker(170), 1))
                p.drawPath(path)
            f = p.font()
            f.setPixelSize(9)
            f.setBold(True)
            p.setFont(f)
            for n in self.notes:
                if not (P_LO <= n[2] <= P_HI):
                    continue
                if KW + (n[0] + n[1]) * CW < rc.left() - 40 or KW + n[0] * CW > rc.right() + 40:
                    continue
                r = self.note_rect(n)
                chosen = n[6] in self.sel
                p.setBrush(col.lighter(135) if chosen else col)
                p.setPen(QPen(QColor("#ffffff") if chosen else col.darker(170), 2 if chosen else 1))
                p.drawRoundedRect(r, 2, 2)
                if n is self.slur_from:  # waiting for the second click of a slur
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(QColor("#ffffff"), 2, Qt.PenStyle.DashLine))
                    p.drawRoundedRect(r.adjusted(-2, -2, 2, 2), 3, 3)
                if choir and r.width() > 10:
                    v = n[5] if n[5] >= 0 else self.ch.vowel
                    p.setPen(QColor("#1b1b1b"))
                    p.drawText(r, Qt.AlignmentFlag.AlignCenter, VOWEL_LETTERS[min(4, max(0, int(v)))])
            if self.rb is not None:
                p.setPen(QPen(QColor(ACC), 1, Qt.PenStyle.DashLine))
                p.setBrush(QColor(255, 154, 46, 45))
                p.drawRect(self.rb)
        if self.px >= 0:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            p.fillRect(QRectF(self.px - 1, 0, 3, H), QColor(255, 154, 46, 200))

    # ---- mouse
    def _near_playbar(self, x):
        return x >= self.KW and abs(x - self.px) <= 5

    def _tile(self, x, y):
        return int(max(0, min(self.main.project.length - 1, int((x - self.KW) // self.CW)))), self.row_pitch(y)

    def mousePressEvent(self, e):
        x, y, btn, mods = e.position().x(), e.position().y(), e.button(), e.modifiers()
        ch = self.ch
        if not ch:
            return
        if btn == Qt.MouseButton.LeftButton and self._near_playbar(x):
            self.mode = "playbar"
            return
        if self.clip_mode():
            if x >= self.KW:
                self.clip_press(btn, mods, (x - self.KW) / self.CW, 6.0 / self.CW, e.globalPosition().toPoint())
                self.update()
                self.main.rack.refresh_steps()
            return
        eng, L = self.main.engine, self.main.project.length
        if x < self.KW:
            if btn == Qt.MouseButton.LeftButton:
                self.key = self.row_pitch(y)
                eng.note_on(self.pch(), self.key, 0.9)
            return
        n, sx = self.note_at(x, y), (x - self.KW) / self.CW
        if btn == Qt.MouseButton.MiddleButton:
            if n is not None:  # middle-click a note: select it (if needed) and show the copy menu
                if n[6] not in self.sel:
                    if not (mods & Qt.KeyboardModifier.ControlModifier):
                        self.sel.clear()
                    self.sel.add(n[6])
                self.update()
                self.note_menu()
                self.mode = self.mid = self.rb = None
            else:  # empty tile: drag = selection box, plain click = paste
                self.mid = dict(x0=x, y0=y, moved=False, add=bool(mods & Qt.KeyboardModifier.ControlModifier))
                self.rb = QRectF(x, y, 0, 0)
            return
        if btn == Qt.MouseButton.RightButton:
            if n is not None:
                self.notes.remove(n)
                self.sel.discard(n[6])
                for m in self.notes:
                    if m[4] == n[6]:
                        m[4] = 0
            else:  # right-click on a slur band removes that slur
                for a, b in self.slur_pairs():
                    if self.ribbon(a, b).contains(QPointF(x, y)):
                        a[4] = 0
                        for m in self.notes:
                            if m[6] == a[6]:
                                m[4] = 0
                        break
            self.update()
            self.main.rack.refresh_steps()
            return
        synth = self.synthlike()
        if synth and (self.slur_from is not None or self.slur_mode or (mods & Qt.KeyboardModifier.ControlModifier)):
            self.slur_click(n)
            return
        if n is not None and synth and (mods & Qt.KeyboardModifier.AltModifier) and self.fam() == "choir":
            n[5] = self.new_vowel  # sing a different vowel on this note
            self.update()
            return
        if n is not None and n[6] in self.sel:  # grab the whole selection
            end_zone = sx > n[0] + n[1] - 0.3
            self.group = dict(ref=n, sx0=sx, p0=self.row_pitch(y), len0=n[1],
                              items=[(m, m[0], int(m[2]), m[1]) for m in self.selected_notes()])
            self.mode = "gresize" if end_zone else "gmove"
            eng.note_on(self.pch(), int(n[2]), 0.9, hold=int(0.2 * SR))
            return
        if n is None and self.sel:  # click on an empty tile: unselect everything
            self.sel.clear()
            self.update()
            return
        if n is not None:
            self.sel.clear()
            self.note = n
            if sx > n[0] + n[1] - 0.3:
                self.mode = "resize"
            else:
                self.mode, self.off = "move", sx - n[0]
        else:
            vow = self.new_vowel if self.fam() == "choir" else -1
            n = mknote(max(0, min(L - 1, int(sx))), self.lastlen, self.row_pitch(y), 0.9, 0, vow)
            self.notes.append(n)
            self.note, self.mode = n, "resize"
        eng.note_on(self.pch(), int(n[2]), 0.9, hold=int(0.25 * SR))
        self.update()
        self.main.rack.refresh_steps()

    def mouseMoveEvent(self, e):
        x, y, L = e.position().x(), e.position().y(), self.main.project.length
        if self.mode == "playbar":
            step = (x - self.KW) / self.CW
            self.main.seek_step(step if e.modifiers() & Qt.KeyboardModifier.AltModifier else round(step))
            return
        if self.ch and self.clip_mode():
            sx = (x - self.KW) / self.CW
            if self.cdrag:
                if self.clip_move(sx, e.modifiers()):
                    self.update()
                    self.main.rack.refresh_steps()
            elif not e.buttons():
                self.clip_hover(sx, 6.0 / self.CW)
            return
        if self.mid is not None and (e.buttons() & Qt.MouseButton.MiddleButton):
            self.rb = QRectF(QPointF(self.mid["x0"], self.mid["y0"]), QPointF(x, y)).normalized()
            if abs(x - self.mid["x0"]) > 4 or abs(y - self.mid["y0"]) > 4:
                self.mid["moved"] = True
            self.update()
            return
        if not (e.buttons() & Qt.MouseButton.LeftButton):
            self.setCursor(Qt.CursorShape.SizeHorCursor if self._near_playbar(x) else Qt.CursorShape.ArrowCursor)
            return
        sx = (x - self.KW) / self.CW
        g = self.group
        if g and self.mode in ("gmove", "gresize"):
            if self.mode == "gmove":
                items = g["items"]
                dstep = round(sx - g["sx0"])
                dstep = max(-min(i[1] for i in items), min(dstep, (L - 1) - max(i[1] for i in items)))
                dp = self.row_pitch(y) - g["p0"]
                dp = max(P_LO - min(i[2] for i in items), min(dp, P_HI - max(i[2] for i in items)))
                for m, st, pi, ln in items:
                    m[0], m[2] = float(st + dstep), int(pi + dp)
                if dp != g.get("last_dp", 0):
                    self.main.engine.note_on(self.pch(), int(g["ref"][2]), 0.9, hold=int(0.15 * SR))
                g["last_dp"] = dp
            else:
                delta = max(1.0, float(round(sx - g["ref"][0]))) - g["len0"]
                for m, st, pi, ln in g["items"]:
                    m[1] = max(1.0, ln + delta)
            self.update()
            self.main.rack.refresh_steps()
            return
        if not self.note:
            return
        n = self.note
        if self.mode == "resize":
            n[1] = self.lastlen = max(1.0, float(round(sx - n[0])))
        else:
            n[0] = float(max(0, min(L - 1, round(sx - self.off))))
            pitch = self.row_pitch(y)
            if pitch != n[2]:
                n[2] = pitch
                self.main.engine.note_on(self.pch(), pitch, 0.9, hold=int(0.2 * SR))
        self.update()
        self.main.rack.refresh_steps()

    def mouseReleaseEvent(self, e):
        x, y = e.position().x(), e.position().y()
        if self.ch and self.clip_mode():
            if self.cdrag:
                self.clip_release()
                self.update()
                self.main.rack.refresh_steps()
        elif e.button() == Qt.MouseButton.MiddleButton and self.mid is not None:
            if self.mid["moved"] and self.rb is not None:  # finish the selection box
                hit = {n[6] for n in self.notes if self.note_rect(n).intersects(self.rb)}
                self.sel = (self.sel | hit) if self.mid["add"] else hit
            elif self.main.note_clipboard and x >= self.KW:  # plain middle-click on an empty tile: paste
                step, pitch = self._tile(x, y)
                self.paste_at(step, pitch)
            self.mid = self.rb = None
            self.update()
        if self.key is not None:
            self.main.engine.note_off(self.pch(), self.key)
        if self.group or self.note:
            self.main.ensure_length(max([n[0] + n[1] for n in (self.notes if self.ch else [])] + [0]))
        self.key = self.mode = self.note = self.group = None


class Mixer(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main, self.ctrl = main, {}
        self.lay = QtWidgets.QHBoxLayout(self)
        self.lay.setSpacing(5)
        self.lay.setContentsMargins(6, 4, 6, 4)

    def sync(self, ch):
        c = self.ctrl.get(ch)
        if c:
            c["vol"].blockSignals(True)
            c["vol"].setValue(int(ch.vol * 100))
            c["vol"].blockSignals(False)
            c["pan"].setValue(ch.pan)
            c["mute"].blockSignals(True)
            c["mute"].setChecked(ch.mute)
            c["mute"].blockSignals(False)

    def rebuild(self):
        clear(self.lay)
        self.ctrl = {}
        for ch in self.main.project.channels:
            self.lay.addWidget(self.strip(ch))
        self.lay.addStretch()
        self.lay.addWidget(self.master())

    def strip(self, ch):
        f = QtWidgets.QWidget()
        f.setObjectName("strip")
        f.setFixedWidth(74)
        v = QtWidgets.QVBoxLayout(f)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)
        nm = QtWidgets.QLabel(ch.name)
        nm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nm.setStyleSheet(f"background:{ch.color};color:#111;font-weight:bold;border-radius:3px;padding:2px;")
        v.addWidget(nm)
        c = {}
        for label, attr, lo, hi in [("Pan", "pan", -1, 1), ("Cut", "cut", 0, 1), ("Drv", "drive", 0, 1),
                                    ("Rev", "rev", 0, 1), ("Dly", "dly", 0, 1)]:
            k = Knob(label, getattr(ch, attr), lo, hi, size=30)
            k.default = 1.0 if attr == "cut" else (0.0)
            k.changed.connect(lambda val, a=attr: (setattr(ch, a, val), self.main.sync(ch, from_mixer=True)))
            v.addWidget(k, alignment=Qt.AlignmentFlag.AlignHCenter)
            if attr == "pan":
                c["pan"] = k
        sl = QtWidgets.QSlider(Qt.Orientation.Vertical)
        sl.setRange(0, 100)
        sl.setValue(int(ch.vol * 100))
        sl.setMinimumHeight(90)
        sl.valueChanged.connect(lambda val: (setattr(ch, "vol", val / 100), self.main.sync(ch, from_mixer=True)))
        v.addWidget(sl, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        m = QtWidgets.QPushButton("M")
        m.setCheckable(True)
        m.setChecked(ch.mute)
        m.toggled.connect(lambda on: (setattr(ch, "mute", on), self.main.sync(ch, from_mixer=True)))
        v.addWidget(m)
        c.update(vol=sl, mute=m)
        self.ctrl[ch] = c
        return f

    def master(self):
        f = QtWidgets.QWidget()
        f.setObjectName("strip")
        f.setFixedWidth(80)
        v = QtWidgets.QVBoxLayout(f)
        nm = QtWidgets.QLabel("MASTER")
        nm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nm.setStyleSheet(f"background:{ACC};color:#111;font-weight:bold;border-radius:3px;padding:2px;")
        v.addWidget(nm)
        sl = QtWidgets.QSlider(Qt.Orientation.Vertical)
        sl.setRange(0, 100)
        sl.setValue(int(self.main.project.master * 100))
        sl.valueChanged.connect(lambda val: (setattr(self.main.project, "master", val / 100), self.main.master_knob.setValue(val / 100)))
        v.addWidget(sl, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.master_slider = sl
        return f


class Settings(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setMinimumWidth(250)
        self.lay = QtWidgets.QVBoxLayout(self)

    def fx_ui(self, ch):
        """Effects for an audio-clip track: Auto-Tune, Auto-Harmonizer, Vocoder."""
        main, fx, base = self.main, ch.fx, default_fx()

        def setter(section, key, offline):
            def f(v):
                (fx[section] if section else fx)[key] = v
                main.fx_changed(ch, offline)
            return f

        def combo(items, current, section, key, offline, conv=None):
            cb = QtWidgets.QComboBox()
            cb.addItems(items)
            if current in items:
                cb.setCurrentIndex(items.index(current))
            cb.activated.connect(lambda i: setter(section, key, offline)(conv(items[i]) if conv else items[i]))
            return cb

        def knobs(specs, section, offline):
            grid = QtWidgets.QGridLayout()
            for i, (nm, key, lo, hi) in enumerate(specs):
                k = Knob(nm, fx[section][key], lo, hi, size=40)
                k.default = base[section][key]
                k.changed.connect(setter(section, key, offline))
                grid.addWidget(k, i // 3, i % 3)
            return grid

        def check(text, section, key, offline, tip=""):
            c = QtWidgets.QCheckBox(text)
            c.setChecked(bool(fx[section].get(key)))
            c.setToolTip(tip)
            c.toggled.connect(setter(section, key, offline))
            return c

        def group(title, section, offline, tip):
            g = QtWidgets.QGroupBox(title)
            g.setCheckable(True)
            g.setChecked(bool(fx[section]["on"]))
            g.setToolTip(tip)
            g.toggled.connect(setter(section, "on", offline))
            return g, QtWidgets.QVBoxLayout(g)

        head = QtWidgets.QLabel("Effects on this track")
        head.setStyleSheet(f"font-weight:bold; color:{ACC}; padding-top:6px;")
        self.lay.addWidget(head)
        draw = QtWidgets.QPushButton(f"Draw notes for the effects...  ({len(ch.pnotes)} notes)")
        draw.setToolTip("Switches the piano roll to this track's notes.  Draw notes and chords there like on any synth:\n"
                        "Auto-Tune sings them, the Harmonizer turns chords into harmonies, the Vocoder plays them.")
        draw.clicked.connect(lambda: main.set_notes_mode(True))
        self.lay.addWidget(draw)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Key"))
        row.addWidget(combo(KEY_NAMES, KEY_NAMES[int(fx["key"])], None, "key", True, KEY_NAMES.index))
        row.addWidget(combo(list(SCALES), fx["scale"], None, "scale", True))
        self.lay.addLayout(row)
        det = QtWidgets.QPushButton("Detect key from this clip")
        det.setToolTip("Listens to the clip and picks the key and major/minor scale that fit it best")
        det.clicked.connect(lambda: main.detect_key(ch))
        self.lay.addWidget(det)

        g, v = group("Auto-Tune", "tune", True, "Snaps the voice to the nearest note of the key above. Works on one voice or instrument at a time.")
        v.addLayout(knobs([("Speed", "speed", 0.0, 0.4), ("Amount", "amount", 0.0, 1.0)], "tune", True))
        v.addWidget(check("Follow my notes (instead of the key)", "tune", "notes", True,
                          "The voice is tuned to the notes you draw for the effects (chords = any chord tone).\n"
                          "Where no note is drawn the voice is left alone."))
        v.addWidget(check("Stay near the voice's octave", "tune", "fold", True))
        tip = QtWidgets.QLabel("Speed 0 = instant, robotic tune.\nHigher = gentle, natural correction.")
        tip.setStyleSheet(f"color:{DIM};")
        v.addWidget(tip)
        self.lay.addWidget(g)

        g, v = group("Auto-Harmonizer", "harm", True, "Adds harmony voices that follow the melody and stay in the key.")
        form = QtWidgets.QFormLayout()
        for i in (1, 2, 3):
            form.addRow(f"Voice {i}", combo(list(HARM_CHOICES), fx["harm"][f"v{i}"], "harm", f"v{i}", True))
        v.addLayout(form)
        v.addWidget(check("Chord from my notes (up to 4 voices)", "harm", "notes", True,
                          "Instead of the fixed intervals above: every note of the chord you draw for the effects\n"
                          "becomes a harmony voice, so the harmonies change as your chords change."))
        v.addWidget(check("Stay near the voice's octave", "harm", "fold", True))
        v.addLayout(knobs([("Lvl 1", "l1", 0, 1), ("Lvl 2", "l2", 0, 1), ("Lvl 3", "l3", 0, 1),
                           ("Dry", "dry", 0, 1), ("Spread", "spread", 0, 1), ("Human", "human", 0, 1), ("Chord", "lc", 0, 1)],
                          "harm", True))
        self.lay.addWidget(g)

        g, v = group("Vocoder", "voc", False, "The clip's voice shapes a carrier sound, so it 'talks' through the carrier.")
        car = fx["voc"]
        opts = [("notes", 0, "Notes on this track (play it like a synth)"), ("drone", 0, "Saw chord drone"),
                ("noise", 0, "Noise (whisper)")] + \
               [("track", c.uid, f"Track: {c.name}") for c in main.project.channels if c.kind == "synth"]
        cb = QtWidgets.QComboBox()
        cb.addItems([o[2] for o in opts])
        cur = next((i for i, o in enumerate(opts) if o[0] == car["carrier"] and (o[0] != "track" or o[1] == car["carrier_uid"])), 0)
        cb.setCurrentIndex(cur)
        cb.setToolTip("Carrier = the sound the voice is imposed on.  Pick an instrument track and play notes/chords on it.")

        def pick(i):
            car["carrier"], car["carrier_uid"] = opts[i][0], opts[i][1]
            main.fx_changed(ch, False)
        cb.activated.connect(pick)
        form = QtWidgets.QFormLayout()
        form.addRow("Carrier", cb)
        form.addRow("Carrier sound", combo(list(INSTRUMENTS), car.get("inst", "Synth Pad"), "voc", "inst", False))
        form.addRow("Drone root", combo(KEY_NAMES, KEY_NAMES[int(car["root"])], "voc", "root", False, KEY_NAMES.index))
        form.addRow("Drone chord", combo(list(CHORDS), car["chord"], "voc", "chord", False))
        form.addRow("Bands", combo([str(x) for x in VOC_BANDS], str(int(car["bands"])), "voc", "bands", False, int))
        v.addLayout(form)
        hide = QtWidgets.QCheckBox("Hide the carrier track (hear only the vocoder)")
        hide.setChecked(bool(car["hide"]))
        hide.toggled.connect(setter("voc", "hide", False))
        v.addWidget(hide)
        v.addLayout(knobs([("Sibil.", "sib", 0, 1), ("Smooth", "smooth", 0.001, 0.05), ("Mix", "mix", 0, 1), ("Gain", "gain", 0, 3)],
                          "voc", False))
        self.lay.addWidget(g)
        self.fx_label = QtWidgets.QLabel(ch.fx_status)
        self.fx_label.setStyleSheet(f"color:{ACC};")
        self.lay.addWidget(self.fx_label)

    def show_channel(self, ch):
        clear(self.lay)
        self.fx_label = None
        if ch is None:
            return
        t = QtWidgets.QLabel(ch.name)
        t.setStyleSheet(f"font:bold 16px; color:{ch.color};")
        self.lay.addWidget(t)
        if ch.kind == "synth":
            sp = ch.spec
            self.lay.addWidget(QtWidgets.QLabel("Instrument"))
            ib = QtWidgets.QComboBox()
            ib.addItems(list(INSTRUMENTS))
            ib.setCurrentText(ch.inst)
            ib.activated.connect(lambda i, ib=ib: self.main.set_instrument(ch, ib.itemText(i)))
            self.lay.addWidget(ib)
            if sp["fam"] == "osc":
                self.lay.addWidget(QtWidgets.QLabel("Oscillator"))
                cb = QtWidgets.QComboBox()
                cb.addItems(Channel.WAVES)
                cb.setCurrentIndex(ch.wave)
                cb.currentIndexChanged.connect(lambda i: setattr(ch, "wave", i))
                self.lay.addWidget(cb)
            if sp["fam"] == "choir":
                self.lay.addWidget(QtWidgets.QLabel("Default vowel"))
                vb = QtWidgets.QComboBox()
                vb.addItems(VOWEL_NAMES)
                vb.setCurrentIndex(int(ch.vowel))
                vb.currentIndexChanged.connect(lambda i: (setattr(ch, "vowel", i), self.main.roll.update()))
                self.lay.addWidget(vb)
            knobs = [("Attack", "a", 0, 1.5), ("Decay", "d", 0, 1.5), ("Sustain", "s", 0, 1), ("Release", "r", 0, 2)]
            if sp["fam"] == "osc" or sp.get("ens"):
                knobs.append(("Detune", "detune", 0, 50))
            knobs.append(("Slur", "glide", 0, 0.5))
            knobs.append(("Vibrato", "vib", 0, 1))
            if sp["fam"] != "osc":
                knobs.append(("Tone", "tone", 0, 1))
            grid = QtWidgets.QGridLayout()
            for i, (nm, key, lo, hi) in enumerate(knobs):
                k = Knob(nm, getattr(ch, key), lo, hi, size=44)
                if key == "tone":
                    k.default = 0.5
                k.changed.connect(lambda v, key=key: setattr(ch, key, v))
                grid.addWidget(k, i // 3, i % 3)
            self.lay.addLayout(grid)
            hum = QtWidgets.QPushButton("\U0001F3A4  Hum notes into this track...")
            hum.clicked.connect(self.main.open_hum)
            self.lay.addWidget(hum)
            hint = QtWidgets.QLabel("Slur: turn on Slur mode (or hold Ctrl), click the\nfirst note, then click the last note.\n"
                                    "Play with your typing keyboard or a\nMIDI keyboard (Options menu).")
        elif ch.kind == "audio":
            self.lay.addWidget(QtWidgets.QLabel("Audio clip\n" + os.path.basename(ch.path or "(missing file)")))
            if ch.sample is not None:
                self.lay.addWidget(QtWidgets.QLabel(f"Length: {ch.sample.shape[1] / SR:.1f} s"))
            b = QtWidgets.QPushButton("Load another file...")
            b.clicked.connect(lambda: self.main.replace_sample(ch))
            self.lay.addWidget(b)
            self.fx_ui(ch)
            hint = QtWidgets.QLabel("Plays at its original speed.\nDrag a clip to move it, drag its edges to\ncut the length, Ctrl+drag to copy,\n"
                                    "right-click for Copy / Cut / Split / Delete.\n"
                                    "Auto-Tune / Harmonizer work best on one\nvoice or instrument at a time.")
        else:
            src = ch.path or ("Built-in drum: " + str(ch.builtin))
            self.lay.addWidget(QtWidgets.QLabel("Sampler\n" + os.path.basename(src) if ch.path else "Sampler\n" + src))
            b = QtWidgets.QPushButton("Load sample...")
            b.clicked.connect(lambda: self.main.replace_sample(ch))
            self.lay.addWidget(b)
            gate = QtWidgets.QCheckBox("Play only as long as the note")
            gate.setChecked(bool(ch.gate))
            gate.setToolTip("On: the sample is cut off where the note ends (resize notes in the piano roll to\n"
                            "make it play longer). Off: it always plays to the end, like a drum one-shot.")
            gate.toggled.connect(lambda on: setattr(ch, "gate", on))
            self.lay.addWidget(gate)
            hint = QtWidgets.QLabel("Notes pitch-shift the sample\n(root note = C4). Make a note longer in the\n"
                                    "piano roll to let the sample play longer.")
        hint.setStyleSheet(f"color:{DIM};")
        self.lay.addWidget(hint)
        self.lay.addStretch()


class Browser(QtWidgets.QTreeView):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.fs = QtGui.QFileSystemModel()
        root = os.path.expanduser("~")
        self.fs.setRootPath(root)
        self.fs.setNameFilters(["*" + e for e in AUDIO_EXT + MIDI_EXT + (".pyfl",)])
        self.fs.setNameFilterDisables(False)
        self.setModel(self.fs)
        self.setRootIndex(self.fs.index(root))
        for c in (1, 2, 3):
            self.hideColumn(c)
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setMinimumWidth(200)
        self.doubleClicked.connect(self.open_it)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menu)

    def path(self, idx):
        return self.fs.filePath(idx)

    def preview(self, p):
        """Only plays when asked for (right-click > Preview); the first 8 seconds, stoppable with Stop preview."""
        self.main.engine.stop_previews()
        if os.path.isfile(p) and os.path.getsize(p) < 60_000_000:
            try:
                ch = Channel("preview", "sample", load_sample(p)[:, :SR * 8])
                ch.is_preview, ch.gate = True, False
                self.main.engine.note_on(ch, ROOT, 0.9)
            except Exception as ex:
                self.main.flash(f"Could not preview: {ex}")

    def open_it(self, idx):
        p = self.path(idx)
        if os.path.isfile(p):
            self.main.handle_files([p])

    def menu(self, pos):
        idx = self.indexAt(pos)
        p = self.path(idx) if idx.isValid() else ""
        if not (p and os.path.isfile(p) and p.lower().endswith(AUDIO_EXT)):
            return
        m = QtWidgets.QMenu(self)
        a_pre = m.addAction("Preview (play)")
        a_stop = m.addAction("Stop preview")
        m.addSeparator()
        a_clip = m.addAction("Add as audio clip (at the play bar)")
        a_smp = m.addAction("Add as sample (plays as long as the note)")
        act = m.exec(self.viewport().mapToGlobal(pos))
        if act == a_pre:
            self.preview(p)
        elif act == a_stop:
            self.main.engine.stop_previews()
        elif act in (a_clip, a_smp):
            try:
                self.main.add_audio_path(p, "clip" if act == a_clip else "sample")
                self.main.refresh_all()
            except Exception as ex:
                self.main.err(f"Could not load {os.path.basename(p)}:\n{ex}")


class Dropper(QtCore.QObject):
    def __init__(self, main):
        super().__init__(main)
        self.main = main

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QtCore.QEvent.Type.DragEnter, QtCore.QEvent.Type.DragMove) and ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            return True
        if t == QtCore.QEvent.Type.Drop and ev.mimeData().hasUrls():
            self.main.handle_files([u.toLocalFile() for u in ev.mimeData().urls()])
            ev.acceptProposedAction()
            return True
        return False


class ProjectTabs(QtWidgets.QTabBar):
    """Project tabs.  The close cross only appears on the tab under the mouse (middle-click closes too,
    double-click renames, drag to reorder)."""
    closeRequested = pyqtSignal(int)
    renameRequested = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setMovable(True)
        self.setExpanding(False)
        self.setDrawBase(False)
        self.setUsesScrollButtons(True)
        self.setElideMode(Qt.TextElideMode.ElideRight)
        self.hover, self.hover_x = -1, False

    def tabSizeHint(self, i):
        w = self.fontMetrics().horizontalAdvance(self.tabText(i)) + 58
        return QtCore.QSize(int(min(max(w, 130), 280)), 32)

    def minimumTabSizeHint(self, i):
        return QtCore.QSize(90, 32)

    def _x_rect(self, i):
        r = self.tabRect(i)
        return QRect(r.right() - 26, r.center().y() - 8, 18, 18)

    def mouseMoveEvent(self, e):
        pt = e.position().toPoint()
        i = self.tabAt(pt)
        hx = i >= 0 and self._x_rect(i).contains(pt)
        if i != self.hover or hx != self.hover_x:
            self.hover, self.hover_x = i, hx
            self.update()
        super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        self.hover, self.hover_x = -1, False
        self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        pt = e.position().toPoint()
        i = self.tabAt(pt)
        if i >= 0 and e.button() == Qt.MouseButton.LeftButton and self._x_rect(i).contains(pt):
            self.closeRequested.emit(i)
            return
        if i >= 0 and e.button() == Qt.MouseButton.MiddleButton:
            self.closeRequested.emit(i)
            return
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        pt = e.position().toPoint()
        i = self.tabAt(pt)
        if i >= 0 and not self._x_rect(i).contains(pt):
            self.renameRequested.emit(i)
            return
        super().mouseDoubleClickEvent(e)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fm = self.fontMetrics()
        for i in range(self.count()):
            r = QRectF(self.tabRect(i)).adjusted(1, 4, -1, 0)
            sel, hov = i == self.currentIndex(), i == self.hover
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#4a5560" if sel else ("#313a41" if hov else "#262d32")))
            p.drawRoundedRect(r.adjusted(0, 0, 0, 10), 7, 7)  # bottom corners fall outside the widget and are clipped
            if sel:
                p.setBrush(QColor(ACC))
                p.drawRoundedRect(QRectF(r.left() + 6, r.top(), r.width() - 12, 2.5), 1.2, 1.2)
            text_w = int(r.width() - 44)
            p.setPen(QColor(TXT if sel else DIM))
            p.drawText(QRectF(r.left() + 12, r.top(), text_w, r.height()),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       fm.elidedText(self.tabText(i), Qt.TextElideMode.ElideRight, text_w))
            if hov:  # the close cross, only while the mouse is over this tab
                xr = QRectF(self._x_rect(i))
                if self.hover_x:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor("#c9473b"))
                    p.drawEllipse(xr)
                p.setPen(QPen(QColor("#ffffff") if self.hover_x else QColor(TXT), 1.8, Qt.PenStyle.SolidLine,
                              Qt.PenCapStyle.RoundCap))
                c = xr.center()
                d = 3.6
                p.drawLine(QPointF(c.x() - d, c.y() - d), QPointF(c.x() + d, c.y() + d))
                p.drawLine(QPointF(c.x() - d, c.y() + d), QPointF(c.x() + d, c.y() - d))


# ------------------------------------------------------------ hum -> notes
class HumDialog(QtWidgets.QDialog):
    """Record the microphone while you hum, then write the detected notes into the selected track."""
    QS = [("Fine  (1/4 step)", 0.25), ("1/16 note  (1 step)", 1.0), ("1/8 note  (2 steps)", 2.0), ("1/4 note  (4 steps)", 4.0)]

    def __init__(self, main, ch):
        super().__init__(main)
        self.main, self.ch = main, ch
        self.setWindowTitle("Hum notes")
        self.setMinimumWidth(480)
        self.stream, self.chunks, self.frames, self.tail = None, [], 0, np.zeros(0, np.float32)
        self.level, self.state, self.t0, self.sr, self.lat, self.start_step = 0.0, "idle", 0, SR, 0.0, 0.0
        self.click_ch = Channel("click", "sample", make_drums()["hat"])
        self.click_n, self.click_timer, self.ui_timer = 0, QTimer(self), QTimer(self)
        self.click_timer.timeout.connect(self.click_tick)
        self.ui_timer.timeout.connect(self.ui_tick)

        v = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(f"Recording into:  {ch.name}   ({ch.inst})")
        title.setStyleSheet(f"font:bold 15px; color:{ch.color};")
        v.addWidget(title)
        info = QtWidgets.QLabel("Hum or sing one note at a time into your microphone. When you stop, the notes and their\n"
                                "lengths are written into this track's piano roll. Use headphones if the song plays along.")
        info.setStyleSheet(f"color:{DIM};")
        v.addWidget(info)
        form = QtWidgets.QFormLayout()
        self.dev = QtWidgets.QComboBox()
        devs = self.input_devices()
        for i, nm in devs:
            self.dev.addItem(nm, i)
        try:
            d = sd.default.device[0]
            if d is not None and d >= 0:
                k = self.dev.findData(d)
                if k >= 0:
                    self.dev.setCurrentIndex(k)
        except Exception:
            pass
        form.addRow("Microphone", self.dev)
        self.q_box = QtWidgets.QComboBox()
        for nm, _ in self.QS:
            self.q_box.addItem(nm)
        self.q_box.setCurrentIndex(1)
        form.addRow("Note timing", self.q_box)
        self.start_box = QtWidgets.QComboBox()
        self.start_box.addItems(["At the play bar", "At the start of the song"])
        form.addRow("Write notes", self.start_box)
        self.oct = QtWidgets.QSpinBox()
        self.oct.setRange(-3, 3)
        self.oct.setToolTip("Shift the written notes up/down in octaves (your voice vs. the instrument's range)")
        form.addRow("Octave shift", self.oct)
        v.addLayout(form)
        self.chk_count = QtWidgets.QCheckBox("Count-in (4 clicks) before recording")
        self.chk_play = QtWidgets.QCheckBox("Play the song while I hum (use headphones)")
        self.chk_rep = QtWidgets.QCheckBox("Replace existing notes in the recorded range")
        self.chk_rep.setChecked(True)
        for c in (self.chk_count, self.chk_play, self.chk_rep):
            v.addWidget(c)
        self.big = QtWidgets.QLabel("--")
        self.big.setObjectName("bignote")
        self.big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.big)
        self.meter = QtWidgets.QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        v.addWidget(self.meter)
        self.status = QtWidgets.QLabel("Ready. Press Record, then hum.")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        self.btn = QtWidgets.QPushButton("\u25CF  Record")
        self.btn.setObjectName("rec")
        self.btn.clicked.connect(self.toggle)
        self.undo_btn = QtWidgets.QPushButton("Undo last hum")
        self.undo_btn.clicked.connect(self.do_undo)
        self.undo_btn.setEnabled(main.hum_backup is not None)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        for w in (self.btn, self.undo_btn):
            row.addWidget(w)
        row.addStretch()
        row.addWidget(close)
        v.addLayout(row)
        if not devs:
            self.btn.setEnabled(False)
            self.status.setText("No microphone found. Plug one in (or allow microphone access for this app) and reopen.")

    @staticmethod
    def input_devices():
        out = []
        try:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    out.append((i, f"{d['name']}  ({sd.query_hostapis(d['hostapi'])['name']})"))
        except Exception:
            pass
        return out

    # ---- recording
    def cb(self, indata, frames, t, status):
        x = indata[:, 0].copy()
        self.chunks.append(x)
        self.frames += frames
        self.level = float(np.abs(x).max())
        self.tail = np.concatenate([self.tail, x])[-4096:]

    def toggle(self):
        if self.state == "idle":
            self.start()
        else:
            self.stop()

    def start(self):
        idx = self.dev.currentData()
        try:
            self.sr = int(sd.query_devices(idx)["default_samplerate"])
            self.stream = sd.InputStream(device=idx, channels=1, samplerate=self.sr, blocksize=256,
                                         dtype="float32", callback=self.cb)
            self.stream.start()
        except Exception as ex:
            self.stream = None
            self.status.setText(f"Could not open the microphone:\n{ex}")
            return
        self.chunks, self.frames, self.tail, self.level = [], 0, np.zeros(0, np.float32), 0.0
        self.main.engine.pause()
        self.start_step = 0.0 if self.start_box.currentIndex() == 1 else float(round(self.main.engine.pos / self.main.engine.step_samples()))
        self.btn.setText("\u25A0  Stop && write notes")
        self.ui_timer.start(50)
        if self.chk_count.isChecked():
            self.state, self.click_n = "count", 0
            self.click_tick()
            self.click_timer.start(int(60000.0 / self.main.project.bpm))
        else:
            self.begin_take()

    def click_tick(self):
        if self.click_n < 4:
            self.main.engine.note_on(self.click_ch, ROOT + (7 if self.click_n == 0 else 0), 0.9)
            self.status.setText(f"Count-in...  {4 - self.click_n}")
            self.click_n += 1
        else:
            self.click_timer.stop()
            self.main.engine.note_on(self.click_ch, ROOT + 12, 0.9)
            self.begin_take()

    def begin_take(self):
        self.state, self.t0 = "rec", self.frames
        backing = self.chk_play.isChecked()
        in_lat = float(getattr(self.stream, "latency", 0.0) or 0.0)
        out_lat = 0.0
        if backing:
            st = self.main.audio.stream
            out_lat = float(getattr(st, "latency", 0.0) or 0.0) if st else 0.0
            self.main.seek_step(self.start_step)
            self.main.play_origin = self.main.engine.pos
            self.main.engine.play()
        self.lat = min(0.4, max(0.0, in_lat + out_lat))
        self.status.setText("Recording... hum now. Press Stop when you're done.")

    def ui_tick(self):
        self.meter.setValue(min(100, int(self.level * 300)))
        self.level *= 0.7
        if len(self.tail) >= 2048 and self.level > 0.01:
            f0, conf = yin_frame(self.tail[-2048:].astype(np.float64), self.sr)
            self.big.setText(midi_name(hz_to_midi(f0)) if (f0 > 0 and conf > 0.7) else "--")
        else:
            self.big.setText("--")
        if self.state == "rec":
            self.status.setText(f"Recording...  {(self.frames - self.t0) / self.sr:5.1f} s   (press Stop when you're done)")

    def teardown(self):
        self.click_timer.stop()
        self.ui_timer.stop()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.state != "idle":
            self.main.engine.pause()
            self.main.seek_step(self.start_step)
        self.main.sync_play_btn()

    def stop(self):
        was = self.state
        self.teardown()
        self.state = "idle"
        self.btn.setText("\u25CF  Record")
        self.meter.setValue(0)
        self.big.setText("--")
        if was != "rec" or self.frames - self.t0 < self.sr * 0.2:
            self.status.setText("Nothing recorded. Press Record, then hum.")
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            audio = np.concatenate(self.chunks)[self.t0:]
            events = hum_to_events(audio, self.sr)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not events:
            self.status.setText("I couldn't hear a clear pitch. Hum a little louder or closer to the microphone, "
                                "one steady note at a time, and try again.")
            return
        n = self.write_notes(events)
        self.status.setText(f"Wrote {n} notes into {self.ch.name}. Press Play to hear them, or record again "
                            f"(\"Replace\" overwrites the same range).")
        self.undo_btn.setEnabled(True)

    def write_notes(self, events):
        main, ch, p = self.main, self.ch, self.main.project
        sec = 60.0 / p.bpm / 4
        q = self.QS[self.q_box.currentIndex()][1]
        shift = 12 * self.oct.value()
        notes = []
        for ta, tb, midi, loud in events:
            a = self.start_step + max(0.0, ta - self.lat) / sec
            b = self.start_step + max(0.0, tb - self.lat) / sec
            a, b = round(a / q) * q, round(b / q) * q
            if b - a < q:
                b = a + q
            pitch = int(min(P_HI, max(P_LO, int(midi) + shift)))
            notes.append(mknote(a, b - a, pitch, min(1.0, max(0.45, 0.45 + 0.55 * float(loud)))))
        out = []
        for nt in notes:  # keep the melody monophonic after quantising
            if out:
                prev = out[-1]
                if nt[0] < prev[0] + prev[1]:
                    newlen = nt[0] - prev[0]
                    if newlen >= q - 1e-9:
                        prev[1] = newlen
                    else:
                        nt[0] = prev[0] + prev[1]
            out.append(nt)
        lo, hi = out[0][0], max(n[0] + n[1] for n in out)
        main.hum_backup = (ch, [list(n) for n in ch.notes])
        if self.chk_rep.isChecked():
            ch.notes[:] = [n for n in ch.notes if not (n[0] < hi and n[0] + n[1] > lo)]
        ch.notes.extend(out)
        if hi > p.length:
            p.length = fit_length(hi)
            main.refresh_all()
            out = [n for n in out if n[0] < p.length]
        main.rack.refresh_steps()
        main.roll.update()
        main.show_notes(out)
        return len(out)

    def do_undo(self):
        if self.main.undo_hum():
            self.status.setText("Undid the last hum recording.")
            self.undo_btn.setEnabled(False)

    def reject(self):
        self.teardown()
        self.state = "idle"
        super().reject()


KEYMAP = {k: i for i, k in enumerate("zsxdcvgbhnjm")}
KEYMAP.update({k: 12 + i for i, k in enumerate("q2w3er5t6y7ui9o0p")})


class Doc:
    """One open project (one tab): the project, its own audio engine and its UI state."""

    def __init__(self, project, file=None, name=None):
        self.project, self.file = project, file
        self.engine = Engine(project)
        self.name = name or (os.path.splitext(os.path.basename(file))[0] if file else "Untitled")
        self.sel, self.play_origin, self.hum_backup = None, 0, None
        self.sig, self.cached = None, False

    @staticmethod
    def signature(project):
        return hash(json.dumps(project.to_json(), sort_keys=True, default=str))

    def mark_saved(self):
        self.sig, self.cached = Doc.signature(self.project), False

    def dirty(self):
        return self.sig != Doc.signature(self.project)


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyFL Studio")
        self.resize(1500, 900)
        first = Doc(Project(), name="Untitled")
        default_project(first.project)
        first.mark_saved()
        self.docs, self.cur = [first], first
        self.project, self.engine = first.project, first.engine
        self.audio = Audio(self.engine)
        self.sel, self.keys_down, self.midi_in = None, {}, None
        self.play_origin, self.hum_backup, self._ticks = 0, None, 0
        # clipboards live on the window (not on a project), so notes can be pasted into other open projects
        self.note_clipboard, self.clip_clipboard = None, None

        self.rack, self.mixer, self.settings = Rack(self), Mixer(self), Settings(self)
        self.roll = PianoRoll(self)
        self.roll_scroll = QtWidgets.QScrollArea()
        self.roll_scroll.setWidget(self.roll)
        # piano-roll ruler (play bar) scrolls horizontally together with the roll
        self.roll_ruler = Ruler(self, PianoRoll.CW, PianoRoll.KW, "TIME", pad=40)
        self.ruler_scroll = QtWidgets.QScrollArea()
        self.ruler_scroll.setWidget(self.roll_ruler)
        self.ruler_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.ruler_scroll.setFixedHeight(Ruler.H + 2)
        self.ruler_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ruler_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.roll_scroll.horizontalScrollBar().valueChanged.connect(self.ruler_scroll.horizontalScrollBar().setValue)
        central = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(central)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self.build_roll_bar())
        cv.addWidget(self.ruler_scroll)
        cv.addWidget(self.roll_scroll, 1)
        self.setCentralWidget(central)

        self.browser = Browser(self)
        self.settings_scroll = QtWidgets.QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setWidget(self.settings)
        docks = {}
        for title, w, area in [("Browser", self.browser, Qt.DockWidgetArea.LeftDockWidgetArea),
                               ("Channel rack", self.rack, Qt.DockWidgetArea.TopDockWidgetArea),
                               ("Mixer", self.mixer, Qt.DockWidgetArea.BottomDockWidgetArea),
                               ("Channel settings", self.settings_scroll, Qt.DockWidgetArea.RightDockWidgetArea)]:
            d = QtWidgets.QDockWidget(title, self)
            d.setWidget(w)
            d.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
            self.addDockWidget(area, d)
            docks[title] = d
        self.docks = docks
        self.resizeDocks([docks["Channel rack"], docks["Mixer"]], [270, 270], Qt.Orientation.Vertical)

        self.build_tabs()
        self.build_toolbar()
        self.build_menu()
        self.drop = Dropper(self)
        for w in (self, self.rack.scroll.viewport(), self.roll_scroll.viewport(), self.ruler_scroll.viewport(), self.mixer):
            w.setAcceptDrops(True)
            w.installEventFilter(self.drop)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.toggle_play)
        QtGui.QShortcut(QtGui.QKeySequence("Home"), self, activated=lambda: self.seek_step(0))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Tab"), self, activated=lambda: self.step_tab(1))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Tab"), self, activated=lambda: self.step_tab(-1))
        self.refresh_all()
        QTimer.singleShot(0, lambda: self.roll_scroll.verticalScrollBar().setValue((P_HI - 76) * PianoRoll.RH))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    # ---- UI construction
    def build_roll_bar(self):
        bar = QtWidgets.QWidget()
        bar.setObjectName("rollbar")
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        self.roll_title = QtWidgets.QLabel("Piano roll")
        self.roll_title.setStyleSheet("font-weight:bold;")
        h.addWidget(self.roll_title)
        self.slur_btn = QtWidgets.QPushButton("\u2312  Slur mode")
        self.slur_btn.setCheckable(True)
        self.slur_btn.setToolTip("Slur: click the FIRST note, then click the LAST note (any distance apart).\n"
                                 "They become one smooth, connected sound with a pitch glide between them.\n"
                                 "Clicking a slurred first note removes the slur; right-click the band removes it too.\n"
                                 "Holding Ctrl while clicking works without turning the mode on.")
        self.slur_btn.toggled.connect(lambda on: setattr(self.roll, "slur_mode", on))
        h.addWidget(self.slur_btn)
        self.vowel_lbl = QtWidgets.QLabel("Vowel:")
        self.vowel_box = QtWidgets.QComboBox()
        self.vowel_box.addItems(VOWEL_NAMES)
        self.vowel_box.setToolTip("Vowel for new notes.  Alt+click an existing note to change its vowel.")
        self.vowel_box.currentIndexChanged.connect(lambda i: setattr(self.roll, "new_vowel", i))
        h.addWidget(self.vowel_lbl)
        h.addWidget(self.vowel_box)
        self.notes_btn = QtWidgets.QPushButton("\u266B  Notes for effects")
        self.notes_btn.setCheckable(True)
        self.notes_btn.setToolTip("Audio tracks only.  OFF: edit the audio clips.  ON: draw notes and chords that play the effects\n"
                                  "(Auto-Tune targets, harmony chords, vocoder synth) - just like on any synth.")
        self.notes_btn.toggled.connect(self.set_notes_mode)
        h.addWidget(self.notes_btn)
        self.hum_btn = QtWidgets.QPushButton("\U0001F3A4  Hum notes...")
        self.hum_btn.setToolTip("Hum into the microphone: the notes are written into the selected instrument track")
        self.hum_btn.clicked.connect(self.open_hum)
        h.addWidget(self.hum_btn)
        self.roll_hint = QtWidgets.QLabel("")
        self.roll_hint.setStyleSheet(f"color:{ACC};font-weight:bold;")
        h.addWidget(self.roll_hint)
        h.addStretch()
        tip = QtWidgets.QLabel("click = note   right-click = delete   middle-drag = select   middle-click note = copy menu   "
                               "middle-click empty = paste")
        tip.setStyleSheet(f"color:{DIM};")
        h.addWidget(tip)
        return bar

    def build_toolbar(self):
        tb = self.addToolBar("Transport")
        tb.setMovable(False)
        restart = QtWidgets.QPushButton("\u23EE")
        restart.setObjectName("restart")
        restart.setToolTip("Restart: jump to the start of the song and play")
        restart.clicked.connect(self.restart)
        self.play_btn = QtWidgets.QPushButton("\u25B6")
        self.play_btn.setObjectName("play")
        self.play_btn.setCheckable(True)
        self.play_btn.setToolTip("Play from the play bar / pause")
        self.play_btn.clicked.connect(self.toggle_play)
        stop = QtWidgets.QPushButton("\u25A0")
        stop.setObjectName("stop")
        stop.setToolTip("Stop and return to where you pressed play (press again to go to the start)")
        stop.clicked.connect(self.stop)
        self.time = QtWidgets.QLabel("001:1:1")
        self.time.setObjectName("time")
        self.bpm = QtWidgets.QDoubleSpinBox()
        self.bpm.setObjectName("bpm")
        self.bpm.setRange(40, 300)
        self.bpm.setDecimals(2)
        self.bpm.setValue(self.project.bpm)
        self.bpm.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.bpm.valueChanged.connect(self.set_bpm)
        self.bpm.editingFinished.connect(self.bpm.clearFocus)
        self.len_box = QtWidgets.QComboBox()
        self.len_box.addItems([str(c) for c in LEN_CHOICES])
        self.len_box.currentTextChanged.connect(self.set_len)
        self.follow = QtWidgets.QCheckBox("Follow")
        self.follow.setChecked(True)
        self.follow.setToolTip("Scroll the view to keep the play bar visible while playing")
        self.master_knob = Knob("Master", self.project.master, 0, 1, size=34, show=False)
        self.master_knob.changed.connect(self.set_master)
        self.meter = QtWidgets.QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        self.meter.setFixedWidth(120)
        for w in (restart, self.play_btn, stop, self.time, QtWidgets.QLabel("BPM"), self.bpm, QtWidgets.QLabel("Steps"),
                  self.len_box, self.follow, QtWidgets.QLabel("Master"), self.master_knob, self.meter):
            tb.addWidget(w)

    def build_menu(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        for item in [("New project (new tab)", self.new_tab, "Ctrl+N"), ("Open project...", self.open_project, "Ctrl+O"),
                     ("Save project", self.save_project, "Ctrl+S"), ("Save project as...", self.save_project_as, "Ctrl+Shift+S"),
                     ("Close tab", self.close_current_tab, "Ctrl+W"), None,
                     ("Import audio file (mp3 / wav / flac)...", self.add_audio_dialog, "Ctrl+I"),
                     ("Import MIDI...", self.import_midi_dialog, None), ("Export MIDI...", self.export_midi_dialog, None),
                     ("Export WAV...", self.export_wav, None), None, ("Quit", self.close, "Ctrl+Q")]:
            if item is None:
                f.addSeparator()
                continue
            text, fn, sc = item
            a = f.addAction(text, fn)
            if sc:
                a.setShortcut(sc)
        e = mb.addMenu("&Edit")
        for text, fn, sc in [("Copy selected notes", lambda: self.roll.copy_selected(), "Ctrl+C"),
                             ("Cut selected notes", lambda: self.roll.cut_selected(), "Ctrl+X"),
                             ("Paste at the play bar", lambda: self.roll.paste_at(
                                 int(round(self.engine.pos / self.engine.step_samples())), None), "Ctrl+V"),
                             ("Delete selected notes", lambda: self.roll.delete_selected(), "Del"),
                             ("Select all notes", lambda: self.roll.select_all(), "Ctrl+A")]:
            e.addAction(text, fn).setShortcut(sc)
        e.addSeparator()
        e.addAction("Undo last hum recording", self.undo_hum)
        a = mb.addMenu("&Add track")
        build_add_menu(a, self)
        a.addSeparator()
        h = a.addAction("Hum notes into the selected track...", self.open_hum)
        h.setShortcut("Ctrl+H")
        self.midi_menu = mb.addMenu("&Options").addMenu("MIDI input")
        self.midi_menu.aboutToShow.connect(self.fill_midi_menu)
        v = mb.addMenu("&View")
        for d in self.docks.values():
            v.addAction(d.toggleViewAction())

    # ---- project tabs
    def build_tabs(self):
        tb = QtWidgets.QToolBar("Projects")
        tb.setMovable(False)
        self.tabs = ProjectTabs()
        plus = QtWidgets.QPushButton("+")
        plus.setFixedSize(28, 28)
        plus.setToolTip("New project (Ctrl+N)")
        plus.clicked.connect(self.new_tab)
        box = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)
        hl.addWidget(self.tabs)
        hl.addWidget(plus)
        hl.addStretch()
        box.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        tb.addWidget(box)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self.tabs.addTab(self.cur.name)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.tabs.tabMoved.connect(self.on_tab_moved)
        self.tabs.closeRequested.connect(self.on_close_tab)
        self.tabs.renameRequested.connect(self.on_rename_tab)

    def untitled_name(self):
        names = {d.name for d in self.docs}
        if "Untitled" not in names:
            return "Untitled"
        k = 2
        while f"Untitled {k}" in names:
            k += 1
        return f"Untitled {k}"

    def update_tab_titles(self):
        for i, d in enumerate(self.docs):
            if i < self.tabs.count():
                if d is self.cur:
                    d.cached = d.dirty()
                t = d.name + ("  \u2022" if d.cached else "")  # a dot marks unsaved changes
                if self.tabs.tabText(i) != t:
                    self.tabs.setTabText(i, t)
        self.setWindowTitle(f"PyFL Studio - {self.cur.name}")

    def refresh_tabs(self):
        self.tabs.blockSignals(True)
        while self.tabs.count() < len(self.docs):
            self.tabs.addTab("")
        while self.tabs.count() > len(self.docs):
            self.tabs.removeTab(self.tabs.count() - 1)
        self.update_tab_titles()
        if self.cur in self.docs:
            self.tabs.setCurrentIndex(self.docs.index(self.cur))
        self.tabs.blockSignals(False)
        self.tabs.update()

    def stash(self):
        d = self.cur
        d.sel, d.play_origin, d.hum_backup = self.sel, self.play_origin, self.hum_backup
        d.cached = d.dirty()

    def activate(self, doc):
        """Make a project the visible one.  Only the active project plays."""
        if doc is self.cur:
            self.refresh_tabs()
            return
        if self.cur in self.docs:
            self.stash()
            self.engine.pause()
            with self.engine.lock:
                self.engine.voices = []
        self.cur = doc
        self.project, self.engine = doc.project, doc.engine
        self.sel, self.play_origin, self.hum_backup = doc.sel, doc.play_origin, doc.hum_backup
        self.keys_down.clear()
        self.audio.eng = self.engine
        self.roll.ch = None
        self.refresh_all()
        self.refresh_tabs()
        self.sync_play_btn()

    def add_doc(self, doc):
        self.docs.append(doc)
        self.activate(doc)

    def new_tab(self, *_):
        d = Doc(Project(), name=self.untitled_name())
        d.mark_saved()
        self.add_doc(d)

    def step_tab(self, delta):
        if len(self.docs) > 1:
            self.activate(self.docs[(self.docs.index(self.cur) + delta) % len(self.docs)])

    def confirm_doc(self, d):
        """True when it is fine to discard d (saved, or the user chose not to save)."""
        if not d.dirty():
            return True
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("PyFL Studio")
        box.setText(f'Save changes to "{d.name}" before closing?')
        box.setInformativeText("Your changes will be lost if you don't save them.")
        b_save = box.addButton("Save", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        b_drop = box.addButton("Don't save", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(b_save)
        box.exec()
        c = box.clickedButton()
        if c is b_save:
            return self.save_doc(d)
        return c is b_drop

    def close_doc(self, d):
        if not self.confirm_doc(d):
            return False
        if len(self.docs) == 1:  # there is always one tab: closing the last one leaves a fresh empty project
            fresh = Doc(Project(), name="Untitled")
            fresh.mark_saved()
            self.docs.append(fresh)
        idx, was_cur = self.docs.index(d), d is self.cur
        d.engine.pause()
        self.docs.remove(d)
        if was_cur:
            self.cur = None
            self.activate(self.docs[min(idx, len(self.docs) - 1)])
        else:
            self.refresh_tabs()
        return True

    def close_current_tab(self, *_):
        self.close_doc(self.cur)

    def on_tab_changed(self, i):
        if 0 <= i < len(self.docs):
            self.activate(self.docs[i])

    def on_tab_moved(self, a, b):
        self.docs[a], self.docs[b] = self.docs[b], self.docs[a]
        self.update_tab_titles()

    def on_close_tab(self, i):
        if 0 <= i < len(self.docs):
            self.close_doc(self.docs[i])

    def on_rename_tab(self, i):
        if not 0 <= i < len(self.docs):
            return
        d = self.docs[i]
        t, ok = QtWidgets.QInputDialog.getText(self, "Rename project", "Project name:", text=d.name)
        if ok and t.strip():
            d.name = t.strip()[:40]
            self.refresh_tabs()

    # ---- state / refresh
    def refresh_all(self):
        keep = self.sel if self.sel in self.project.channels else None
        self.rack.rebuild()
        self.mixer.rebuild()
        self.bpm.blockSignals(True)
        self.bpm.setValue(self.project.bpm)
        self.bpm.blockSignals(False)
        self.len_box.blockSignals(True)
        if str(self.project.length) not in [self.len_box.itemText(i) for i in range(self.len_box.count())]:
            self.len_box.addItem(str(self.project.length))
        self.len_box.setCurrentText(str(self.project.length))
        self.len_box.blockSignals(False)
        self.master_knob.setValue(self.project.master)
        for r in self.rack.rows:
            r.steps.fit()
        self.roll.fit()
        self.roll_ruler.fit()
        self.select_channel(keep or (self.project.channels[0] if self.project.channels else None))
        for r in self.rack.rows:
            r.btn.setChecked(r.ch is self.sel)
        self.settings.show_channel(self.sel)
        self.update_roll_bar()
        self.rack.inner.adjustSize()
        self.refresh_playhead()

    def select_channel(self, ch):
        if ch is self.sel and self.roll.ch is ch:
            return
        self.sel = ch
        for r in self.rack.rows:
            r.btn.setChecked(r.ch is ch)
        self.roll.set_channel(ch)
        self.settings.show_channel(ch)
        self.update_roll_bar()
        if ch is not None and ch.kind == "audio":
            QTimer.singleShot(0, self.scroll_notes if ch.notes_mode else self.center_roll)

    # ---- audio-clip effects
    def fx_changed(self, ch, offline=True):
        """A setting changed.  Auto-Tune/Harmonizer renders restart by themselves a moment later (ensure_fx is polled
        in tick()), so dragging a knob doesn't start many renders; the Vocoder is live and needs nothing."""
        if offline and fx_needs_processing(ch):
            ch.fx_status = "Waiting for changes to settle..."

    def detect_key(self, ch):
        detect_key_async(ch)

    def update_fx_status(self):
        ch = self.sel
        if ch is None or ch.kind != "audio":
            return
        if ch.fx_refresh_ui:
            ch.fx_refresh_ui = False
            self.settings.show_channel(ch)
            return
        lab = getattr(self.settings, "fx_label", None)
        if lab is not None and lab.text() != ch.fx_status:
            lab.setText(ch.fx_status)

    def set_hint(self, text):
        if hasattr(self, "roll_hint"):
            self.roll_hint.setText(text)

    def flash(self, msg):
        self.statusBar().showMessage(msg, 5000)

    def ensure_length(self, end_step):
        """Grow the song if something now extends past its end."""
        if end_step > self.project.length and self.project.length < MAX_LEN:
            self.project.length = fit_length(end_step)
            QTimer.singleShot(0, self.refresh_all)

    def center_roll(self):
        vb = self.roll_scroll.verticalScrollBar()
        vb.setValue(int(self.roll.height() / 2 - self.roll_scroll.viewport().height() / 2))

    def update_roll_bar(self):
        ch = self.sel
        audio = ch is not None and ch.kind == "audio"
        nm = bool(audio and ch.notes_mode)
        synth = ch is not None and ch.kind == "synth"
        like = synth or nm
        title = "Piano roll" + (f"  -  {ch.name}" if ch else "")
        self.roll_title.setText(title + ("  (notes for effects)" if nm else ""))
        self.slur_btn.setEnabled(like)
        self.hum_btn.setEnabled(synth)
        self.notes_btn.setVisible(audio)
        self.notes_btn.blockSignals(True)
        self.notes_btn.setChecked(nm)
        self.notes_btn.blockSignals(False)
        choir = like and self.roll.fam() == "choir"
        self.vowel_lbl.setVisible(choir)
        self.vowel_box.setVisible(choir)

    def set_notes_mode(self, on):
        """Audio tracks: switch the piano roll between the clips and the notes that play the effects."""
        ch = self.sel
        if ch is None or ch.kind != "audio":
            return
        ch.notes_mode = bool(on)
        self.roll.sel.clear()
        self.roll.slur_from = self.roll.group = None
        self.update_roll_bar()
        self.roll.update()
        QTimer.singleShot(0, self.scroll_notes if on else self.center_roll)

    def scroll_notes(self):
        self.roll_scroll.verticalScrollBar().setValue((P_HI - 76) * PianoRoll.RH)

    def play_target(self):
        """Channel that sounds for the typing/MIDI keyboard: the track itself, or the carrier synth while editing
        an audio track's notes for effects; None for an audio track's clips."""
        ch = self.sel
        if ch is None:
            return None
        if ch.kind == "audio":
            return self.engine.carrier(ch, True) if ch.notes_mode else None
        return ch

    def set_instrument(self, ch, name):
        apply_preset(ch, name)
        self.settings.show_channel(ch)
        self.mixer.rebuild()
        self.update_roll_bar()
        self.roll.update()
        for r in self.rack.rows:
            if r.ch is ch:
                r.btn.setToolTip(name)

    def sync(self, ch, from_mixer=False):
        for r in self.rack.rows:
            if r.ch is ch:
                r.pank.setValue(ch.pan)
                r.volk.setValue(ch.vol)
                r.led.blockSignals(True)
                r.led.setChecked(not ch.mute)
                r.led.blockSignals(False)
        if not from_mixer:
            self.mixer.sync(ch)

    def set_len(self, text):
        self.project.length = int(text)
        for r in self.rack.rows:
            r.steps.fit()
            r.steps.update()
        self.rack.ruler.fit()
        self.roll_ruler.fit()
        self.roll.fit()
        self.roll.update()
        self.engine.seek(min(self.engine.pos, self.engine.loop_len() - 1))

    def set_bpm(self, v):
        old = self.project.bpm
        with self.engine.lock:  # keep the play bar on the same step when the tempo changes
            self.engine.pos = int(self.engine.pos * old / v)
            self.project.bpm = v
        for ch in self.project.channels:
            if ch.kind == "audio":
                for n in ch.notes:
                    n[1] = ch.clip_steps_of(n, v)
        self.rack.refresh_steps()
        self.roll.update()

    def set_master(self, v):
        self.project.master = v
        if hasattr(self.mixer, "master_slider"):
            self.mixer.master_slider.blockSignals(True)
            self.mixer.master_slider.setValue(int(v * 100))
            self.mixer.master_slider.blockSignals(False)

    # ---- transport / play bar
    def sync_play_btn(self):
        playing = self.engine.playing
        self.play_btn.setChecked(playing)
        self.play_btn.setText("\u23F8" if playing else "\u25B6")

    def toggle_play(self):
        e = self.engine
        if e.playing:
            e.pause()
        else:
            self.play_origin = e.pos
            e.play()
        self.sync_play_btn()

    def restart(self):
        self.engine.seek(0)
        self.play_origin = 0
        if not self.engine.playing:
            self.engine.play()
        self.sync_play_btn()
        self.refresh_playhead()

    def stop(self):
        e = self.engine
        e.silence()  # also ends held keyboard notes and any browser preview
        if e.playing:
            e.pause()
            e.seek(self.play_origin)
        else:
            e.seek(0)
            self.play_origin = 0
        self.sync_play_btn()
        self.refresh_playhead()

    def seek_step(self, step):
        step = max(0.0, min(float(step), self.project.length - 1e-3))
        self.engine.seek(int(step * self.engine.step_samples()))
        if not self.engine.playing:
            self.play_origin = self.engine.pos
        self.refresh_playhead()

    def refresh_playhead(self):
        self.show_playhead(self.engine.pos / self.engine.step_samples(), self.engine.playing)

    def show_playhead(self, stepf, playing):
        for r in self.rack.rows:
            r.steps.set_play(int(stepf) if playing else -1)
        self.roll.set_play(stepf)
        self.roll_ruler.set_play(stepf)
        if self.rack.ruler is not None:
            self.rack.ruler.set_play(stepf)

    def tick(self):
        self._ticks += 1
        if self._ticks % 30 == 0:
            self.update_tab_titles()
        if self._ticks % 15 == 0:
            ensure_fx(self.project)
        if self._ticks % 6 == 0:
            self.update_fx_status()
        e = self.engine
        stepf = e.pos / e.step_samples()
        self.show_playhead(stepf, e.playing)
        s = int(max(stepf, 0))
        self.time.setText(f"{s // 16 + 1:03d}:{(s % 16) // 4 + 1}:{s % 4 + 1}")
        self.meter.setValue(int(min(1.0, e.peak) * 100))
        self.sync_play_btn()
        if e.playing and self.follow.isChecked():
            vb = self.roll_scroll.verticalScrollBar()
            self.roll_scroll.ensureVisible(int(self.roll.px), vb.value() + self.roll_scroll.viewport().height() // 2, 140, 0)
            rb = self.rack.scroll.verticalScrollBar()
            self.rack.scroll.ensureVisible(int(210 + stepf * StepRow.CW), rb.value() + self.rack.scroll.viewport().height() // 2, 140, 0)

    # ---- file operations
    def err(self, msg):
        QtWidgets.QMessageBox.warning(self, "PyFL Studio", str(msg))

    def handle_files(self, paths):
        for p in paths:
            low = p.lower()
            try:
                if low.endswith(AUDIO_EXT):
                    self.add_audio_path(p, "auto")
                elif low.endswith(MIDI_EXT):
                    self.do_import_midi(p)
                elif low.endswith(".pyfl"):
                    self.do_open(p)
            except Exception as ex:
                self.err(f"Could not load {os.path.basename(p)}:\n{ex}")
        self.refresh_all()

    def add_audio_path(self, p, mode="auto"):
        """mode: 'clip' = audio-clip track at the play bar, 'sample' = pitched one-shot sampler,
        'auto' = clips for files of 4 seconds or more, samples for shorter ones."""
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            data = load_sample(p)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        as_clip = mode == "clip" or (mode == "auto" and data.shape[1] / SR >= 4.0)
        if not as_clip:
            self.select_channel(self.project.add_sample(p, data))
            return
        start = float(round(self.engine.pos / self.engine.step_samples()))
        ch = self.project.add_audio(p, data, start)
        end = start + ch.notes[0][1]
        if end > self.project.length:
            self.project.length = fit_length(end)
            if end > MAX_LEN:
                self.err(f"The clip is longer than the maximum song length ({MAX_LEN} steps); the end will be cut off.\n"
                         "Lower the BPM or trim the file to fit the whole clip.")
        self.select_channel(ch)

    def add_instrument(self, name):
        self.select_channel(self.project.add_instrument(name))
        self.refresh_all()

    def add_synth(self):
        self.add_instrument("Synth")

    def _audio_dialog(self, title, mode):
        ps, _ = QtWidgets.QFileDialog.getOpenFileNames(self, title, "", "Audio (*.wav *.mp3 *.flac *.ogg *.aif *.aiff)")
        for p in ps:
            try:
                self.add_audio_path(p, mode)
            except Exception as ex:
                self.err(f"Could not load {os.path.basename(p)}:\n{ex}")
        if ps:
            self.refresh_all()

    def add_audio_dialog(self):
        self._audio_dialog("Add audio clip", "clip")

    def add_sample_dialog(self):
        self._audio_dialog("Add sample", "sample")

    def replace_sample(self, ch):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load audio", "", "Audio (*.wav *.mp3 *.flac *.ogg *.aif *.aiff)")
        if p:
            try:
                ch.set_audio(load_sample(p), p)
                if ch.kind == "audio":
                    for n in ch.notes:  # a new file: show the whole file again
                        n[5], n[7] = 0.0, 0.0
                        n[1] = ch.clip_steps(self.project.bpm)
                self.settings.show_channel(ch)
                self.rack.refresh_steps()
            except Exception as ex:
                self.err(ex)

    def do_import_midi(self, p):
        if mido is None:
            return self.err("Install mido:  pip install mido python-rtmidi")
        n = import_midi(p, self.project)
        if not n:
            self.err("No melodic notes found in that MIDI file.")

    def import_midi_dialog(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import MIDI", "", "MIDI (*.mid *.midi)")
        if p:
            self.handle_files([p])

    def export_midi_dialog(self):
        if mido is None:
            return self.err("Install mido:  pip install mido")
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export MIDI", "song.mid", "MIDI (*.mid)")
        if p:
            export_midi(p, self.project)

    def export_wav(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export WAV", "song.wav", "WAV (*.wav)")
        if not p:
            return
        self.engine.pause()
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            fx_wait(self.project, tick=QtWidgets.QApplication.processEvents)  # let Auto-Tune/Harmonizer finish
            audio = Engine(self.project).render_offline()
            wavfile.write(p, SR, (np.clip(audio.T, -1, 1) * 32767).astype(np.int16))
        except Exception as ex:
            self.err(ex)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def save_doc(self, doc, as_new=False):
        path = doc.file
        if as_new or not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", doc.file or (doc.name + ".pyfl"),
                                                            "PyFL project (*.pyfl)")
            if not path:
                return False
            if not path.lower().endswith(".pyfl"):
                path += ".pyfl"
        try:
            with open(path, "w") as fh:
                json.dump(doc.project.to_json(), fh)
        except Exception as ex:
            self.err(f"Could not save the project:\n{ex}")
            return False
        doc.file, doc.name = path, os.path.splitext(os.path.basename(path))[0]
        doc.mark_saved()
        self.refresh_tabs()
        self.flash(f"Saved {path}")
        return True

    def save_project(self):
        self.save_doc(self.cur)

    def save_project_as(self):
        self.save_doc(self.cur, True)

    def do_open(self, p):
        """Open a .pyfl project in its own tab (switches to it if it is already open)."""
        ab = os.path.abspath(p)
        for d in self.docs:
            if d.file and os.path.abspath(d.file) == ab:
                self.activate(d)
                return
        with open(p) as fh:
            data = json.load(fh)
        proj = Project()
        proj.load_json(data)
        d = Doc(proj, file=p)
        d.mark_saved()
        blank = self.cur if (self.cur.file is None and not self.cur.project.channels and not self.cur.dirty()) else None
        self.add_doc(d)
        if blank is not None and blank in self.docs:  # an untouched empty tab is replaced instead of kept
            self.docs.remove(blank)
            self.refresh_tabs()

    def open_project(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", "", "PyFL project (*.pyfl)")
        if p:
            self.handle_files([p])

    # ---- hum notes
    def open_hum(self):
        ch = self.sel
        if ch is None or ch.kind != "synth":
            return self.err("First select an instrument or synth track (click its name in the channel rack),\n"
                            "then press Hum notes.")
        if sd is None:
            return self.err("The sounddevice package is missing (needed for the microphone).\nRun ./install.sh")
        self.engine.pause()
        self.sync_play_btn()
        HumDialog(self, ch).exec()
        self.sync_play_btn()

    def undo_hum(self):
        if not self.hum_backup:
            return False
        ch, notes = self.hum_backup
        self.hum_backup = None
        if ch in self.project.channels:
            ch.notes[:] = notes
            self.rack.refresh_steps()
            self.roll.update()
            return True
        return False

    def show_notes(self, notes):
        """Scroll the piano roll so freshly written notes are visible."""
        if not notes:
            return
        pitches = sorted(n[2] for n in notes)
        mid = pitches[len(pitches) // 2]
        vb, hb = self.roll_scroll.verticalScrollBar(), self.roll_scroll.horizontalScrollBar()
        vb.setValue(max(0, int((P_HI - mid) * PianoRoll.RH - self.roll_scroll.viewport().height() / 2)))
        hb.setValue(max(0, int(PianoRoll.KW + min(n[0] for n in notes) * PianoRoll.CW - 120)))

    # ---- MIDI keyboard + typing keyboard
    def fill_midi_menu(self):
        self.midi_menu.clear()
        if mido is None:
            self.midi_menu.addAction("mido not installed").setEnabled(False)
            return
        try:
            names = mido.get_input_names()
        except Exception:
            names = []
        if not names:
            self.midi_menu.addAction("No MIDI inputs found").setEnabled(False)
        for n in names:
            a = self.midi_menu.addAction(n)
            a.setCheckable(True)
            a.setChecked(self.midi_in is not None and self.midi_in.name == n)
            a.triggered.connect(lambda _=False, n=n: self.open_midi(n))

    def open_midi(self, name):
        try:
            if self.midi_in:
                self.midi_in.close()
            self.midi_in = mido.open_input(name, callback=self.on_midi)
        except Exception as ex:
            self.err(f"Could not open MIDI input:\n{ex}")

    def on_midi(self, msg):
        ch = self.play_target()
        if ch is None:
            return
        if msg.type == "note_on" and msg.velocity > 0:
            self.engine.note_on(ch, msg.note, msg.velocity / 127.0)
        elif msg.type in ("note_off", "note_on"):
            self.engine.note_off(ch, msg.note)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.roll.cancel()
            return
        k = e.text().lower()
        if k in KEYMAP and not e.isAutoRepeat() and self.sel and not (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            tgt = self.play_target()
            if tgt is not None:
                pitch = 60 + KEYMAP[k]
                if k not in self.keys_down:
                    self.keys_down[k] = (tgt, pitch)
                    self.engine.note_on(tgt, pitch, 0.9)
                return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        k = e.text().lower()
        if k in self.keys_down and not e.isAutoRepeat():
            ch, pitch = self.keys_down.pop(k)
            self.engine.note_off(ch, pitch)
            return
        super().keyReleaseEvent(e)

    def closeEvent(self, e):
        for d in list(self.docs):
            if d.dirty():
                self.activate(d)
                if not self.confirm_doc(d):
                    e.ignore()
                    return
        self.timer.stop()
        if self.midi_in:
            self.midi_in.close()
        self.audio.close()
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
