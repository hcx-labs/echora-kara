"""Musical augmentation for training windows (train split only).

The rule: an augmentation never moves a timing label. Pitch, formant, EQ, compression, echo,
noise and codec changes keep the timeline; tempo changes scale it, and every token time is scaled
with the audio (``scale_tokens``). Most effects run as one FFmpeg filter chain inside the decode
that already happens per window, so augmentation adds almost no CPU work.
"""
from __future__ import annotations

import functools
import math
import random
import subprocess
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16_000
TIME_KEYS = ("start", "end", "region_start", "region_end")


@dataclass(frozen=True)
class AugmentConfig:
    pitch_probability: float = 0.4
    pitch_semitones: float = 3.0
    formant_probability: float = 0.2  # of pitch-shifted windows: let formants move with the pitch
    tempo_probability: float = 0.3
    tempo_range: tuple[float, float] = (0.9, 1.1)
    compression_probability: float = 0.3
    eq_probability: float = 0.4
    telephone_probability: float = 0.1
    echo_probability: float = 0.2
    noise_probability: float = 0.2
    noise_snr_db: tuple[float, float] = (20.0, 35.0)
    codec_probability: float = 0.15


@dataclass
class Plan:
    """The effects chosen for one window."""
    filters: list[str] = field(default_factory=list)
    tempo: float = 1.0
    noise_snr_db: float | None = None
    codec: bool = False
    names: list[str] = field(default_factory=list)


def plan(config: AugmentConfig, rng: random.Random) -> Plan:
    chosen = Plan()
    pitch, tempo = 1.0, 1.0
    if rng.random() < config.pitch_probability:
        semitones = rng.uniform(-config.pitch_semitones, config.pitch_semitones)
        pitch = 2 ** (semitones / 12)
        chosen.names.append(f"pitch{semitones:+.1f}")
    if rng.random() < config.tempo_probability:
        tempo = rng.uniform(*config.tempo_range)
        chosen.names.append(f"tempo{tempo:.2f}")
    if pitch != 1.0 or tempo != 1.0:
        formant = "shifted" if pitch != 1.0 and rng.random() < config.formant_probability else "preserved"
        chosen.filters.append(f"rubberband=pitch={pitch:.5f}:tempo={tempo:.5f}:formant={formant}:pitchq=quality")
        chosen.tempo = tempo
    if rng.random() < config.eq_probability:
        for _ in range(rng.randint(1, 3)):
            chosen.filters.append(
                f"equalizer=f={rng.choice((120, 250, 500, 1000, 2500, 4000, 7000))}:t=q:w={rng.uniform(0.7, 2.0):.2f}"
                f":g={rng.uniform(-8, 8):.1f}")
        chosen.names.append("eq")
    if rng.random() < config.telephone_probability:
        chosen.filters.append(f"highpass=f={rng.uniform(250, 450):.0f},lowpass=f={rng.uniform(3000, 4000):.0f}")
        chosen.names.append("telephone")
    if rng.random() < config.compression_probability:
        chosen.filters.append(f"acompressor=threshold={rng.uniform(0.05, 0.3):.3f}:ratio={rng.uniform(2, 6):.1f}"
                              f":attack={rng.uniform(5, 30):.0f}:release={rng.uniform(50, 250):.0f}")
        chosen.names.append("compress")
    if rng.random() < config.echo_probability:
        # Light, short reflections: long reverb tails would spill sound past word ends.
        delay = rng.uniform(15, 60)
        chosen.filters.append(f"aecho=0.8:{rng.uniform(0.6, 0.85):.2f}:{delay:.0f}|{delay * 1.7:.0f}"
                              f":{rng.uniform(0.1, 0.3):.2f}|{rng.uniform(0.05, 0.2):.2f}")
        chosen.names.append("echo")
    if rng.random() < config.noise_probability:
        chosen.noise_snr_db = rng.uniform(*config.noise_snr_db)
        chosen.names.append(f"noise{chosen.noise_snr_db:.0f}dB")
    if rng.random() < config.codec_probability:
        chosen.codec = True
        chosen.names.append("codec")
    return chosen


def scale_tokens(tokens: list[dict], tempo: float) -> list[dict]:
    """Token times after a tempo change: audio at ``tempo``× speed lasts 1/tempo as long."""
    if tempo == 1.0:
        return tokens
    return [{**t, **{k: t[k] / tempo for k in TIME_KEYS if k in t}} for t in tokens]


def add_noise(waveform: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    """Pink-ish noise (crowd and room tone are mostly low-frequency) at the given SNR."""
    generator = np.random.default_rng(rng.getrandbits(32))
    white = generator.standard_normal(waveform.size).astype(np.float32)
    spectrum = np.fft.rfft(white)
    spectrum /= np.sqrt(np.maximum(np.arange(spectrum.size), 1))
    noise = np.fft.irfft(spectrum, n=waveform.size).astype(np.float32)
    signal_power = float(np.mean(waveform ** 2)) or 1e-8
    noise *= math.sqrt(signal_power / (10 ** (snr_db / 10)) / (float(np.mean(noise ** 2)) or 1e-8))
    return waveform + noise


CODECS = (("libmp3lame", "48k"), ("libmp3lame", "64k"), ("aac", "48k"), ("aac", "64k"))


def _round_trip(waveform: np.ndarray, codec: str, bitrate: str) -> np.ndarray:
    muxer, demuxer = ("mp3", "mp3") if codec == "libmp3lame" else ("adts", "aac")
    encoded = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
         "-c:a", codec, "-b:a", bitrate, "-f", muxer, "pipe:1"],
        input=waveform.astype(np.float32).tobytes(), stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", demuxer, "-i", "pipe:0", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        input=encoded, stdout=subprocess.PIPE, check=True).stdout, dtype=np.float32)


@functools.lru_cache(maxsize=None)
def codec_delay(codec: str, bitrate: str) -> int:
    """The samples a codec round trip adds at the start, measured once with a noise click.

    The delay is fixed per codec and bitrate. Estimating it from each window's own audio is
    unreliable: sustained tones correlate equally well at several lags.
    """
    probe = np.zeros(SAMPLE_RATE * 2, np.float32)
    click = np.random.default_rng(0).standard_normal(800).astype(np.float32) * 0.5
    probe[SAMPLE_RATE // 2:SAMPLE_RATE // 2 + click.size] = click
    decoded = _round_trip(probe, codec, bitrate)
    scores = [float(np.dot(click, decoded[SAMPLE_RATE // 2 + lag:SAMPLE_RATE // 2 + lag + click.size]))
              for lag in range(0, 4096)]
    return int(np.argmax(scores))


def codec_round_trip(waveform: np.ndarray, rng: random.Random) -> np.ndarray:
    """Re-encode at a low bitrate and decode, then remove the codec's delay so labels stay put."""
    codec, bitrate = rng.choice(CODECS)
    decoded = _round_trip(waveform, codec, bitrate)
    delay = codec_delay(codec, bitrate)
    aligned = decoded[delay:delay + waveform.size]
    if aligned.size < waveform.size:
        aligned = np.pad(aligned, (0, waveform.size - aligned.size))
    return aligned.astype(np.float32)


def finish(waveform: np.ndarray, chosen: Plan, rng: random.Random) -> np.ndarray:
    """The effects that run on decoded samples rather than in the FFmpeg chain."""
    if chosen.codec:
        waveform = codec_round_trip(waveform, rng)
    if chosen.noise_snr_db is not None:
        waveform = add_noise(waveform, chosen.noise_snr_db, rng)
    return waveform
