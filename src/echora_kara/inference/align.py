"""Per-token forced alignment with torchaudio's MMS_FA pipeline (the CLI's ``--model mms`` path), plus a variant on
the yohane-derived ``engine.aligner.TorchAudioForcedAligner``; times come back as "[mm:ss:cc]" strings and seconds."""

import torch
import torchaudio
import math
import time


def align_audio_with_text(audio_file_path, text_tokens, non_silent_ranges=[], sr=None, speed=1, use_gpu=True):
    """Align ``text_tokens`` to the audio in one MMS_FA CTC pass.

    ``audio_file_path`` is a file path or a mono sample array at ``sr`` Hz. When ``non_silent_ranges`` ((start, end)
    seconds on the original timeline) are given, only those ranges are cut out, concatenated and aligned; ``speed``
    is the time-stretch factor already applied to the audio. Both are undone so times refer to the original audio.
    Empty tokens are dropped. Returns one dict per kept token: ``token``, ``start``/``end`` as "[mm:ss:cc]" strings
    (``'[error]'`` when the token got no span) and ``original_start``/``original_end`` in seconds. Any error is
    printed and yields []."""
    start_time = time.time()
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    try:
        bundle = torchaudio.pipelines.MMS_FA
        if isinstance(audio_file_path, str):
            waveform, sample_rate = torchaudio.load(audio_file_path)
        else:
            waveform = torch.tensor(audio_file_path).float()
            waveform = waveform.unsqueeze(0)
            sample_rate = sr

        # keep only the non-silent ranges
        if non_silent_ranges:
            # seconds to sample indexes
            total_samples = waveform.shape[1]
            sample_ranges = []
            for start_sec, end_sec in non_silent_ranges:
                start_sample = int(start_sec * sample_rate / speed)
                end_sample = min(int(end_sec * sample_rate / speed), total_samples)
                sample_ranges.append((start_sample, end_sample))

            # extract and concatenate the non-silent segments
            segments = []
            for start, end in sample_ranges:
                segments.append(waveform[:, start:end])
            waveform = torch.cat(segments, dim=1)

        # mono
        waveform = waveform.mean(0, keepdim=True)

        # resample
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

        # load the model
        model = bundle.get_model().to(device)
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()

        # non-empty tokens
        valid_tokens = [token for token in text_tokens if token]

        with torch.inference_mode():
            emission, _ = model(waveform.to(device))
            tokens = tokenizer(valid_tokens)
            token_spans = aligner(emission[0], tokens)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # time conversion parameters
        frame_duration = 1.0 / bundle.sample_rate * 320 * speed
        results = []

        # map back to the original time
        def map_to_original_time(adjusted_time):
            """Map a time in the processed audio back to the original audio."""
            if not non_silent_ranges:
                return adjusted_time

            cumulative_duration = 0.0
            for start_sec, end_sec in non_silent_ranges:
                segment_duration = end_sec - start_sec
                if adjusted_time < cumulative_duration + segment_duration:
                    return start_sec + (adjusted_time - cumulative_duration)
                cumulative_duration += segment_duration
            return non_silent_ranges[-1][1]  # past the end: the last time

        # time formatting
        def format_time(time_sec):
            minutes, remainder = divmod(time_sec, 60)
            seconds, centiseconds = divmod(remainder, 1)
            return f"[{int(minutes):02d}:{int(seconds):02d}:{math.floor(centiseconds * 100):02d}]"

        # time of every aligned token
        for i, spans in enumerate(token_spans):
            if not spans:
                results.append({"token": valid_tokens[i], "start": "[error]", "end": "[error]"})
                continue

            # adjusted time
            adjusted_start = spans[0].start * frame_duration
            adjusted_end = spans[-1].end * frame_duration

            # map back to the original audio time
            original_start = map_to_original_time(adjusted_start)
            original_end = map_to_original_time(adjusted_end)

            results.append(
                {
                    "token": valid_tokens[i],
                    "start": format_time(original_start),
                    "end": format_time(original_end),
                    "original_start": original_start,
                    "original_end": original_end,
                }
            )

        end_time = time.time()
        print("Alignment inference executed in", round(end_time - start_time, 3), "seconds")
        return results

    except Exception as e:
        print(f"Error during alignment: {e}")
        return []


def align_audio_with_text_torch_yohane(
    audio_file_path, text_tokens, non_silent_ranges=[], sr=None, speed=1, use_gpu=True
):
    "Reproduction using the TorchAudioForcedAligner class from yohane"

    from echora_kara.engine.aligner import TorchAudioForcedAligner

    start_time = time.time()

    try:
        if isinstance(audio_file_path, str):
            waveform, sample_rate = torchaudio.load(audio_file_path)
        else:
            waveform = torch.tensor(audio_file_path).float()
            waveform = waveform.unsqueeze(0)
            sample_rate = sr

        # keep only the non-silent ranges
        if non_silent_ranges:
            # seconds to sample indexes
            total_samples = waveform.shape[1]
            sample_ranges = []
            for start_sec, end_sec in non_silent_ranges:
                start_sample = int(start_sec * sample_rate / speed)
                end_sample = min(int(end_sec * sample_rate / speed), total_samples)
                sample_ranges.append((start_sample, end_sample))

            # extract and concatenate the non-silent segments
            segments = []
            for start, end in sample_ranges:
                segments.append(waveform[:, start:end])
            waveform = torch.cat(segments, dim=1)

        # non-empty tokens
        valid_tokens = [token for token in text_tokens if token]

        # from yohane-2026.5.0
        torch_aligner = TorchAudioForcedAligner()
        tokens = torch_aligner.tokenize(valid_tokens)
        _, token_spans, tgt_sample_rate = torch_aligner.align(tokens, waveform, sample_rate)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # time conversion parameters
        frame_duration = 1.0 / tgt_sample_rate * 320 * speed
        results = []

        # map back to the original time
        def map_to_original_time(adjusted_time):
            """Map a time in the processed audio back to the original audio."""
            if not non_silent_ranges:
                return adjusted_time

            cumulative_duration = 0.0
            for start_sec, end_sec in non_silent_ranges:
                segment_duration = end_sec - start_sec
                if adjusted_time < cumulative_duration + segment_duration:
                    return start_sec + (adjusted_time - cumulative_duration)
                cumulative_duration += segment_duration
            return non_silent_ranges[-1][1]  # past the end: the last time

        # time formatting
        def format_time(time_sec):
            minutes, remainder = divmod(time_sec, 60)
            seconds, centiseconds = divmod(remainder, 1)
            return f"[{int(minutes):02d}:{int(seconds):02d}:{math.floor(centiseconds * 100):02d}]"

        # time of every aligned token
        for i, spans in enumerate(token_spans):
            if not spans:
                results.append({"token": valid_tokens[i], "start": "[error]", "end": "[error]"})
                continue

            # adjusted time
            adjusted_start = spans[0].start * frame_duration
            adjusted_end = spans[-1].end * frame_duration

            # map back to the original audio time
            original_start = map_to_original_time(adjusted_start)
            original_end = map_to_original_time(adjusted_end)

            results.append(
                {
                    "token": valid_tokens[i],
                    "start": format_time(original_start),
                    "end": format_time(original_end),
                    "original_start": original_start,
                    "original_end": original_end,
                }
            )

        end_time = time.time()
        print("Alignment inference executed in", round(end_time - start_time, 3), "seconds")
        return results

    except Exception as e:
        print(f"Error during alignment: {e}")
        return []
