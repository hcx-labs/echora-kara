import bisect
import librosa
import numpy as np
import time

from echora_kara.inference.utils import parse_time_to_hundredths, format_hundredths_to_time_str

def non_silent_recog(audio_file, sr = None, frame_second = 1, threspct = 10, thresrto = .1):
    'Detect non-silent segments'
    frame_length = int(sr * frame_second)
    hop_length = frame_length // 2  # 50% overlap
    energy = librosa.feature.rms(y=audio_file, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = np.percentile(energy, 100-threspct) * thresrto
    non_silent_frames = energy > threshold
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length) # to times
    segments = [] # merge consecutive segments
    start = None
    for i, (t, active) in enumerate(zip(times, non_silent_frames)):
        if active and start is None:
            start = max(t-frame_second/4, 0)
        elif not active and start is not None:
            segments.append((start, t+frame_second/4))
            start = None
    if start is not None:
        segments.append((start, times[-1]))
    return segments

def time_stretch_audio(audio: np.ndarray, speed: float, verbose: bool = True) -> np.ndarray:
    """
    Time-stretch (speed-change) the audio.

    Args:
        audio (np.ndarray): original audio signal (1-D).
        speed (float): speed factor; > 1 speeds up (shorter), < 1 slows down (longer), 1 unchanged.
        verbose (bool): print run time and status, default True.

    Returns:
        np.ndarray: the stretched audio; the original audio when speed == 1.
    """
    if speed == 1:
        return audio

    if verbose:
        print('Changing the audio speed...')
        start_time = time.time()

    y_processed = librosa.effects.time_stretch(audio, rate=speed)

    if verbose:
        end_time = time.time()
        print(f"Audio speed changing executed in {round(end_time - start_time, 3)} seconds")

    return y_processed

def func_tail_correct_v250615(result_list, audio_file, sr, tail_thres_pct, tail_thres_ratio, frame_second=.02):
    'Line-tail adjustment post-processing (current version)'
    if not result_list:
        return []
    ns_small = non_silent_recog(audio_file, sr, frame_second, tail_thres_pct, tail_thres_ratio)
    ns_ends = [int(np.ceil(ns_end * 100)) for _, ns_end in ns_small]
    for i in range(len(result_list)-1):
        if result_list[i]['type'] != 0 and result_list[i+1]['type'] == 0:
            current_end = parse_time_to_hundredths(result_list[i]['end'])
            next_ind = i + 2
            next_start = np.inf
            while next_ind < len(result_list):
                if 'start' in result_list[next_ind]:
                    next_start = parse_time_to_hundredths(result_list[next_ind]['start'])
                    break
                next_ind += 1
            left_index = bisect.bisect_left(ns_ends, current_end)
            right_index = bisect.bisect_left(ns_ends, next_start)
            if left_index < right_index and left_index < len(ns_ends):
                result_list[i]['end'] = format_hundredths_to_time_str(ns_ends[left_index])
            else:
                interval_covered = False # does a non-silent segment cover the whole range
                for nss_start, nss_end in ns_small:
                    if int(nss_start * 100) > current_end:
                        break
                    if int(nss_start * 100) <= current_end and int(np.ceil(nss_end * 100)) >= next_start:
                        interval_covered = True
                        break
                if interval_covered:
                    result_list[i]['end'] = format_hundredths_to_time_str(max(next_start-2, current_end))
    return result_list