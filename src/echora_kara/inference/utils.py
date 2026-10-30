"""Timestamp conversions ("[mm:ss:cc]" LRC strings, ASS times, integer centiseconds) and post-processing of the
aligned items: long-line splitting, line-head adjustment to non-silent ranges and the legacy line-tail correction."""

import numpy as np
import re
import unicodedata


def parse_time_to_hundredths(time_str: str) -> int:
    ".lrc timestamp to integer centiseconds"
    match = re.match(r"\[(\d{2}):(\d{2}):(\d{2})\]", time_str)
    minutes, seconds, hundredths = int(match.group(1)), int(match.group(2)), int(match.group(3))
    return minutes * 6000 + seconds * 100 + hundredths


def format_hundredths_to_time_str(total_hundredths: int) -> str:
    "Integer centiseconds to an .lrc timestamp"
    total_hundredths = round(total_hundredths)
    minutes = total_hundredths // 6000
    remaining = total_hundredths % 6000
    seconds = remaining // 100
    cs = remaining % 100
    return f"[{minutes:02d}:{seconds:02d}:{cs:02d}]"


def int2asstime(cs: int) -> str:
    "Integer centiseconds to an .ass timestamp"
    cs = round(cs)
    hours = cs // 360000
    cs %= 360000
    minutes = cs // 6000
    cs %= 6000
    seconds = cs // 100
    cs %= 100
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def calculate_length(surface):
    "String length in full-width units (half-width characters count 0.5)"
    length = 0.0
    for char in surface:
        if unicodedata.east_asian_width(char) in ("F", "W", "A"):
            length += 1
        else:
            length += 0.5
    return length


def split_long_segments(elements, max_length=20):
    """
    Keep the text between two newlines at most max_length long: when it is longer, the best
    space is replaced by a newline. Modifies the list in place!
    """
    current_length = 0.0
    space_positions = []  # space positions and the length before each
    i = 0
    while i <= len(elements):
        if i == len(elements) or elements[i].get("type") == 0 and elements[i].get("orig") == "\n":
            if current_length > max_length and space_positions:
                # the space that splits the line most evenly
                n_cuts = current_length // max_length + 1
                n_cut_length = current_length / n_cuts
                sorted_spaces = sorted(
                    space_positions,
                    key=lambda x: (
                        0 if x[1] <= max_length else 1,
                        abs(x[1] - n_cut_length) if x[1] <= max_length else -x[1],
                    ),
                )
                best_position = sorted_spaces[0][0]
                elements[best_position]["orig"] = "\n"

                i = best_position  # continue after the split point
                current_length = 0.0
                space_positions = []
            else:
                current_length = 0.0
                space_positions = []

        elif i < len(elements):
            elem = elements[i]
            surface = elem.get("orig")
            elem_length = calculate_length(surface)
            if surface in (" ", "　") and elem.get("type") == 0:
                space_positions.append((i, current_length))
            current_length += elem_length
        i += 1


def non_silent_head_adjust(result_list, non_silent_ranges):
    """
    Adjust start times so that each phrase lies entirely inside one non-silent range.

    Walks `result_list`, grouping consecutive non-silent items into phrases by their `type`.
    A phrase whose time span no non-silent range fully covers has its start moved to the
    start of the non-silent range that covers its end.

    Args:
        result_list (List[Dict[str, Any]]): aligned items, each a dict with at least:
            - 'type' (int): 0 for a silent separator, non-zero for sung content.
            - 'start' (str): start time, "[MM:SS:CC]".
            - 'end' (str): end time, same format.

        non_silent_ranges (List[Tuple[float, float]]): non-silent ranges
            `(start seconds, end seconds)`, sorted by start.

    Returns:
        List[Dict[str, Any]]: `result_list`, modified in place; only some 'start' fields change.
    """
    if not non_silent_ranges:
        return result_list
    else:
        # split into phrases
        i = si = 0
        sentences_list = []
        st = None
        while i < len(result_list):
            if result_list[i].get("type") == 0:
                if st:
                    sentences_list.append((si, i - 1, st, result_list[i - 1].get("end")))
                    st = None
            elif not st:
                si = i
                st = result_list[i].get("start")
            i += 1
        # adjust the times
        for inds, inde, sst, sen in sentences_list:
            sst = parse_time_to_hundredths(sst)
            sen = parse_time_to_hundredths(sen)
            interval_covered = False
            for ns_start, ns_end in non_silent_ranges:
                if int(ns_start * 100) > sst:
                    break
                # does a non-silent segment cover the whole range
                if int(ns_start * 100) <= sst and int(np.ceil(ns_end * 100)) >= sen:
                    interval_covered = True
                    break
            if not interval_covered:
                end_covered = False
                for i in range(len(non_silent_ranges)):
                    ns_start = int(non_silent_ranges[i][0] * 100)
                    ns_end = int(np.ceil(non_silent_ranges[i][1] * 100))
                    if ns_start > sen:
                        break
                    if ns_start <= sen:
                        if ns_end >= sen:
                            end_covered = True
                            adjust_target = ns_start
                            break
                    elif ns_start <= parse_time_to_hundredths(result_list[inde].get("start")) <= ns_end:
                        end_covered = True
                        adjust_target = ns_start
                        break
                if not end_covered:
                    print("Errors ignored while trying to correct end sounds...")
                    break
                else:
                    adjust_target = min(parse_time_to_hundredths(result_list[inds]["end"]), adjust_target)
                    result_list[inds]["start"] = format_hundredths_to_time_str(adjust_target)
        return result_list


def func_tail_correct_v250611(result_list, tail_mode=1):
    "Line-tail adjustment pre-processing (older version)"
    if not result_list:
        return []
    if tail_mode == 1:
        for i in range(len(result_list)):
            if result_list[i]["type"] == 0:
                try:
                    if result_list[i - 1].get("pron") and result_list[i - 1]["type"] != 0:
                        pre_vowel = result_list[i - 1]["pron"][-1]
                        post_consonant = ""
                        if i < len(result_list) - 1:
                            post_i = i + 1
                            while post_i < len(result_list):
                                if "pron" in result_list[post_i] and len(result_list[post_i]["pron"]) >= 1:
                                    post_consonant = result_list[post_i]["pron"][0]
                                    break
                                else:
                                    post_i += 1
                        if pre_vowel != post_consonant and post_consonant not in ("a", "e", "i", "o", "u"):
                            result_list[i]["pron"] = pre_vowel + "h"
                except:
                    continue
    elif tail_mode == 2:
        for i in range(len(result_list)):
            if result_list[i]["type"] == 0:
                try:  # exploit the baseline model's tail behaviour
                    if len(result_list[i - 1]["pron"]) >= 1 and result_list[i - 1]["type"] != 0:
                        result_list[i]["pron"] = result_list[i - 1]["pron"][-1] + "h"
                except:
                    continue
    return result_list
