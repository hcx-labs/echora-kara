"""Converters from lyrics-site markup (Moegirlpedia wikitext, UtaTen HTML) to the hrh text format the CLI reads,
where a ruby word is ``{kanji|reading}``."""

import re

newnums = [
    "①",
    "②",
    "③",
    "④",
    "⑤",
    "⑥",
    "⑦",
    "⑧",
    "⑨",
    "⑩",
    "⑪",
    "⑫",
    "⑬",
    "⑭",
    "⑮",
    "⑯",
    "⑰",
    "⑱",
    "⑲",
    "⑳",
    "㉑",
    "㉒",
    "㉓",
    "㉔",
    "㉕",
    "㉖",
    "㉗",
    "㉘",
    "㉙",
    "㉚",
]


def moeg_process_line(line):
    """One line of Moegirlpedia wikitext lyrics in hrh format: drops ``|``-prefixed table rows and ``#NoHover``,
    turns ``{{Photrans|kanji|reading}}`` into ``{kanji|reading}``, unwraps ``{{lj|...}}`` and maps ``@1``..``@30`` to
    ①..㉚."""
    if not line or line[0] == "|":
        return ""
    result = line.replace("#NoHover", "")
    result = re.sub(r"<--.*?-->", "", result)
    result = re.sub(r"{{Photrans\|([^|]+)\|([^}]+)}}", r"{\1|\2}", result)
    result = re.sub(r"{{lj\|([^}]+)}}", r"\1", result)
    for i in range(len(newnums)):
        result = result.replace("@" + str(len(newnums) - i), newnums[-i - 1])
    return result


def utat_process(text):
    """Lyrics lines from an UtaTen page's HTML: keeps the ``<div class="hiragana">`` body, turns each ruby span into
    ``{base|reading}`` and splits at ``<br>``. Returns newline-terminated lines."""
    start_match = re.search(r'<div\s+class="hiragana"\s*[^>]*>', text)
    if start_match:
        text = text[start_match.end() :].strip()
    end_match = re.search(r"</div>", text)
    if end_match:
        text = text[: end_match.start()].strip()
    text = re.sub(
        r'<span class="ruby"><span class="rb">(.*?)</span><span class="rt">(.*?)</span></span>', r"{\1|\2}", text
    )
    result = re.split(r"<br\s*/?>\s*", text)
    result = [i + "\n" for i in result]
    return result
