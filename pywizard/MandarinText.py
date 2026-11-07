"""Mandarin text to English spellings for the existing Flite tone input."""

import re


HAN_TEXT = re.compile(
    r"[\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]+")
HAN_RUN = re.compile(f"({HAN_TEXT.pattern})([，。！？；：、,.!?;:]*)")

INITIAL_SPELLINGS = {
    "": "", "b": "b", "p": "p", "m": "m", "f": "f",
    "d": "d", "t": "t", "n": "n", "l": "l", "g": "g", "k": "k",
    "h": "h", "j": "j", "q": "ch", "x": "sh", "zh": "j",
    "ch": "ch", "sh": "sh", "r": "r", "z": "dz", "c": "ts", "s": "s",
}

FINAL_SPELLINGS = {
    "a": "ah", "o": "aw", "e": "uh", "ê": "eh",
    "ai": "eye", "ei": "ay", "ao": "ow", "ou": "oh",
    "an": "ahn", "en": "un", "ang": "ahng", "eng": "ung", "ong": "oong",
    "i": "ee", "ia": "yah", "ie": "yeh", "iao": "yow", "iou": "yoh",
    "ian": "yen", "in": "een", "iang": "yahng", "ing": "ing", "iong": "yoong",
    "u": "oo", "ua": "wah", "uo": "aw", "uai": "why", "uei": "way",
    "uan": "wahn", "uen": "woon", "uang": "wahng", "ueng": "woong",
    "v": "yoo", "ve": "yeh", "van": "yen", "vn": "yoon", "er": "er",
}

# English spelling rules need whole-syllable choices for these pronunciations.
SYLLABLE_SPELLINGS = {
    "ni": "knee", "wo": "woo", "hao": "how", "jia": "jah", "ming": "mean",
    "shi": "she", "zhi": "jir", "chi": "chir", "ri": "rir",
    "zi": "zee", "ci": "tsee", "si": "see", "bai": "bye",
    "m": "mm", "n": "un", "ng": "ung", "hm": "hum", "hng": "hung",
}

PUNCTUATION = str.maketrans({
    "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
    "、": ",", "（": "(", "）": ")", "【": "[", "】": "]",
    "“": '"', "”": '"', "‘": "'", "’": "'", "《": '"', "》": '"',
    "…": "...", "—": "-",
})


def mandarin_to_english(text):
    """Convert Han runs while preserving English words and manual tone suffixes."""
    if not HAN_TEXT.search(text):
        return text
    try:
        import jieba
        from pypinyin import Style, lazy_pinyin
        from pypinyin.contrib.tone_convert import to_finals, to_initials
    except ImportError as error:
        raise RuntimeError(
            "Chinese text requires jieba and pypinyin. Install them with: "
            "python -m pip install jieba pypinyin") from error

    pieces = []
    cursor = 0
    for match in HAN_RUN.finditer(text):
        untouched = text[cursor:match.start()]
        if untouched:
            pieces.append(untouched)
        words = list(jieba.cut(match.group(1), HMM=False))
        syllables = lazy_pinyin(words, style=Style.TONE3, neutral_tone_with_five=True,
                                tone_sandhi=True)
        spellings = []
        for syllable in syllables:
            if len(syllable) < 2 or syllable[-1] not in "12345":
                raise ValueError(f"No Mandarin reading for {syllable!r}. Enter an English approximation instead.")
            sound, tone = syllable[:-1], syllable[-1]
            spelling = SYLLABLE_SPELLINGS.get(sound)
            if spelling is None:
                initial, final = to_initials(sound), to_finals(sound)
                if initial not in INITIAL_SPELLINGS or final not in FINAL_SPELLINGS:
                    raise ValueError(f"No English approximation for {syllable!r}. Enter one manually instead.")
                spelling = INITIAL_SPELLINGS[initial] + FINAL_SPELLINGS[final]
            spellings.append(spelling + tone)
        before = pieces[-1][-1:] if pieces else ""
        after = text[match.end():match.end() + 1]
        prefix = " " if before and not before.isspace() and before not in '([{"“‘《【（' else ""
        suffix = " " if after and not after.isspace() and (after.isalnum() or match.group(2)) else ""
        pieces.append(prefix + " ".join(spellings) + match.group(2) + suffix)
        cursor = match.end()

    pieces.append(text[cursor:])
    return "".join(pieces).translate(PUNCTUATION)
