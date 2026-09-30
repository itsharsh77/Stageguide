"""Small, explicit English normalization rules. No statistical models."""

from dataclasses import dataclass
from decimal import Decimal
import re
import unicodedata
from typing import FrozenSet, List, Optional, Tuple


_SMALL = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
    range(20),
))
_TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
_SCALES = {"thousand": 1000, "million": 1000000, "billion": 1000000000, "trillion": 1000000000000}
_LITERAL = r"[+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_LEX = re.compile(rf"(?<![\w.]){_LITERAL}(?!\w)|[^\W\d_]+(?:['’][^\W\d_]+)?|[%％]", re.UNICODE)
_LITERAL_FULL = re.compile(rf"{_LITERAL}\Z")
_WORDS = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
_STOP = frozenset("""
a an the and or but of to in on at by for from with as is are was were be been
being am it its this that these those we our us i my me you your they their them
he his she her will would shall should can could may might do does did have has
had very also around about approximately roughly want wants today
""".split())
_FORMS = {
    "first": "initial", "initially": "initial",
    "prepare": "preparation", "preparing": "preparation", "prepared": "preparation",
    "reduce": "reduce", "reduces": "reduce", "reduced": "reduce", "reducing": "reduce",
    "reduction": "reduce", "decrease": "reduce", "decreases": "reduce", "lower": "reduce",
    "increase": "increase", "increases": "increase", "increased": "increase", "increasing": "increase",
    "explain": "explain", "explained": "explain", "explaining": "explain",
    "targeting": "target", "targeted": "target",
    "easier": "easy", "universities": "university",
}
NEGATIONS = frozenset({"not", "no", "never", "without"})


@dataclass(frozen=True)
class NumericMention:
    text: str
    value: str
    kind: str
    start: int
    end: int

    @property
    def key(self) -> Tuple[str, str]:
        return self.kind, self.value


def canonical_number(value: str) -> str:
    number = Decimal(value.replace(",", "").replace("−", "-"))
    return "0" if number == 0 else format(number.normalize(), "f")


def _small_group(words: List[str], start: int) -> Optional[Tuple[Decimal, int]]:
    """Parse a literal, 0..19, or tens plus an optional unit."""
    if start >= len(words):
        return None
    word = words[start]
    if _LITERAL_FULL.fullmatch(word):
        return Decimal(word.replace(",", "").replace("−", "-")), start + 1
    if word in _SMALL:
        return Decimal(_SMALL[word]), start + 1
    if word in _TENS:
        value, end = _TENS[word], start + 1
        if end < len(words) and words[end] in _SMALL and 0 < _SMALL[words[end]] < 10:
            value += _SMALL[words[end]]
            end += 1
        return Decimal(value), end
    return None


def _group(words: List[str], start: int) -> Optional[Tuple[Decimal, int]]:
    result = _small_group(words, start)
    if result is None:
        return None
    value, end = result
    if end < len(words) and words[end] == "hundred" and 0 < value < 100:
        value *= 100
        end += 1
        next_start = end + 1 if end < len(words) and words[end] == "and" else end
        rest = _small_group(words, next_start)
        if rest is not None and 0 <= rest[0] < 100:
            value += rest[0]
            end = rest[1]
    if end < len(words) and words[end] == "point":
        digits = []
        cursor = end + 1
        while cursor < len(words):
            word = words[cursor]
            if word in _SMALL and _SMALL[word] < 10:
                digits.append(str(_SMALL[word]))
            elif len(word) == 1 and word.isdigit():
                digits.append(word)
            else:
                break
            cursor += 1
        if digits:
            value += Decimal("0." + "".join(digits)) * (-1 if value < 0 else 1)
            end = cursor
    return value, end


def _number(words: List[str], start: int) -> Optional[Tuple[Decimal, int]]:
    sign = 1
    cursor = start
    if words[cursor] in {"minus", "negative", "plus", "positive"}:
        sign = -1 if words[cursor] in {"minus", "negative"} else 1
        cursor += 1
    result = _group(words, cursor)
    if result is None:
        return None
    value, end = result
    if value < 0:
        sign *= -1
        value = -value
    total = Decimal(0)
    last_scale = float("inf")
    while end < len(words) and words[end] in _SCALES and _SCALES[words[end]] < last_scale:
        last_scale = _SCALES[words[end]]
        total += value * last_scale
        end += 1
        next_start = end + 1 if end < len(words) and words[end] == "and" else end
        rest = _group(words, next_start)
        if rest is None:
            return sign * total, end
        value, end = rest
    return sign * (total + value), end


def numeric_mentions(text: str) -> List[NumericMention]:
    """Normalize common cardinal numbers and percentages, preserving source spans.

    Decimal words are digit-by-digit ('five point two five'). Percentage points
    remain distinct from percentages. Word units remain in text context;
    currency symbols are not interpreted.
    """
    tokens = list(_LEX.finditer(text))
    words = [match.group().casefold() for match in tokens]
    mentions = []
    index = 0
    while index < len(tokens):
        if (words[index] not in _SMALL and words[index] not in _TENS
                and words[index] not in {"minus", "negative", "plus", "positive"}
                and not _LITERAL_FULL.fullmatch(words[index])):
            index += 1
            continue
        # Do not combine number words across sentence/list punctuation. A hyphen
        # and whitespace may connect words, e.g. twenty-five / five million.
        stop = index + 1
        while stop < len(tokens) and re.fullmatch(r"[\s-]*", text[tokens[stop - 1].end():tokens[stop].start()]):
            stop += 1
        result = _number(words[:stop], index)
        if result is None:
            index += 1
            continue
        value, end = result
        kind = "number"
        if end < stop and words[end] in {"%", "％", "percent", "percentage"}:
            kind = "percentage"
            end += 1
        elif end + 1 < stop and words[end:end + 2] == ["per", "cent"]:
            kind = "percentage"
            end += 2
        if kind == "percentage" and end < stop and words[end] in {"point", "points"}:
            kind = "percentage_points"
            end += 1
        begin, finish = tokens[index].start(), tokens[end - 1].end()
        mentions.append(NumericMention(text[begin:finish], canonical_number(str(value)), kind, begin, finish))
        index = end
    return mentions


def content_tokens(text: str) -> FrozenSet[str]:
    """Unique nonnumeric keywords, with a small auditable word-form map."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    for mention in reversed(numeric_mentions(text)):
        text = text[:mention.start] + " " + text[mention.end:]
    text = re.sub(r"\b\w+n't\b", " not ", text)
    tokens = set()
    for word in _WORDS.findall(text):
        if word.endswith("'s"):
            word = word[:-2]
        if word in _STOP:
            continue
        word = _FORMS.get(word, word)
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        tokens.add(word)
    return frozenset(tokens)


def literal_values(text: str) -> FrozenSet[str]:
    """Find raw digit values to deduplicate the ingestion parser's number list."""
    return frozenset(canonical_number(match.group()) for match in _LEX.finditer(text)
                     if _LITERAL_FULL.fullmatch(match.group()))
