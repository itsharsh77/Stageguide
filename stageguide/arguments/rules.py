"""Narrow English patterns and support-attempt detection, not fact checking."""

from dataclasses import dataclass
import re
from typing import List, Tuple

from stageguide.alignment.normalization import canonical_number, content_tokens, numeric_mentions


BENEFIT = re.compile(
    r"\b(?:improves?|improvement|reduces?|reduction|increases?|faster|better|"
    r"more efficient|more accurate|saves?|lowers?)\b", re.I,
)
VAGUE = re.compile(r"\b(?:significantly better|very effective|major improvement|huge market|highly efficient)\b", re.I)
BUSINESS_ASSUMPTION = re.compile(
    r"\b(?:users?|customers?|universit(?:y|ies)|colleges?|schools?|enterprises?|"
    r"companies|clients?|partners?|hp)\b[^.!?;]{0,60}?"
    r"\b(?:will|would|could|can|may|might)\s+(?:\w+\s+){0,2}?"
    r"(?:pay|adopt|buy|purchase|subscribe|bundle|integrate)\b", re.I,
)
ACKNOWLEDGED = re.compile(
    r"\b(?:assum(?:e|es|ing|ption)|hypothes(?:is|ize)|hypothetical|"
    r"needs? validation|to validate|plan to (?:test|validate)|not yet validated)\b", re.I,
)
GOAL = re.compile(
    r"\b(?:goal|objective|aim|hope)\b|\b(?:designed|intended) to\b|"
    r"\b(?:may|might|could) (?:improve|reduce|increase|save|lower)\b", re.I,
)
ASSERTION = re.compile(
    r"\b(?:is|are|has|have|improves?|reduces?|increases?|saves?|lowers?|"
    r"supports?|provides?|enables?|offers?|requires?|includes?|delivers?|"
    r"targets?|helps?|keeps?|costs?|works?)\b", re.I,
)
ASSERTED_BENEFIT = re.compile(
    r"\b(?:improves|reduces|increases|saves|lowers)\b|"
    r"\b(?:is|are)\s+(?:\w+\s+){0,2}?(?:faster|better|more efficient|more accurate)\b", re.I,
)
MEASURABLE_SUBJECT = re.compile(
    r"\b(?:quality|time|costs?|speed|accuracy|efficiency|latency|throughput|memory|"
    r"energy|performance|productivity|delivery|preparation|processing|inference)\b", re.I,
)
_CUES = re.compile(
    r"\b(?:according to|based on|survey|study|research|source|tested|pilot|"
    r"experiment|respondents|benchmark|data|measured)\b", re.I,
)
_RESULT = re.compile(r"\b(?:found|showed|shows|reported|observed|results?|respondents|measured|tested|adopted|paid|purchased)\b", re.I)
_NOT_SUPPORT = re.compile(
    r"\b(?:no|not|never|without|lack|lacking|need|needs|needed)\b|\b\w+n['’]t\b", re.I,
)
_PLANNED_SUPPORT = re.compile(
    r"\b(?:will|would|could|may|might|should|plans?|planned|propos(?:e|ed))\b"
    r"[^.!?;]{0,40}\b(?:survey|study|research|pilot|experiment|benchmark|test(?:ed)?|measur(?:e|ed))\b|"
    r"\b(?:survey|study|research|pilot|experiment|benchmark)\s+(?:will|would|may|might)\b", re.I,
)
_COMPARISON = re.compile(r"\b(?:compared (?:to|with)|versus|vs\.?|instead of|than|baseline)\b", re.I)
_METHOD = re.compile(r"\b(?:because|by using|through)\b", re.I)
FOLLOWUP = re.compile(
    r"^(?:(?:this|that|it)\s+(?:was|is|has been|comes from|needs|remains)\b|"
    r"(?:this|that|the)\s+(?:figure|result|estimate|improvement|reduction|claim|hypothesis|assumption)\b)", re.I,
)
LABEL = re.compile(
    r"^(?:slide|page|chapter|section|version|revision|agenda|contents|overview|introduction|summary)"
    r"(?:\s*[:#-]?\s*\d+(?:\.\d+)*)?[.!]?$", re.I,
)
_MULTIPLIER = re.compile(r"(?<![\w.])(?P<value>\d+(?:\.\d+)?)\s*[x×](?!\w)", re.I)


@dataclass(frozen=True)
class ClaimNumber:
    text: str
    key: Tuple[str, str]


def claim_numbers(text: str) -> List[ClaimNumber]:
    """Reuse alignment normalization, adding only multiplier notation such as 2x."""
    found = []
    for mention in numeric_mentions(text):
        suffix = re.match(r"\s*(?:times\b|[x×](?!\w))", text[mention.end:], re.I)
        if suffix and mention.kind == "number":
            found.append((mention.start, ClaimNumber(
                text[mention.start:mention.end + suffix.end()], ("multiplier", mention.value),
            )))
        else:
            found.append((mention.start, ClaimNumber(mention.text, mention.key)))
    for match in _MULTIPLIER.finditer(text):
        found.append((match.start(), ClaimNumber(match.group(), ("multiplier", canonical_number(match.group("value"))))))
    result, seen = [], set()
    for _, number in sorted(found, key=lambda entry: entry[0]):
        if number.key not in seen:
            seen.add(number.key)
            result.append(number)
    return result


def claim_tokens(text: str):
    # These tokens encode multiplier notation, not subject/context keywords.
    text = _MULTIPLIER.sub(" ", text)
    for mention in reversed(numeric_mentions(text)):
        suffix = re.match(r"\s*times\b", text[mention.end:], re.I)
        if suffix:
            text = text[:mention.start] + " " + text[mention.end + suffix.end():]
    return content_tokens(text)


def support_attempt(text: str) -> bool:
    """Recognize an attempted source, measurement, comparison or mechanism.

    Cues do not prove evidence is true. A bare 'data' or a planned/negated study
    is insufficient. Callers must establish connection to a particular claim.
    """
    if _NOT_SUPPORT.search(text) or _PLANNED_SUPPORT.search(text):
        return False
    cue = _CUES.search(text)
    if cue:
        if re.search(r"\b(?:according to|based on)\b", text, re.I):
            return len(content_tokens(text[cue.end():])) >= 2
        if _RESULT.search(text) or claim_numbers(text):
            return len(content_tokens(text)) >= 3
    comparison = _COMPARISON.search(text)
    if comparison and (len(content_tokens(text[comparison.end():])) >= 2 or len(claim_numbers(text)) >= 2):
        return True
    method = _METHOD.search(text)
    return bool(method and len(content_tokens(text[method.end():])) >= 3)
