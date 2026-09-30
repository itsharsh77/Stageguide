"""Conservative output checks, not a semantic truth or evidence verifier."""

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from stageguide.alignment.normalization import content_tokens, numeric_mentions

from .models import JudgeQuestion
from .prompts import grounding_context


class GroundingError(ValueError):
    """An output cannot safely replace the deterministic question."""


# These add question grammar, not domain facts or named sources. Unknown domain
# words are rejected rather than guessing whether they are harmless paraphrases.
_QUESTION_WORDS = content_tokens("""
what how why which when where whether could can would please tell clarify explain
clarification elaborate mean meaning measure measurable measurement measured test
tested evidence support supports supporting supported basis based baseline comparison
compare reasoning reason validate validation assumption assumed currently estimate
estimated figure number value quantified mentioned claim claimed said answer response
provide provided establish established determine determined derive derived arrive
arrived calculate calculated calculation willing willingness pay payment improve
improvement reduction increase amount complete missing part audience explain
specifically detail details source sources study survey research pilot experiment
benchmark observed observation actual expectation expected hypothetical intend intended
expect expecting verify verification demonstrate demonstration substantiate
""")
_CURRENCIES = {"₹": "INR", "inr": "INR", "rs": "INR", "rupees": "INR",
               "$": "USD", "usd": "USD", "dollars": "USD",
               "€": "EUR", "eur": "EUR", "euros": "EUR",
               "£": "GBP", "gbp": "GBP", "pounds": "GBP"}
_UNITS = {"x": "multiplier", "times": "multiplier", "tops": "TOPS",
          "minutes": "minute", "minute": "minute", "hours": "hour", "hour": "hour",
          "seconds": "second", "second": "second", "days": "day", "day": "day",
          "months": "month", "month": "month", "years": "year", "year": "year"}


def _text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("−", "-")
    text = re.sub(r"([+-])\s*([₹$€£])\s*(?=\d)", r"\2\1", text)
    # Existing alignment supports written numbers; expose attached common units.
    text = re.sub(r"(?<=\d)\s*×", " times", text)
    return re.sub(r"(?<=\d)(?=x\b|TOPS\b)", " ", text, flags=re.IGNORECASE)


def numeric_signatures(text: str) -> Set[Tuple[str, str, str, str]]:
    """Retain value, percentage kind, currency and simple physical units.

    Scales and written English numbers use the existing alignment normalizer.
    Unsupported numeric syntax fails closed instead of being silently ignored.
    """
    text = _text(text)
    if any(unicodedata.category(char) == "Sc" and char not in _CURRENCIES for char in text):
        raise GroundingError("Unsupported currency notation")
    if re.search(r"\d\s*[/^:]\s*\d", text):
        raise GroundingError("Unsupported fractional or ratio notation")
    mentions = numeric_mentions(text)
    covered = set()
    signatures = set()
    for mention in mentions:
        covered.update(range(mention.start, mention.end))
        before = text[:mention.start].rstrip().casefold()
        after = text[mention.end:].lstrip().casefold()
        prefix = re.search(r"(₹|\$|€|£|\binr|\busd|\beur|\bgbp|\brs\.?)$", before)
        suffix = re.match(r"(rupees|dollars|euros|pounds)\b", after)
        currency = _CURRENCIES.get(prefix.group().rstrip("."), "") if prefix else ""
        if suffix:
            currency = _CURRENCIES[suffix.group()]
        unit = re.match(r"([a-z]+)\b", after)
        signatures.add((mention.kind, mention.value, currency,
                        _UNITS.get(unit.group(), "") if unit else ""))
    if any(char.isdigit() and index not in covered for index, char in enumerate(text)):
        raise GroundingError("Unsupported numeric notation")
    return signatures


def _unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise GroundingError("Duplicate JSON fields")
        result[key] = value
    return result


def _context_texts(question: JudgeQuestion, answer: Optional[str]) -> List[str]:
    context = grounding_context(question, answer)
    texts = [context["claim"], context["finding_reason"], context["judge_question"]]
    evidence = context["evidence"]
    texts.extend([evidence["transcript_match"] or "", *evidence["supporting_excerpts"]])
    if answer is not None:
        texts.append(answer)
    # Structural IDs and alignment scores never license numbers in questions.
    return texts


def parse_and_validate(raw: str, question: JudgeQuestion, answer: Optional[str] = None) -> str:
    if not isinstance(raw, str) or len(raw) > 12000:
        raise GroundingError("Model response is not bounded JSON text")
    try:
        result = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise GroundingError("Model response is not a single valid JSON object") from exc
    field = "question" if answer is None else "follow_up_question"
    expected = {field, "grounded", "source_question_id"}
    if answer is not None:
        expected.add("reason")
    if not isinstance(result, dict) or set(result) != expected:
        raise GroundingError("Model response does not match the required schema")
    if result["source_question_id"] != question.question_id or result["grounded"] is not True:
        raise GroundingError("Model response has the wrong source or declines grounding")
    if answer is not None and result["reason"] != question.reason:
        raise GroundingError("Follow-up must preserve the authoritative finding reason")
    text = result[field]
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 500:
        raise GroundingError("Question text must be a non-empty bounded string")
    text = text.strip()
    if not text.endswith("?") or text.count("?") != 1 or "\n" in text:
        raise GroundingError("Expected exactly one concise question")
    if not re.match(r"^(what|how|why|which|when|where|can|could|would|is|are|was|were|"
                    r"do|does|did|have|has|will|you mentioned)\b", text, re.IGNORECASE):
        raise GroundingError("Expected a professional question, not an assertion")

    contexts = _context_texts(question, answer)
    # Only the claim and answer may introduce numeric subject matter. Evidence
    # excerpts are lexical context, not permission to borrow unrelated figures.
    allowed_numbers = numeric_signatures(question.claim)
    if answer is not None:
        allowed_numbers |= numeric_signatures(answer)
    required_numbers = numeric_signatures(question.question)
    generated_numbers = numeric_signatures(text)
    if not generated_numbers <= allowed_numbers or not required_numbers <= generated_numbers:
        raise GroundingError("Generated question changes, omits or introduces an unsupported numeric value/unit")

    allowed_words = set(_QUESTION_WORDS)
    for context in contexts:
        allowed_words.update(content_tokens(context))
    if not content_tokens(text) <= allowed_words:
        raise GroundingError("Generated question introduces unsupported subject matter")

    claim_words = content_tokens(question.claim)
    generated_words = content_tokens(text)
    for direction, opposite in (("reduce", "increase"), ("increase", "reduce")):
        if direction in claim_words and opposite not in claim_words and opposite in generated_words:
            raise GroundingError("Generated question reverses the claim's comparison")
    if claim_words & {"not", "no", "never", "without"} and not generated_words & {"not", "no", "never", "without"}:
        raise GroundingError("Generated question drops claim negation")
    for qualifier in ("only", "up to", "at least", "at most"):
        if re.search(rf"\b{qualifier}\b", question.claim, re.IGNORECASE) and not re.search(
                rf"\b{qualifier}\b", text, re.IGNORECASE):
            raise GroundingError("Generated question drops a claim qualifier")
    if text.casefold().startswith("you mentioned"):
        # An optional conversational preface may restate the claim, not assert
        # newly invented support. The rest must still be a single question.
        preface = re.split(r"(?<!\d)\.\s+|\.\s+(?=[A-Z])", text, maxsplit=1)[0]
        if preface == text or not content_tokens(preface) <= claim_words | content_tokens("mentioned claim figure"):
            raise GroundingError("Conversational preface adds an unsupported assertion")

    # Asking whether evidence exists is allowed; presupposing a particular study
    # exists is not, unless that type of support actually appears in source data.
    facts = " ".join([question.claim, question.evidence_reference.finding.evidence.transcript_match or "",
                      *question.evidence_reference.finding.evidence.supporting_excerpts[:2], answer or ""])
    for match in re.finditer(r"\b(?:the|your|this|that)\s+(?:[^\s?.,;]+\s+){0,3}"
                             r"(study|survey|pilot|experiment|benchmark|competitor)\b",
                             text, re.IGNORECASE):
        if not re.search(rf"\b{re.escape(match.group(1))}\b", facts, re.IGNORECASE):
            raise GroundingError("Generated question presupposes evidence not supplied")
    if re.search(r"\b(lying|liar|dishonest|stupid|fraud|wrong|false)\b", text, re.IGNORECASE):
        raise GroundingError("Generated question accuses the presenter rather than requesting support")
    return text
