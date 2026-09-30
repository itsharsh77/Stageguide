# StageGuide — Milestones 1–7

Local presentation ingestion, speech transcription, deterministic slide–speech
alignment, conservative argument analysis, grounded judge question planning and
optional local judge refinement with one follow-up. Requires Python 3.9
or later. Speech is an optional dependency; presentation ingestion is unchanged.

Milestone 5A plans questions deterministically from argument findings. Milestone
5B optionally refines those questions locally and supports one answered turn.
The deterministic planner remains usable without any language model.

Milestone 6 adds an optional local Streamlit demo connecting these existing
modules. From the repository root:

```sh
source .venv/bin/activate
python -m pip install -e '.[demo,speech-cpu]'
streamlit run app.py
```

Open **http://127.0.0.1:8501**. For a guided demonstration, choose **Sample demo**
and click **Run complete sample demo**. This uses actual local files and fresh
Whisper inference; it requires the local speech weights described below.
See [Milestone 6](#milestone-6-minimal-end-to-end-demo-ui) for configuration and
the distinction between real inference and deterministic results.

Milestone 7 verifies **real Qwen3-4B-Instruct-2507 Q4_K_M inference on Apple M4
with Metal**. The installed GGUF remains optional; deterministic questions work
without it. See [Milestone 7](#milestone-7-real-local-judge-inference) for the
verified model, launch configuration, measured results and grounding limits.

## Files to commit

Commit the source code, tests, `app.py`, `pyproject.toml`, README, `.gitignore`,
`.streamlit/config.toml` and the intentional `sample_data/` demo assets. Dependencies
are declared in `pyproject.toml`; other machines install them into their own
virtual environments.

`.gitignore` excludes virtual environments and installed packages, `.env` files,
Streamlit secrets, model weights, Python caches, build artifacts and local logs.
The documented `sample_data/real_judge_runtime.log` is intentionally retained as
reviewed demo evidence. Sanitized `.env.example` / `.env.template` files may be
committed, but must contain placeholders rather than credentials.

Use Git to upload the project: manual folder/ZIP uploads do not apply `.gitignore`.
Ignore rules also do not remove files already tracked by Git. Before committing,
review `git status --short` and use `git check-ignore -v <path>` to check exclusions.

## Setup and CLI

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m stageguide.presentation.parser sample_data/test.pptx
python -m stageguide.presentation.parser sample_data/test.pdf > parsed.json
python -m pytest -q
```

Supply your own presentation at the example path. Tests generate temporary files
and require no external fixtures, services, or network access.

## Python API

```python
from stageguide.presentation.parser import PresentationError, parse_presentation

try:
    pages = parse_presentation("sample_data/test.pptx")
    for page in pages:
        print(page.page_number, page.title, page.to_dict())
except PresentationError as error:
    print(error)
```

Each `PresentationPage` dataclass contains `page_number` (one-based for both
formats), `title` (string or `None`), `body_text`, `numbers`, `percentages`, and
`source_filename` (basename). The CLI prints a JSON array to stdout on success
and exits 0. Unsupported, missing, encrypted, or corrupted files produce a JSON
error on stderr and exit 1. Argument usage errors exit 2. Parsing is all-or-nothing.

## Extraction rules and limits

- One record per slide/page, including blank pages. Whitespace is normalized,
  blank lines removed, and meaningful line boundaries retained.
- PPTX uses `python-pptx`: title placeholders, text boxes, grouped shapes and
  table cells. Titles are omitted from the body. Shapes retain document order;
  table cells are separated by ` | `. Speaker notes and master/layout text are
  excluded. A missing or empty title placeholder gives `None`.
- PDF uses PyMuPDF text and font data in sorted geometric order. A title is
  identified only when the first nonempty line lies in the top 35% of the page,
  is at most 200 characters, and its largest font is at least 20% larger than
  every remaining line's largest font. Otherwise the title is `None` and all
  extracted text stays in the body. This heuristic can miss multiline titles
  and cannot guarantee semantic reading order in complex layouts.
- Numeric mentions come from both title and body, in that order. Lists retain
  repeated occurrences and original signs/grouping/decimal formatting as
  strings. English-style integers, comma-grouped thousands, decimals, and
  leading-dot decimals are supported. Digits embedded in identifiers are
  excluded. Written-out numbers, scientific notation, and locale-specific
  decimal/grouping conventions are not interpreted.
- Percentages recognize `%`, `％`, `percent`, `percentage`, and `per cent`.
  Their numeric parts also appear in `numbers`: `12.5%` produces `"12.5"` in
  numbers and `"12.5%"` in percentages. Internal whitespace is normalized.
- Image-only/scanned pages remain empty unless they contain a text layer.
  No OCR or extraction of embedded chart data, SmartArt, or image text is included.
- No LLM integration, RAG, embeddings, databases, cloud inference, frontend or camera features.

## Milestone 2: local speech transcription

The initial backend is **faster-whisper + Whisper tiny.en**, using CPU INT8. It
is the smallest Whisper size and a practical English baseline. PyAV bundles audio
decoding, so neither a system FFmpeg executable nor PyTorch is required. Larger
Whisper / Distil-Whisper models may be substituted as local CTranslate2 models.

Compatibility was checked before installation on macOS 27 ARM64, Python 3.9.6:
faster-whisper 1.2.1, CTranslate2 4.8.2, PyAV 15.1.0, ONNX Runtime 1.19.2 and
NumPy 2.0.2 installed from wheels; CPU INT8 availability was verified. Other
platforms should check wheel and compute-type availability. Speech dependencies
are optional and do not affect presentation-only installation.

```sh
source .venv/bin/activate
python -m pip install -e '.[speech-cpu,test]'

# One-time setup: download public weights, or copy an existing model here.
# This fetches model files only; no audio is sent anywhere.
hf download Systran/faster-whisper-tiny.en model.bin config.json tokenizer.json vocabulary.txt --local-dir models/whisper-tiny.en

# Inference only reads local audio and local model files; no downloads occur.
python -m stageguide.speech.transcriber sample_data/speech_demo.wav
python -m stageguide.speech.transcriber recording.wav --model-path models/whisper-tiny.en --language en > transcript.json
python -m pytest -q
```

The model directory is ignored by Git. The backend requires `model.bin`,
`config.json`, `tokenizer.json`, and `vocabulary.txt` (or `vocabulary.json` for
models that use that format). Missing files produce a
clear error, rather than triggering a download. CPU inference uses four threads,
INT8, segment timestamps, and the bundled voice activity detector (VAD).

Supported input extensions: WAV, MP3, FLAC, M4A, OGG, AIFF/AIF, case-insensitive.
Audio is decoded locally, mixed to mono and resampled to 16 kHz by the adapter.
Only local filesystem paths are accepted. A supported extension is not proof
of valid audio; decode failures are reported as `SpeechError`. The CLI uses
JSON stdout / exit 0 for success, JSON stderr / exit 1 for processing errors,
and exit 2 for invalid arguments.

### Python API and backend replacement

```python
from stageguide.speech import BackendConfig, create_backend
from stageguide.speech.transcriber import transcribe_audio

backend = create_backend(BackendConfig(
    backend="cpu_whisper", model_path="models/whisper-tiny.en", language="en",
))
result = transcribe_audio("sample_data/speech_demo.wav", backend)
print(result.full_transcript)
print(result.to_dict())
```

`SpeechBackend` is a Python `Protocol` with a `metadata: BackendMetadata` property
and `transcribe(path: Path) -> EngineTranscript`. The engine owns decoding,
resampling, inference, and timestamp generation. It returns neutral
`TranscriptSegment(start, end, text)` objects with chronological timestamps
in seconds relative to the original recording, plus its decoded audio duration.
No faster-whisper types appear in StageGuide's public data or metrics code.
The adapter loads the model lazily and reuses it across calls.

A future Qualcomm AI Hub / QNN adapter can implement this protocol without
changing the service or statistics. Qualcomm publishes Whisper and Distil-Whisper
models, which makes this a plausible migration path; CTranslate2 weights are
**not** QNN binaries. Export, preprocessing, token decoding, timestamps, and
device validation would still be required. No Qualcomm or cloud integration is
implemented in this project yet.

References: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[Qualcomm model implementations](https://github.com/qualcomm/ai-hub-models),
[Qualcomm Whisper-Tiny](https://aihub.qualcomm.com/models/whisper_tiny).

### Metric definitions and limits

- `full_transcript` joins nonempty segment text in order and normalizes whitespace.
  Segment text and original timestamp boundaries remain available in `segments`.
- `duration_seconds` estimates total speaking duration as the union of nonempty
  transcript segment intervals. It excludes leading/trailing silence and gaps
  between segments, and does not double-count overlaps. Pauses *within* a segment
  can still be included; this is not precise acoustic speaking-time measurement.
- `audio_duration_seconds` is the decoded recording duration, including silence.
- `word_count` counts Unicode alphanumeric tokens. Internal apostrophes and
  hyphens stay within a word (`it's`, `real-time`); punctuation alone is excluded.
  This is intended for English and other space-separated text, not linguistic
  segmentation of languages such as Chinese.
- `words_per_minute = word_count * 60 / duration_seconds`, rounded to one decimal.
  Zero duration gives 0.0, including for empty/silent transcription.
- `filler_words` counts whole tokens/phrases after transcription, case-insensitively:
  `um`, `uh`, `erm`, `er`, `hmm`, `like`, `you know`, `i mean`. Only observed fillers
  are included. Longest phrase wins for overlaps. These are deterministic lexical
  counts: meaningful uses of `like` still count, and ASR may omit hesitations.
  `detect_fillers(text, fillers=...)` accepts a custom sequence for standalone use.
- Whisper output and timestamps are estimates and may contain errors. The tiny
  English model trades accuracy for size. The VAD reduces silence-related output
  but is not a guarantee against hallucinated transcription.

### Local demonstration and tests

`sample_data/speech_demo.wav` is a short synthetic English recording produced
locally with macOS's installed Samantha voice, with no external audio source:

```sh
say -v Samantha -r 145 -o sample_data/speech_demo.wav --file-format=WAVE --data-format=LEI16@16000 'Today I will explain our presentation plan. We have three goals. First, make practice simple. Second, give clear feedback. Finally, help every speaker improve. You know, good preparation makes a difference.'
```

`sample_data/speech_demo.json` contains the actual local model output, not a mock.
To reproduce with explicit offline mode:

```sh
HF_HUB_OFFLINE=1 python -m stageguide.speech.transcriber sample_data/speech_demo.wav > sample_data/speech_demo.json
```

Unit tests generate temporary audio and mock inference. They run without model
weights, network access, or the optional speech packages. Coverage includes
structured data, timestamps, gaps/overlaps, word counts, WPM, filler matching,
audio validation, dependency/model errors, lazy inference failures, and JSON CLI.
The original presentation tests remain unchanged. Alignment is described in the
Milestone 3 section below; it does not alter speech inference or metrics.

Milestone 2 validation on this machine: **84 tests passed** (34 original presentation tests
and 50 speech tests), with five existing PyMuPDF/SWIG deprecation warnings.
The real offline demo produced 31 words in six segments, 14.76 seconds of
estimated speaking time, 126.0 WPM, and one `you know` match. A separate real
two-second silence check returned no words with socket connections blocked;
a malformed WAV was rejected. `pip check` found no broken requirements.

Environment note: NumPy 2.0.2 emitted three `matmul` runtime warnings during real
feature extraction on this Mac. The resulting features were finite, and the
matrix result matched an independent `einsum` calculation (`rtol=1e-5`,
`atol=1e-6`). This resembles the upstream
[Apple Silicon matrix-warning issue](https://github.com/numpy/numpy/issues/29820);
it is not proof that every possible input is unaffected. Warnings are left
visible on stderr, and the JSON result remains clean on stdout.

## Milestone 2.5: speech backend portability

| Role | Platform | Backend | Status |
| --- | --- | --- | --- |
| Development / fallback | macOS ARM64, CPU | `cpu_whisper` / `WhisperCPUBackend`, Whisper tiny.en | Implemented and locally tested |
| Target competition deployment | Windows ARM64 on Snapdragon | `qnn_whisper` / `QualcommQNNBackend`, Qualcomm QNN / Snapdragon NPU | **Planned; not implemented or tested** |

The CPU implementation retains the existing decoding, inference and timestamp
behavior. Its old name, `FasterWhisperBackend`, remains a compatibility alias.
Application code outside `stageguide.speech` must use `BackendConfig`,
`create_backend`, `SpeechBackend` and neutral transcript types, rather than
importing concrete adapters or their engine libraries. An import-boundary test
guards this rule for application source; adapter-specific unit tests can still
exercise the implementation directly.

Selection is explicit and independent of the host OS. `BackendConfig` defaults
to `cpu_whisper`, `models/whisper-tiny.en`, and English. The factory returns a
`SpeechBackend`; neither the application service nor metrics need to branch on
backend names. There is no automatic CPU fallback after a failed QNN selection.

```sh
# Existing transcription command continues to work, with the same JSON schema.
python -m stageguide.speech.transcriber sample_data/speech_demo.wav --backend cpu_whisper

# Read-only inspection: neither command loads a model or runs inference.
python -m stageguide.speech.transcriber --backend-info
python -m stageguide.speech.transcriber --diagnostics

# Exits 1 with a JSON error; does not load a CPU model or simulate QNN results.
python -m stageguide.speech.transcriber --backend qnn_whisper --backend-info
```

The default CPU backend metadata is:

```json
{
  "backend_name": "WhisperCPUBackend",
  "model_name": "tiny.en",
  "execution_provider": "CPU",
  "device_type": "CPU",
  "accelerator": "none",
  "is_hardware_accelerated": false
}
```

Access it through `backend.metadata.to_dict()`. Metadata describes configured
execution and does not certify that a model has loaded successfully. The model
name is an explicit label (`BackendConfig(model_name=...)` or `--model-name`),
or defaults to the model directory basename with a leading `whisper-` removed.
It is not a model identity inferred or verified from weights. A custom model
directory is therefore never automatically labeled `tiny.en`.

Metadata stays separate from `TranscriptionResult`: segments, full transcript,
durations, counts, WPM and filler fields are unchanged. On this development
backend, CPU vector instructions do not constitute GPU/NPU execution; the
reported accelerator is always `none` and hardware acceleration is `false`.

`platform_diagnostics()` reports `operating_system`, `machine_architecture`, and
`available_execution_backends`. Here, availability means an implemented backend
whose required Python modules are discoverable. Discovery does **not** load
native libraries, check ABI/DLL compatibility, inspect model files, or verify
runtime/compute support. `runtime_verified` is therefore false, and the report
states its availability basis. Diagnostics inspect all registered choices,
independently of CLI `--backend`; they never select one automatically. A Windows
ARM64 host alone never makes QNN available.

The `QualcommQNNBackend` stub in `stageguide/speech/qnn_backend.py` rejects
construction, metadata and transcription on **all** platforms. Future work must
implement its neutral interface, handle audio/model preparation and original
recording timestamps, validate actual device execution, and update factory
availability/diagnostics. Until then it makes no NPU acceleration claims.

Dependencies remain optional: use `pip install -e '.[speech-cpu,test]'` for this
development backend. The original `speech` extra remains a compatibility alias
with the same dependencies. A future Windows Snapdragon package can install
StageGuide without either CPU extra; a QNN dependency group should be introduced
only when real integration requirements are known. No Qualcomm SDK is installed
or required by this milestone, and importing/configuring speech or inspecting
metadata does not import faster-whisper, CTranslate2, PyAV or ONNX Runtime.

All original 84 test cases are retained. The existing CLI success test now mocks
the factory rather than the concrete backend constructor; its assertions are
unchanged. Added tests cover selection, truthful metadata, explicit QNN rejection
on multiple simulated hosts, dependency diagnostics and import isolation. They
require no SDK, model download, or real inference.

Milestone 2.5 validation on this Mac: **110 tests passed** (all original 84 plus
26 portability tests), with the same five PyMuPDF/SWIG warnings. A real offline
CPU CLI run through the factory exactly matched the saved Milestone 2 demo JSON;
the previously documented NumPy warnings remain. QNN selection returned a JSON
error with exit code 1 and no transcript. No SDKs or dependencies were installed
for Milestone 2.5.

## Milestone 3: deterministic slide–speech alignment

The alignment package consumes existing `PresentationPage`, `TranscriptionResult`,
`EngineTranscript` and `TranscriptSegment` data. It does not run presentation
parsers or speech inference, and adds no dependencies, models, LLMs or embeddings.
Existing input structures and speech JSON output are unchanged.

```python
from stageguide.alignment import align_slide, align_presentation
from stageguide.presentation.models import PresentationPage
from stageguide.speech.models import TranscriptSegment

slide = PresentationPage(
    page_number=1, title=None,
    body_text=("StageGuide reduces presentation preparation time by 30%.\n"
               "Our initial target is 5 million university students."),
    numbers=["30", "5"], percentages=["30%"], source_filename="demo.pptx",
)
transcript = ("Our first users will be university students. "
              "We want StageGuide to make presentation preparation easier.")
result = align_slide(slide, transcript)
print(result.to_dict())

# Reuse a complete transcript for each slide; does not infer slide timings.
results = align_presentation([slide], transcript)

# Alternatively, provide text or existing timestamped segments per page number.
results = align_presentation([slide], transcripts_by_slide={
    1: [TranscriptSegment(0.0, 8.0, transcript)],
})
```

`align_slide` accepts a string, a full speech result (uses `full_transcript`), an
engine result, a segment, or a sequence of segments. Segment text is joined in
caller-supplied order, so an ASR segment boundary need not break a phrase. The
input objects and timestamps are never modified. In presentation mode, results
retain slide order. Missing explicit assignments use empty text; duplicate slide
numbers, unknown assignment keys and mixing shared/assigned transcripts raise
`ValueError`. With a shared transcript, the same passage can cover several slides.
There is no automatic live slide-change detection or timestamp allocation.

### Items, normalization and matching

The title and each nonempty body sentence/bullet are text items. Sentence ends,
newlines, semicolons and bullet dots separate items. Items consisting only of
stop words are skipped. Each number or percentage is also a separate item,
with its source sentence as context. Repeated normalized content counts once;
the same value in distinct contexts remains separate. For example, revenue and
cost values are checked independently. The parser's duplicate numeric annotations
(`5` from `5 million`, `30` from `30%`) do not add extra claims. Values present
only in the structured numeric fields are still checked, without invented context.

Matching uses unique keywords after case folding, Unicode text normalization,
punctuation/whitespace cleanup, a fixed stop-word list, simple plural handling and
a small explicit word-form/synonym map. For example, `initial`/`first`,
`preparation`/`preparing`, and `reduces`/`decrease` match. The rules live in
`stageguide/alignment/normalization.py` and use only the standard library.

Supported English numeric forms include signed integers, comma-grouped thousands,
decimals, number words through trillion, and magnitudes such as `2.5 million`.
`five million`, `5 million` and `5,000,000` share a value. Digit-by-digit spoken
decimals such as `five point two five` normalize to `5.25`. `%`, full-width `％`,
`percent`, `percentage` and `per cent` are recognized. `thirty percent` matches
`30%`; plain `30`, `0.3`, and `30 percentage points` do **not** match `30%`.
Numeric equality uses `Decimal`, not float tolerances.

Each item is compared with individual transcript sentences. Keyword similarity
is **matched unique slide keywords / unique slide keywords**, not the fraction
of the whole transcript. Extra explanation words are therefore not penalized.
Keywords are not pooled across unrelated sentences. A simple negation-presence
guard rejects obvious polarity mismatches such as `reduces` vs `doesn't reduce`.

- **Explained:** at least 80% keyword coverage, with all required values/types
  present in that same sentence.
- **Partial:** at least 40% keyword coverage. A text item with an omitted/wrong
  number can receive at most partial credit, even with otherwise exact wording.
- **Missing:** less than 40% keyword coverage. A numeric item is also missing
  whenever its normalized value/type is absent, regardless of keyword overlap.

For a numeric item with source context, both the value/type and that context
must match. A percentage mentioned in an unrelated sentence does not satisfy
the claim. Metadata-only numeric items have no context, so they require only
the value/type. When several sentences qualify, higher classification wins,
then higher keyword similarity, then earliest occurrence for a deterministic tie.

### Coverage score and result fields

Text items have weight **1**; numeric/percentage items have weight **2**. Explained
items receive credit **1**, partial items **0.5**, and missing items **0**:

```text
coverage_score = sum(item weight × classification credit) / sum(item weights)
```

The score is rounded to four decimal places. An empty/uninformative slide scores
0.0 and returns empty item lists, rather than claiming full coverage. An empty
transcript leaves every assessable item missing. Repeated transcript wording
does not inflate the score.

`SlideAlignmentResult` contains `slide_number`, `title`, `coverage_score`,
`explained_items`, `partially_explained_items`, `missing_items`, and detailed
`items`. Each detail contains the original text, kind, numeric context (if any),
status (`explained`, `partial`, `missing`), keyword similarity, weight and matched
transcript sentence. Its `to_dict()` returns independent JSON-serializable data.
If two contexts contain the same number, their labels may repeat in the summary
lists; use the item context to distinguish them.

### Demo and limitations

Run the requested synthetic example with no input files or model setup:

```sh
python -m stageguide.alignment.demo
python -m pytest -q
```

The demo marks both body statements **partial**, and separately lists **30%** and
**5 million** as **missing**. Preparation coverage is 3/5 keywords (0.60);
student-target coverage is 3/4 (0.75). Its score is
`(1 × 0.5 + 1 × 0.5 + 2 × 0 + 2 × 0) / 6 = 0.1667`. The low score deliberately
reflects the higher weight of both omitted numerical claims. Detailed actual
output is saved in `sample_data/alignment_demo.json`.

These labels estimate lexical coverage, not proof of understanding, truth or
explanation quality. Unlisted synonyms, broad paraphrases and spelling errors
can be missed. Word order, entities, units, compound clauses and nuanced negation
are not semantically interpreted; matching words can still express different
relationships. Sentence splitting is basic, so abbreviations and long unpunctuated
transcripts may reduce accuracy. Numeric units expressed as words stay in lexical
context, but currency symbols, scientific notation, shorthand like `5m`,
fractions, ordinals and locale-specific numeric formats are not interpreted.
The English rules do not provide multilingual semantic matching. Use explicit
per-slide transcript assignments when available. Argument analysis is described
below. No Judge Mode, QNN inference or later milestone is included.

Milestone 3 validation: **170 tests passed** (all original 110 unchanged, plus
60 alignment tests), with the same five PyMuPDF/SWIG warnings. Tests cover numeric
and percentage equivalence, missing and wrong values, contextual matching,
thresholds, simple negation, empty inputs, shared/assigned transcripts, result
serialization, repeatable scoring and independence from model/parser imports.

## Milestone 4: presentation-oriented argument analysis

`stageguide.arguments` applies a small set of deterministic rules to slide claims,
the caller-selected rehearsal transcript and existing alignment results. It does
not verify facts or assert that a claim is false. A finding identifies a detected
omission, an unrecognized support attempt, vague wording, or a business assumption
that may need validation. It generates **no Judge Mode questions**.

```python
from stageguide.alignment import align_slide
from stageguide.arguments import analyze_slide, analyze_presentation
from stageguide.presentation.models import PresentationPage

slide = PresentationPage(
    1, None,
    "StageGuide reduces presentation preparation time by 30%.\n"
    "Our initial target market is 5 million university students.\n"
    "Universities will pay ₹999 per user each year.",
    ["30", "5", "999"], ["30%"], "demo.pptx",
)
transcript = (
    "StageGuide is designed for university students preparing presentations.\n"
    "It helps them practise their delivery and argument before presenting."
)
alignment = align_slide(slide, transcript)
result = analyze_slide(slide, transcript, alignment)
print(result.to_dict())

# Omitting the third argument computes alignment using the existing engine.
result = analyze_slide(slide, transcript)

# Existing shared-transcript and explicit per-slide assignment modes are reused.
results = analyze_presentation([slide], transcripts_by_slide={1: transcript})
```

Inputs may use the same strings and neutral speech result/segment objects accepted
by alignment. No parser or speech backend is invoked. If an alignment is supplied,
it is consumed without recomputation. It must correspond to the same slide and
transcript: slide number/title, source-claim and quoted-evidence checks catch
obvious mismatches, but cannot detect every stale result. The input objects and
their existing schemas are unchanged.

Claim extraction reuses alignment's sentence/bullet/title items and groups their
numeric subitems by original context. Claim wording, currency symbols and units
are retained in findings (with alignment's whitespace normalization). Equivalent
repeated content counts once. Bare headings, questions, explicit goals, simple
page/version labels and numeric metadata without claim context are skipped.
The existing alignment text helpers are now also exported as `split_sentences`
and `transcript_text`; their behavior is unchanged.

### Rules and deterministic severity

The following priority order produces **at most one finding per claim**:

| Finding | Trigger | Severity |
| --- | --- | --- |
| `ASSUMPTION_TO_VALIDATE` | Explicit future/hypothetical payment, purchase, adoption or bundling assertion involving users, customers, institutions or a named HP partner example; no connected support attempt or acknowledged hypothesis | MEDIUM |
| `UNSUPPORTED_NUMERIC_CLAIM` | Quantified benefit with omitted contextual values, or values repeated without a support attempt | HIGH when all values are omitted; MEDIUM when some values are covered or merely repeated |
| `UNEXPLAINED_NUMBER` | Other contextual slide figures are omitted | HIGH when all values are omitted; MEDIUM when only some are omitted |
| `VAGUE_CLAIM` | One of the explicit vague phrases below, with no quantitative detail in the claim or related rehearsal | LOW |
| `MISSING_KEY_CLAIM` | A substantive assertion has missing alignment coverage and no related support attempt | HIGH for at least three unique content keywords; LOW for shorter assertions |
| `PARTIALLY_EXPLAINED_CLAIM` | Alignment marks the substantive claim partial and no related support attempt resolves the concern | MEDIUM |
| `EVIDENCE_GAP` | A verbally covered, explicit benefit assertion about a measurable subject has no related support attempt | MEDIUM |

Numeric benefits include `improves`, `reduces`, `increases`, `faster`, `better`,
`more efficient`, `more accurate`, `saves`, `lowers` and close listed forms.
Nonnumeric evidence-gap findings are narrower: they require an asserted predicate
(such as `reduces` or `is faster`), at least three content keywords and a listed
measurable subject such as time, quality, costs, accuracy, latency, throughput or
performance. A benefit word by itself, or general advice such as `Practice
improves confidence`, does not produce an evidence-gap finding.

Vague wording is limited to `significantly better`, `very effective`, `major
improvement`, `huge market` and `highly efficient`. A related market-size or other
quantitative explanation suppresses this rule. Business patterns are likewise
explicit, including `Users will pay`, `Universities will adopt` and `HP can
bundle`. They do not establish that adoption or payment is impossible. Labeling
a statement as a hypothesis/assumption, acknowledging validation work, or giving
a connected support attempt suppresses redundant assumption criticism. Pricing
assumptions take precedence over a separate numeric-omission finding.

Numeric matching reuses alignment's written/digit number and percentage
normalization. A narrow adapter additionally recognizes `2x`, `2×`, and `two
times`. `₹500 per month`, `5 million users` and `45 TOPS` retain their original
claim text. This is not unit conversion or hardware verification. Values in
unrelated transcript sentences cannot satisfy a claim merely by being equal.

### Support attempts and false-positive controls

A rehearsal sentence is related when it has at least 40% of the claim's content
keywords and shares at least two of them (one for claims of only one or two
keywords). Basic negation polarity must agree, except when connecting an explicit
acknowledgment of business uncertainty such as `not yet validated`. Support may also come from an
immediately following explicit reference such as `This was measured...` or
`This hypothesis needs validation...`; arbitrary neighboring sentences are not
borrowed. These rules are lexical heuristics, not semantic coreference.

Support cues include `according to`, `based on`, `survey`, `study`, `research`,
`source`, `tested`, `pilot`, `experiment`, `respondents`, `benchmark`, `data` and
`measured`. A cue needs additional detail: a cited source phrase, result or
measurement wording, or quantitative detail. Bare `we have data`, planned studies
and negated evidence do not qualify. Comparisons with an identified baseline or
mechanism phrases such as `because`, `by using` and `through` can also qualify
when followed by explanatory content.

These signals show only an **attempt to support** a claim, not valid evidence.
Connected attempts suppress weak critiques from lexical paraphrase misses.
Unrelated evidence cannot suppress a finding. If a numeric value is still
omitted, an explanation of the general concept alone does not remove the numeric
finding. A source printed on the slide is not automatically counted as verbal
support: the reasons explicitly concern the supplied rehearsal.

### Results and traceability

`SlideArgumentResult` has `slide_number`, `findings` and `to_dict()`. Every
`ArgumentFinding` includes:

- `type`, deterministic `severity`, original `claim` and explanatory `reason`.
- Qualitative `confidence` (`high` for directly detected numeric omissions or
  explicit business patterns, otherwise `medium`). This describes confidence in
  the rule observation, **not a probability that the claim is false**.
- `evidence`: source slide text, alignment's transcript match/status/similarity,
  related transcript excerpts, recognized supporting excerpts, original numeric
  phrases, values not matched in context, triggering phrases and a stable rule ID.

The exact rule IDs are `business-assumption`, `numeric-omission`, `numeric-support`,
`vague-unmeasured`, `alignment-missing`, `alignment-partial` and `evidence-gap`.
No additional argument score is introduced. Empty slides produce no findings;
an empty transcript means no verbal coverage is available, rather than proof that
the presenter has no evidence elsewhere.

### Requested demo

```sh
python -m stageguide.arguments.demo
python -m pytest -q
```

The actual output is saved in `sample_data/argument_demo.json`. It contains:

| Claim | Finding | Why |
| --- | --- | --- |
| StageGuide reduces presentation preparation time by 30%. | HIGH — `UNSUPPORTED_NUMERIC_CLAIM` | Related rehearsal text covers 60% of the slide's keywords, but never states `30%`. No related support attempt is detected. Rule: `numeric-omission`. |
| Our initial target market is 5 million university students. | HIGH — `UNEXPLAINED_NUMBER` | Related rehearsal text covers the university-student topic (40% keyword coverage), but omits `5 million`. Rule: `numeric-omission`. |
| Universities will pay ₹999 per user each year. | MEDIUM — `ASSUMPTION_TO_VALIDATE` | The explicit `Universities will pay` pattern is a future payment assertion. No related validation attempt or acknowledgment of uncertainty is detected. Rule: `business-assumption`. |

None of these findings says the claim is false. They identify where explanation,
support or validation is needed in the supplied rehearsal.

Limitations: the analyzer inherits alignment's English lexical limitations and
can miss paraphrases, units, clause relationships, subtle negation and implicit
support. It does not verify source credibility, causality, business feasibility
or hardware specifications. A missing finding is not a certificate that an
argument is sound. The explicit rules and tests favor restraint over speculative
criticism. No LLM, embeddings, cloud API, Qualcomm SDK, Judge Mode or conversational
Q&A is added.

Milestone 4 validation: **234 tests passed** (all original 170 unchanged plus 64
argument-analysis tests), with the same five PyMuPDF/SWIG deprecation warnings.
Tests cover all seven finding types, severity, evidence traceability, supported
and unsupported numbers, multiplier notation, repeated claims, unrelated/planned/
negated evidence, acknowledged assumptions, empty inputs, input compatibility,
non-mutation, deterministic output and import isolation. No dependencies or model
downloads were needed.

## Milestone 5A: Grounded judge question planning

The `stageguide.judge` package consumes existing `SlideArgumentResult` objects.
The planner does not re-run argument analysis or inspect slides or transcripts.
Its only source of claims, severity and evidence is the supplied findings.
There are no new dependencies, inference calls or downloads.

```python
from stageguide.arguments import analyze_presentation
from stageguide.judge import get_top_questions, plan_questions, plan_slide_questions

findings_by_slide = analyze_presentation(pages, transcript)
questions = plan_questions(findings_by_slide)
top_five = get_top_questions(findings_by_slide, limit=5)

# When holding individual ArgumentFinding objects, supply their slide number:
if findings_by_slide:
    first = findings_by_slide[0]
    one_slide = plan_slide_questions(first.slide_number, first.findings)
if questions:
    print(questions[0].to_dict())
```

`JudgeQuestion` contains `slide_number`, `question_id`, `priority`, `category`,
`source_finding_type`, original `claim`, generated `question`, original `reason`,
`evidence_reference` and `additional_evidence_references`. Each reference has a
slide number, finding ID and full `ArgumentFinding` snapshot, including every
evidence field and the original confidence. `to_dict()` produces JSON-compatible
data. Snapshot lists are copied so modifying input evidence later does not alter
planned questions. These deterministic findings remain the source of truth for
any future phrasing layer.

### Templates, priority and selection

| Finding type | Question focus | Category |
| --- | --- | --- |
| `UNSUPPORTED_NUMERIC_CLAIM` | Evidence for the quantified claim | evidence |
| `UNEXPLAINED_NUMBER` | How the figure was derived | validation |
| `MISSING_KEY_CLAIM` | Explain the claim to the audience | clarification |
| `PARTIALLY_EXPLAINED_CLAIM` | Complete the explanation | clarification |
| `EVIDENCE_GAP` | Supporting evidence or comparison | evidence |
| `VAGUE_CLAIM` | Meaning in measurable terms | measurement |
| `ASSUMPTION_TO_VALIDATE` | Validate the business assumption | validation |

Templates quote the claim, with narrow grammar rewrites for simple percentage
reductions, target-market estimates and payment assumptions. More complex claims
retain their full wording, including conditions, negations and numerical values.
The planner never invents a baseline, source, number or counterclaim, and does
not state that a finding proves the presenter wrong.

Priority exactly matches finding severity: HIGH, MEDIUM, LOW. Ranking uses
severity first, then the following explicit type tie-break: unsupported numeric,
unexplained number, missing claim, evidence gap, assumption, partial explanation,
vague claim. This deliberately favors the requested numeric/evidence challenges
over other questions of the same severity. Equal severity and type preserve
the supplied presentation order and then finding order; slides are not reordered
by their numeric labels. There is no additional score.

`get_top_questions` takes a non-negative integer limit (default 5), selects after
deduplication and keeps IDs unchanged. Zero returns no questions; limits larger
than the plan return the full plan. Empty findings produce no questions.

### Deduplication and finding IDs

Within each slide, case, whitespace, quotation marks and ordinary punctuation
variants of a claim form one group. Word order, negations, numeric values,
decimal separators, percentage signs, currencies and comparison signs remain
significant. The highest-ranked finding supplies the question, and all other
findings in that group remain in `additional_evidence_references`. For example,
an unsupported numeric claim and an evidence gap for the same claim produce
one question with two source references. Identical claims on different slides
remain separate so their slide-specific evidence is preserved.

This is conservative textual deduplication, not semantic paraphrase detection.
Substantially reworded claims may still yield separate questions; approximate
similarity is avoided to prevent merging distinct quantitative assertions.

Existing findings have no persistent IDs. The planner uses their original
one-based positions: `slide4_f2` refers to the second finding on slide 4, and
`slide4_q2` is the question it supplies. These are deterministic references within
an input snapshot, not permanent database identifiers. IDs may have gaps after
deduplication and change if source findings are reordered. Supply each slide
number once per call; duplicate or invalid slide numbers, empty claims, unknown
types/severities and invalid limits fail with a clear `ValueError`.

### Demo and validation

```sh
python -m stageguide.judge.demo
python -m pytest -q
```

The demo composes the existing argument analyzer with the planner for the exact
Milestone 4 slide and transcript. Complete output, including original findings,
is saved in `sample_data/judge_demo.json`:

1. **HIGH**, `slide1_q1` → `slide1_f1`, `UNSUPPORTED_NUMERIC_CLAIM`:
   “What evidence supports the claimed 30% reduction in presentation preparation time by StageGuide?”
2. **HIGH**, `slide1_q2` → `slide1_f2`, `UNEXPLAINED_NUMBER`:
   “How did you arrive at the estimate of 5 million university students?”
3. **MEDIUM**, `slide1_q3` → `slide1_f3`, `ASSUMPTION_TO_VALIDATE`:
   “What evidence suggests universities would be willing to pay ₹999 per user each year?”

The first two findings identify values omitted from related rehearsal sentences;
the third identifies an explicit payment assumption without a detected validation
attempt. Their original claims, explanations and evidence are retained verbatim
in the saved output. No new criticism is inferred by the question planner.

Milestone 5A stops at deterministic question planning. Local LLM phrasing,
conversational follow-ups and later Judge Mode functionality remain unimplemented.

Validation: **303 tests passed** (all 234 existing tests unchanged plus 69 new
planner tests), with the same five PyMuPDF/SWIG deprecation warnings. Coverage
includes all seven templates, priority and ordering, top-N selection, duplicate
suppression and retained references, exact evidence snapshots, empty inputs,
determinism, numeric/negation preservation, the real analysis-to-planner demo
and isolation from optional model libraries. No downloads were required.

## Milestone 5B: Local AI Judge

`LocalJudge` consumes existing `JudgeQuestion` objects. The original question,
claim, priority, reason and complete finding references are never replaced by
model output. The optional model improves phrasing or asks **one follow-up**
about the supplied answer and original weakness. It does not generate findings,
grade answers, verify evidence or run an autonomous conversation loop.

### Runtime choice and Snapdragon portability

The development adapter uses **llama.cpp via `llama-cpp-python`**, with an existing
local GGUF file, in-process chat inference and JSON-schema constrained generation.
Only `local_backend.py` imports that package, lazily. No HTTP service, remote LLM
API, PyTorch, Transformers, model downloader or server framework is involved.
The optional `judge-local` extra is separate from core and speech dependencies.

The inspected Mac is Darwin ARM64, using Python 3.9.6. Upstream package metadata
supports Python 3.8+, but its documented prebuilt Metal wheels target Python
3.10–3.12. Python 3.9 therefore needs a suitable source build with a compiler.
The original Milestone 5B adapter set `n_gpu_layers=0` and disabled KV
offloading. Milestone 7 adds verified Metal execution while preserving explicit
CPU selection and the same backend interface. Neither path reports NPU use.
See [installation and structured-output documentation](https://github.com/abetlen/llama-cpp-python)
and [package compatibility](https://github.com/abetlen/llama-cpp-python/blob/main/pyproject.toml).

The configured target model is **Qwen3-4B-Instruct-2507**, an instruction model
without thinking mode. Use a corresponding trusted quantized GGUF for development
(for example Q4_K_M); conversion/quantization is a separate provisioning step.
The model name is configurable, and its metadata label does not verify file
identity. See the [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

Qualcomm publishes separate Qwen3-4B-Instruct-2507 deployment assets for
Snapdragon X Elite and X2 Elite and documents GenieX/QAIRT deployment. This gives
the model family a supported target direction, but does not make a GGUF file
interchangeable with Qualcomm assets. See the
[Qualcomm model card and deployment links](https://huggingface.co/qualcomm/Qwen3-4B-Instruct-2507).
Validate the actual Windows ARM64 device, asset and SDK combination when that
backend is implemented. **StageGuide's Qualcomm adapter is currently a stub on
every platform**, and no Qualcomm SDK is installed or executed by this milestone.

`JudgeBackend` exposes only `metadata` and `generate(JudgeRequest) -> str`.
Requests contain a system instruction, one finding's JSON context, a response
schema and conservative generation settings. A future Qualcomm adapter must
implement that same contract using supported on-device APIs. Prompt construction,
parsing, grounding checks, interaction records and fallback remain outside it.

### Setup and configuration

The runtime and weights are optional. Nothing downloads automatically, including
when a model is missing. The default backend is `disabled`.

```sh
# Optional runtime only; this may compile native code on Python 3.9/macOS.
python -m pip install -e '.[judge-local]'

# Supply a previously downloaded, trusted GGUF file. The default label is
# Qwen3-4B-Instruct-2507; change --model if using another compatible chat model.
python -m stageguide.judge.local_demo --backend local \
  --model-path /absolute/path/Qwen3-4B-Instruct-2507-Q4_K_M.gguf

# Both commands work without inference dependencies or model files:
python -m stageguide.judge.local_demo --backend disabled
python -m stageguide.judge.local_demo --backend qualcomm
```

```python
from stageguide.judge import JudgeConfig, LocalJudge, create_judge_backend

config = JudgeConfig(
    backend="local", model="Qwen3-4B-Instruct-2507",
    model_path="/absolute/path/model.gguf", context_size=4096,
)
backend = create_judge_backend(config)
judge = LocalJudge(backend)

# question is an existing Milestone 5A JudgeQuestion.
refined = judge.refine(question)
follow_up = judge.follow_up(question, "We expect AI to automate preparation.")

# Or construct one complete in-memory record in a single call:
interaction = judge.interact(question, "We expect AI to automate preparation.")
print(interaction.to_dict())
print(backend.metadata.to_dict())
```

`interact` makes at most two model calls: refinement and one follow-up. Independent
`refine`/`follow_up` methods allow callers to display a question before receiving
the answer. Pass the original `JudgeQuestion` as the source in either case.
There is no internal history loop or database. An empty/whitespace-only answer
returns no follow-up and makes no follow-up inference call.

The factory supports `disabled`, `local` and `qualcomm`. Unknown selections raise
`ValueError`. Selecting `qualcomm` exposes `not_implemented` metadata; direct
generation raises `JudgeBackendError`, while `LocalJudge` returns the deterministic
fallback. It never silently routes Qualcomm requests to the CPU backend.

### Grounding, structured output and fallback

The prompt includes only the original claim, finding type/reason, deterministic
question, primary evidence reference, matched transcript excerpt and up to two
supporting excerpts. Follow-ups also receive the presenter's answer. Other slides
and their findings are not sent. Context strings are explicitly treated as data,
not instructions. Context exceeding 8,000 characters falls back instead of being
silently truncated. The runtime also enforces its configured token context limit.

Generation uses temperature 0, seed 0 and a 384-token output limit. JSON-schema
generation is followed by independent strict parsing and validation:

- Require exactly the requested fields, a matching source question ID, boolean
  `grounded: true`, and one bounded question. Reject malformed JSON, duplicate
  keys, extra fields, code fences, incomplete generations and multiple questions.
- Normalize numeric values using the existing alignment utility. Keep percentage
  vs percentage-point values, scales, signs, common currencies, multipliers and
  simple time units distinct. Every number in the deterministic question must
  remain present. Generated values must occur in the original claim or, for a
  follow-up, the answer. Structural IDs, scores and unrelated evidence figures
  never license new numbers. Unsupported numeric syntax fails conservatively.
- Restrict content vocabulary to the supplied context plus neutral judging
  language. Reject unknown domain terms/entities, obvious unsupported study
  presuppositions, reversed increase/reduction wording, dropped negation or key
  qualifiers, and accusatory wording. A conversational `You mentioned…` preface
  may restate the claim but cannot assert new support.
- Follow-up `reason` must copy the original finding reason exactly, enforced both
  in the schema and parser. This explains why the challenge exists without
  inventing a new finding or pretending to have graded the answer.

`grounded` in an accepted response means **these conservative checks passed**,
not that a semantic entailment engine verified the text or that the claim is true.
Lexical checks can reject valid paraphrases and cannot prove every possible
sentence relationship. Specialized units/identifiers and lengthy inputs may also
fall back. The original finding remains authoritative in every case.

For missing models, disabled/unavailable backends, failed inference or failed
validation, the result's question is **the original Milestone 5A text unchanged**.
`used_fallback: true` and `fallback_reason` explain the outcome; rejected raw model
output is not displayed as an accepted question. A failed follow-up likewise
returns the original question with fallback status, allowing a future caller to
display it or suppress repetition. It is not labeled as a generated follow-up.

`JudgeResponse` retains `source_question`, all nested evidence, the original
reason, source ID, metadata and fallback status. Its `question` property is the
display text; `to_dict()` uses `question` for refinement or `follow_up_question`
for follow-ups. `JudgeInteraction` adds the exact answer and both responses.
Input evidence lists are copied, preserving traceability after later mutation.

### Metadata and actual demonstration on this Mac

Metadata contains `backend_name`, `model_name`, `execution_provider`, `device_type`,
`accelerator`, `is_hardware_accelerated`, `available` and `status`. Local status can
be `model_missing`, `runtime_missing`, `ready_to_load`, `loaded` or
`inference_failed`. `ready_to_load` checks prerequisites only; it is not proof of
successful inference. Provider/device fields describe the configured execution
path, and must be read together with availability/status.

No local language-model runtime or weights were installed during Milestone 5B.
This historical result is superseded by the real-model verification in Milestone 7.
The real command `python -m stageguide.judge.local_demo --backend local` produced
the saved `sample_data/local_judge_demo.json`, with full finding trace and:

```json
{
  "backend_name": "LocalDevelopmentJudgeBackend",
  "model_name": "Qwen3-4B-Instruct-2507",
  "execution_provider": "llama.cpp/CPU",
  "device_type": "CPU",
  "accelerator": "none",
  "is_hardware_accelerated": false,
  "available": false,
  "status": "model_missing"
}
```

The actual output therefore preserves:
“What evidence supports the claimed 30% reduction in presentation preparation
time by StageGuide?” Both the missing-model path and explicitly disabled path
return that text with fallback status. The answer is stored as:
“We expect that because AI automates some preparation work.” No actual model
refinement or generated follow-up is claimed in this saved demo.

Unit tests exercise these accepted examples using **mock inference only**:

- Refinement: “You mentioned a 30% reduction in preparation time. What evidence
  supports that figure?”
- Follow-up: “Is the 30% figure based on a measured test, or is it currently an
  assumption?”

Both retain `slide1_q1` → `slide1_f1` → `UNSUPPORTED_NUMERIC_CLAIM` →
“StageGuide reduces presentation preparation time by 30%.” The original reason
and complete evidence remain attached. These are test fixtures, not recorded
local-model outputs.

### Tests and optional real-model integration

```sh
python -m pytest -q

# Explicit opt-in only, with existing files; never downloads weights:
STAGEGUIDE_RUN_LOCAL_JUDGE=1 \
STAGEGUIDE_JUDGE_MODEL=/absolute/path/model.gguf \
python -m pytest -q -m local_model
```

The integration test skips unless explicitly enabled and the local model/runtime
are present. When enabled, it requires real refinement and follow-up responses
to pass validation without fallback, so a broken model cannot produce a false
integration success. Normal unit tests mock inference and need no model files.

This milestone stops at a controlled single follow-up. No UI, camera, remote
LLM API, Qualcomm runtime implementation or autonomous conversation is added.

Milestone 5B validation: **405 passed, 1 skipped**, with the same five
PyMuPDF/SWIG deprecation warnings. All original 303 tests remain unchanged;
102 new unit tests cover refinement, follow-ups, fallback, schema parsing,
numeric grounding, evidence preservation, backend configuration and isolation.
At Milestone 5B, the optional integration test was skipped because no local
model/runtime was available. Its only changed pre-existing Python file was `judge/__init__.py`,
which exports the new runtime-neutral API; existing implementations are intact.

## Milestone 6: Minimal end-to-end demo UI

The optional Streamlit application in `app.py` provides four workflow tabs and
an On-Device Status sidebar. The thin `stageguide.demo.controller` module manages
in-memory session results and calls existing public APIs:

| UI step | Existing service | Display |
| --- | --- | --- |
| Upload | `parse_presentation` | Filename, page count, extracted titles and previews |
| Rehearse | `create_backend`, `transcribe_audio` | Audio player, duration, words, WPM, fillers, transcript and timestamps |
| Review: coverage | `align_presentation` | Per-slide coverage, explained, partial and missing items |
| Review: arguments | `analyze_slide` with the existing alignment result | Findings grouped HIGH / MEDIUM / LOW, original claims, reasons and evidence |
| Face the Judge | `get_top_questions`, `LocalJudge.refine`, `LocalJudge.follow_up` | Top five grounded questions, one submitted answer and at most one validated follow-up |

No parsing, speech processing, analysis, scoring, question generation or grounding
validation is reimplemented in the UI. The existing backend Python files and
their tests are unchanged. Streamlit is an optional `demo` dependency, constrained
to `>=1.40,<1.51` for this project's Python 3.9 development environment; the
installed and tested version is 1.50.0. No additional web framework is used.

### Install and launch

Use the commands at the top of this README, or launch directly without activating
the existing virtual environment:

```sh
.venv/bin/python -m streamlit run app.py
```

Run from the repository root so `.streamlit/config.toml` is applied. The demo
binds to `127.0.0.1`, disables Streamlit usage telemetry and hides raw exception
details in the normal UI. It is a local competition demonstration, not a hosted
service. Existing controller/unit tests do not import or execute Streamlit.

### Current backends and configuration

The current Mac speech backend is **WhisperCPUBackend / tiny.en / CPU / accelerator
none**, with hardware acceleration false. The default local model directory is
`models/whisper-tiny.en`. The sidebar initially reports configured metadata, not
proof that weights have loaded. After successful transcription it reports local
transcription completed for the session. Missing weights or backend failures show
a short error and do not display stale results.

The judge defaults to the replaceable `local` backend with the
`Qwen3-4B-Instruct-2507` model label. **Current LLM inference is unavailable unless
a compatible local GGUF and the optional `judge-local` runtime are installed.**
With no model, the sidebar reports Model Missing and the question area displays
Deterministic Judge Mode. No weights download automatically.

Backend settings in the sidebar allow local model paths and backend selection.
After adding a compatible model, apply those settings and select a question;
the same UI uses `JudgeBackend` through the existing service. Installing an LLM
is not necessary to demonstrate the full deterministic workflow. A future
implemented Qualcomm adapter can provide provider/device/accelerator metadata
to the same status component; the UI does not infer hardware from the OS.

Optional environment defaults:

```sh
export STAGEGUIDE_SPEECH_BACKEND=cpu_whisper
export STAGEGUIDE_SPEECH_MODEL=/absolute/path/whisper-tiny.en
export STAGEGUIDE_JUDGE_BACKEND=local
export STAGEGUIDE_JUDGE_MODEL=/absolute/path/model.gguf
streamlit run app.py
```

Windows on Snapdragon with Qualcomm QNN/GenieX/QAIRT remains a **planned target**.
The current Qualcomm adapters are unavailable stubs. Selecting one does not
simulate successful inference or route an NPU request silently to CPU. This Mac
application is not currently NPU accelerated.

### Demo inputs and actual end-to-end results

`sample_data/demo_presentation.pdf` is a new two-page illustrative deck, paired
with the existing `sample_data/speech_demo.wav` synthetic voice recording:

- Page 1 contains the three rehearsal goals spoken in the recording.
- Page 2 deliberately introduces the illustrative 30% improvement, five-million
  market-size and INR 999 payment assertions that the recording does not defend.
  These are demonstration inputs, not verified product claims.

Sample mode always parses this PDF and runs fresh local speech transcription.
It never loads saved JSON as simulated results. A real end-to-end run through
Streamlit's application test harness produced `sample_data/end_to_end_demo.json`:

- **Real local model inference:** Whisper tiny.en CPU transcribed 31 words, with
  14.76 seconds of speaking time, 126 WPM and one `you know` filler. The input
  recording is 14.96175 seconds long.
- **Deterministic processing:** parser output, speech metrics, page coverage
  (100% and 0%), the three page-2 argument findings, and grounded judge questions.
- **Judge fallback, no LLM inference:** the original deterministic question is
  displayed and the presenter's answer is retained. The unavailable LLM's fallback
  repeat is not shown as a generated follow-up. The UI explicitly states that no
  validated local-model follow-up is available.

The saved report retains question → finding → original claim references. It is
an inspection artifact only. Users can download their current review or interaction
JSON directly from the UI. No database or persistent session store is added.

### Scope, session behavior and errors

PPTX and PDF uploads use the original parser; PDF OCR is not added. Audio uploads
support the existing WAV, MP3, FLAC, M4A, OGG and AIFF formats, with WAV recommended
for the demo. Uploads are limited to 50 MB. There is no microphone or camera path.

The complete rehearsal is compared with every slide. The UI says this explicitly;
there is no automatic slide-change detection, and no new aggregate score is
invented. Scores and findings come directly from the existing modules.

Each browser session owns its controller and model instances. Uploaded files are
temporarily materialized for the path-based APIs and removed when each operation
finishes, including on failure. Streamlit retains upload bytes and session results
in memory. A server/session restart clears them. Changing presentation or audio
invalidates downstream results; changing speech configuration requires a new
transcription; changing judge configuration preserves review but clears prior
judge responses. Ordinary rerenders reuse results instead of rerunning inference.

Missing/invalid inputs, unsupported or corrupt files, transcription errors, empty
speech, missing findings, unavailable judges and backend errors have explicit UI
states. Empty speech does not silently create an argument critique. No findings
means no generated judge questions. Submitted answers are displayed as supplied;
the application does not claim to grade them. Only a non-fallback response that
passes the existing judge validation is shown as a generated follow-up.

No UI routes call cloud APIs or remote models, and no Qualcomm SDKs, databases,
authentication, camera analysis or additional backend capabilities are introduced.

Validation: **429 passed, 1 skipped**, with the same five PyMuPDF/SWIG
deprecation warnings. All 405 original tests and every pre-existing backend
Python file are unchanged. The 24 new controller tests cover service composition,
temporary-file cleanup, invalid/empty inputs, safe errors, state invalidation,
metadata, evidence preservation, and suppression of fake follow-ups. Streamlit's
application test harness additionally exercised sample selection, real CPU
transcription, review, answer submission, rerenders and switching back to uploads.
The app was launched locally on port 8501 and its rendered layout inspected.

## Milestone 7: Real local Judge inference

The development Mac now runs **Qwen3-4B-Instruct-2507 Q4_K_M** through the existing
optional llama.cpp adapter. No new inference framework, reasoning engine or
Qualcomm SDK was added. All 429 previous unit tests and the original real-model
integration test remain unchanged. Presentation, speech, alignment, argument
analysis, question planning, judge validation and judge service implementations
are unchanged.

### Verified model and runtime

Exactly one model variant was downloaded:

- File: `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.
- Source: [LM Studio community GGUF, quantized by bartowski](https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF),
  derived from [Qwen's original model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
- Revision: `4edb920b6f14e3b9284d4502a6485103d72cde05`.
- License: [Apache 2.0](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE).
  A license copy is stored beside the local weights.
- Download size: **2,497,280,448 bytes**, approximately 2.50 GB / 2.33 GiB.
- Verified SHA-256: `8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc`.
- Runtime: **llama-cpp-python 0.3.35**, built with `GGML_METAL=on` on Python 3.9.6.
  The GGUF reports the `qwen3` architecture and Q4_K Medium quantization; actual
  schema-constrained chat inference verifies compatibility with this runtime.

Q4_K_M is a practical four-bit development compromise between memory and output
quality. The pre-download working-memory estimate was **4–6 GiB with a 4096-token
context**, on this **Apple M4 / 16 GiB unified-memory** Mac. This is a capacity
estimate, not a measured peak-memory claim. The model's much larger advertised
context is not allocated by StageGuide.

Weights live under ignored `models/qwen3-4b-instruct-2507/` and are never fetched
by the app or tests. `sample_data/judge_model_provenance.json` records the exact
source URL, revision, license and verified checksum. No account or API credential
is required for local inference.

### Install and run

The runtime and this one GGUF are already installed on the development Mac.
For a fresh Apple Silicon environment, the optional source-build command is:

```sh
source .venv/bin/activate
CMAKE_ARGS='-DGGML_METAL=on' CMAKE_BUILD_PARALLEL_LEVEL=4 \
  python -m pip install -e '.[judge-local,demo]' --no-cache-dir
```

This uses the existing Xcode Command Line Tools, as described in the
[llama-cpp-python installation documentation](https://github.com/abetlen/llama-cpp-python).
Provision the verified GGUF separately using the pinned URL in the provenance
file; check its byte size and SHA-256 before configuring it.

From the repository root, launch the existing UI with the local model:

```sh
STAGEGUIDE_JUDGE_MODEL="$PWD/models/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
  .venv/bin/streamlit run app.py --server.port 8502
```

The Milestone 7 server uses **http://127.0.0.1:8502**, leaving the earlier server
on port 8501 untouched. Select **Sample demo → Run complete sample demo → Face
the Judge**, then enter an answer and submit it. No UI redesign was needed.
The sidebar changes from Ready To Load to Loaded after real inference, and
reports `llama.cpp/Metal`, `Apple Silicon`, accelerator `Metal`, hardware
acceleration `true` and availability `true`.

`JudgeConfig.acceleration` and `STAGEGUIDE_JUDGE_ACCELERATOR` accept:

- `auto` (factory/UI default): attempt Metal on Apple Silicon with a supporting
  runtime; otherwise explicitly use CPU. Unrecognized offload diagnostics cause
  the candidate model to close and reload with GPU and KV/operator offload disabled.
- `cpu`: explicitly disable model, KV and operator offload. The direct adapter
  constructor retains this default for backward compatibility.
- `metal`: require supported, confirmed Metal execution; failure reaches the
  existing deterministic fallback without claiming successful acceleration.

Metal metadata requires a successful context load plus native diagnostics showing
positive GPU layer offload and an allocated Metal model buffer. The recorded run
offloaded **37/37 layers**. A platform name or requested configuration alone does
not set the acceleration flag. Native load diagnostics are recorded in
`sample_data/real_judge_runtime.log`.

The runtime remains optional and isolated in `judge/local_backend.py`.
`QualcommJudgeBackend` remains an unavailable integration stub; Windows on
Snapdragon / QNN / GenieX / QAIRT is still planned, not implemented or tested.

### Actual grounded result and performance

Run the standalone demonstration:

```sh
.venv/bin/python -m stageguide.judge.local_demo --backend local \
  --model-path models/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --acceleration metal
```

The real run saved in `sample_data/real_local_judge_demo.json` contains:

1. Original claim: **StageGuide reduces presentation preparation time by 30%.**
2. Deterministic question: **What evidence supports the claimed 30% reduction in
   presentation preparation time by StageGuide?**
3. Model refinement: **You mentioned StageGuide reduces presentation preparation
   time by 30%. What evidence supports this claimed reduction?**
4. Presenter answer: **We expect that because AI automates some preparation work.**
5. Model follow-up: **Is the 30% reduction based on a measured test or currently
   an assumption?**

Both model responses passed the existing validator, retained 30%, and preserved
`slide1_q1 → slide1_f1 → UNSUPPORTED_NUMERIC_CLAIM → original claim`. The entire
finding, original reason and evidence trace are attached to each response. The
follow-up's reason is copied from the deterministic finding, not invented by Qwen.

Measured on Apple M4 / llama.cpp / Metal with warmed OS/Metal caches:

| Call | Total elapsed | Model loading | Generation | JSON output tokens | Native decode tokens/s |
|---|---:|---:|---:|---:|---:|
| Refinement | 5.268 s | 1.988 s | 3.280 s | 40 | 26.79 |
| Follow-up | 3.503 s | <0.001 s | 3.503 s | 72 | 29.69 |

These are measurements from one run, not latency guarantees. Total time includes
lazy model loading but excludes subsequent grounding validation. Output token
counts come from runtime usage and include JSON fields. Native decode throughput
uses `llama_perf_context` and excludes prompt evaluation and model loading; it
is not output tokens divided by total elapsed time. Unavailable runtime metrics
are `null`, never estimated or fabricated. Diagnostics do not affect scoring.

### Grounding, fallback and verification

Generation still uses temperature zero, seed zero, bounded output and the original
strict JSON schema. Real responses initially introduced unsupported synonyms;
the existing validator rejected them. The prompt now requests shorter phrasing,
context vocabulary and a precise restatement/question format. **No grounding
rules were relaxed**, and there are no silent retries or fabricated responses.
`sample_data/real_judge_validation_checks.json` retains the actual rejected
prompt-development responses and their measured inference diagnostics.
This conservative lexical validator can still reject harmless paraphrases; its
safe response is the original deterministic question, not weaker validation.

The demo deliberately uses a disabled backend as well, and confirms the exact
original question with `used_fallback: true` and
`fallback_reason: backend_unavailable:disabled`. Reproduce independently with:

```sh
.venv/bin/python -m stageguide.judge.local_demo --backend disabled
```

The existing Streamlit application test harness also exercised the actual sample
PDF and WAV: real Whisper CPU transcription (31 words), deterministic coverage /
findings / planning, real Qwen Metal refinement and follow-up, and then switching
the UI to disabled Judge Mode. It raised no UI exceptions, displayed the loaded
Metal status, and preserved finding references. Full results are in
`sample_data/real_judge_ui_demo.json`. Saved JSON is an audit artifact; the UI does
not replay it as fake inference.

```sh
# Model-independent suite: 451 passed, 1 skipped.
.venv/bin/python -m pytest -q

# Full suite including the original optional real-model integration test:
STAGEGUIDE_RUN_LOCAL_JUDGE=1 \
STAGEGUIDE_JUDGE_MODEL="$PWD/models/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
  .venv/bin/python -m pytest -q
```

Final full validation: **452 passed in 10.66 seconds**, including real refinement
and follow-up without fallback. The five pre-existing PyMuPDF/SWIG deprecation
warnings remain. The 22 new model-independent tests cover acceleration selection,
verified metadata, explicit CPU execution, unverified offload, timing/token
diagnostics and failure fallback. All existing tests remain intact. Normal tests
still skip real inference unless explicitly enabled with an existing model path;
they never download weights.

Milestone 7 stops here. Snapdragon deployment and additional product features
remain outside this implementation.
