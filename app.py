"""Run from the repository root: streamlit run app.py."""

import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path

import streamlit as st

from stageguide.arguments import Severity
from stageguide.demo.controller import DemoController, DemoError, generated_follow_up
from stageguide.judge import JudgeConfig, create_judge_backend
from stageguide.speech import BackendConfig, create_backend
from stageguide.speech.transcriber import SUPPORTED_FORMATS


ROOT = Path(__file__).resolve().parent
SAMPLE_DECK = ROOT / "sample_data/demo_presentation.pdf"
SAMPLE_AUDIO = ROOT / "sample_data/speech_demo.wav"


def label(value):
    return str(value.value if hasattr(value, "value") else value).replace("_", " ").title()


def initial_configs():
    return (
        BackendConfig(backend=os.getenv("STAGEGUIDE_SPEECH_BACKEND", "cpu_whisper"),
                      model_path=os.getenv("STAGEGUIDE_SPEECH_MODEL", str(ROOT / "models/whisper-tiny.en"))),
        JudgeConfig(backend=os.getenv("STAGEGUIDE_JUDGE_BACKEND", "local"),
                    model_path=os.getenv("STAGEGUIDE_JUDGE_MODEL") or None,
                    acceleration=os.getenv("STAGEGUIDE_JUDGE_ACCELERATOR", "auto")),
    )


def build_speech(config):
    try:
        return create_backend(config)
    except Exception:
        return None


def controller_and_settings():
    if "demo_controller" not in st.session_state:
        speech_config, judge_config = initial_configs()
        try:
            judge = create_judge_backend(judge_config)
        except Exception:
            judge = create_judge_backend(JudgeConfig())
            st.session_state["judge_config_error"] = True
        st.session_state.demo_controller = DemoController(build_speech(speech_config), judge)
        st.session_state.speech_config = speech_config
        st.session_state.judge_config = judge_config
    controller = st.session_state.demo_controller
    speech_config = st.session_state.speech_config
    judge_config = st.session_state.judge_config
    with st.sidebar.expander("Backend settings"):
        st.caption("Local files only. Applying judge settings preserves the rehearsal review.")
        with st.form("backend_settings"):
            speech_options = ["cpu_whisper", "qnn_whisper"]
            speech_choice = st.selectbox("Speech backend", speech_options,
                                        index=speech_options.index(speech_config.backend)
                                        if speech_config.backend in speech_options else 0,
                                        format_func=lambda name: name + (" (planned)" if name == "qnn_whisper" else ""))
            speech_path = st.text_input("Speech model directory", str(speech_config.model_path))
            judge_options = ["local", "disabled", "qualcomm"]
            judge_choice = st.selectbox("Judge backend", judge_options,
                                       index=judge_options.index(judge_config.backend)
                                       if judge_config.backend in judge_options else 0,
                                       format_func=lambda name: name + (" (planned)" if name == "qualcomm" else ""))
            judge_path = st.text_input("Judge GGUF file (optional)", str(judge_config.model_path or ""))
            judge_model = st.text_input("Judge model label", judge_config.model)
            apply_settings = st.form_submit_button("Apply backends", use_container_width=True)
        if apply_settings:
            new_speech = BackendConfig(backend=speech_choice, model_path=speech_path)
            new_judge = JudgeConfig(backend=judge_choice, model=judge_model, model_path=judge_path or None,
                                    acceleration=judge_config.acceleration)
            if new_speech != speech_config:
                controller.set_speech_backend(build_speech(new_speech))
            controller.set_judge_backend(create_judge_backend(new_judge))
            st.session_state.speech_config = new_speech
            st.session_state.judge_config = new_judge
            st.session_state.judge_config_error = False
            st.success("Backend settings applied.")
    return controller


def render_status(controller, container):
    metadata = controller.status()
    with container:
        st.subheader("On-Device Status")
        st.caption("Local files · local processing")
        for name in ("speech", "judge"):
            with st.container(border=True):
                st.markdown(f"**{name.title()}**")
                info = metadata[name]
                if info is None:
                    st.warning("Backend unavailable. Check settings below.")
                    continue
                st.text(info["model_name"])
                st.caption(f'{info["execution_provider"]} · {info["device_type"]}')
                if name == "speech":
                    if controller.speech_failure:
                        st.caption("Last transcription failed.")
                    elif controller.transcript is not None:
                        st.caption("Local transcription completed this session.")
                    else:
                        st.caption("Configured; run transcription to verify.")
                elif not info.get("available", False):
                    st.caption(f'{label(info.get("status", "unavailable"))} · deterministic fallback active')
                else:
                    st.caption(label(info["status"]) + " · grounded validation enabled")
                with st.expander(f"{name.title()} details"):
                    for key in ("backend_name", "model_name", "execution_provider", "device_type", "accelerator"):
                        st.text(f"{label(key)}: {info[key]}")
                    st.text(f'Hardware accelerated: {"Yes" if info["is_hardware_accelerated"] else "No"}')
        if st.session_state.get("judge_config_error"):
            st.warning("Judge configuration unavailable; deterministic mode is active.")
        st.caption("No automatic model downloads. Execution details come from the selected backends.")


def sync_presentation(controller, filename, data):
    fingerprint = (filename, hashlib.sha256(data).hexdigest()) if data is not None else None
    if fingerprint != st.session_state.get("presentation_input"):
        st.session_state.presentation_input = fingerprint
        st.session_state.pop("audio_input", None)
        st.session_state.pop("presentation_error", None)
        controller.clear_presentation()
        if data is not None:
            try:
                controller.load_presentation(filename, data)
            except DemoError as exc:
                st.session_state.presentation_error = str(exc)
    if st.session_state.get("presentation_error"):
        st.error(st.session_state.presentation_error)


def upload_section(controller):
    st.subheader("Start with your presentation")
    st.caption("Bring the deck you plan to present. Slide boundaries stay intact.")
    source = st.radio("Input source", ["Your files", "Sample demo"], horizontal=True)
    if st.session_state.get("input_source") != source:
        controller.clear_presentation()
        for key in ("presentation_input", "audio_input", "presentation_error"):
            st.session_state.pop(key, None)
        st.session_state.input_source = source
    if source == "Sample demo":
        st.info("A two-page illustrative deck and an existing synthetic voice recording. All results are computed locally.")
        if not SAMPLE_DECK.is_file() or not SAMPLE_AUDIO.is_file():
            st.warning("Sample files are missing. Use Your files instead.")
            return source
        sync_presentation(controller, SAMPLE_DECK.name, SAMPLE_DECK.read_bytes())
        st.download_button("Download sample deck", SAMPLE_DECK.read_bytes(), SAMPLE_DECK.name, "application/pdf")
        if st.button("Run complete sample demo", type="primary", disabled=not controller.pages):
            try:
                with st.spinner("Transcribing locally, then reviewing the presentation…"):
                    controller.transcribe(SAMPLE_AUDIO.name, SAMPLE_AUDIO.read_bytes())
                    controller.analyze()
                st.success("Demo complete. Open Rehearse, Review and Face the Judge to explore the results.")
            except DemoError as exc:
                st.error(str(exc))
    else:
        upload = st.file_uploader("Presentation", type=["pptx", "pdf"], help="PPTX or text-based PDF, up to 50 MB.")
        sync_presentation(controller, upload.name if upload else "", upload.getvalue() if upload else None)
    if controller.pages:
        left, right = st.columns([3, 1])
        left.text(controller.filename)
        right.metric("Slides / pages", len(controller.pages))
        for page in controller.pages:
            with st.expander(f"Slide {page.page_number}", expanded=len(controller.pages) <= 2):
                st.text(page.title or "Untitled slide")
                if page.body_text:
                    st.text(page.body_text[:700] + ("…" if len(page.body_text) > 700 else ""))
                else:
                    st.caption("No body text extracted. Image-only content is not OCR-processed.")
    else:
        st.caption("Upload a presentation to unlock the rehearsal workflow.")
    return source


def rehearsal_section(controller, source):
    st.subheader("Rehearse in your own words")
    st.caption("Upload audio for local transcription. WAV is the recommended demo format.")
    audio_name, audio_data = "", None
    if source == "Sample demo" and SAMPLE_AUDIO.is_file():
        audio_name, audio_data = SAMPLE_AUDIO.name, SAMPLE_AUDIO.read_bytes()
        st.caption("Sample: synthetic voice recording, approximately 15 seconds.")
    else:
        audio = st.file_uploader("Rehearsal audio", type=sorted(ext[1:] for ext in SUPPORTED_FORMATS))
        if audio is not None:
            audio_name, audio_data = audio.name, audio.getvalue()
        fingerprint = (audio_name, hashlib.sha256(audio_data).hexdigest()) if audio_data is not None else None
        if fingerprint != st.session_state.get("audio_input"):
            controller.clear_rehearsal()
            st.session_state.audio_input = fingerprint
    if audio_data:
        st.audio(audio_data, format=mimetypes.guess_type(audio_name)[0] or "audio/wav")
    if not controller.pages:
        st.info("Upload a presentation first.")
    elif not audio_data:
        st.info("Choose an audio file to begin.")
    if st.button("Transcribe audio", type="primary", disabled=not controller.pages or not audio_data):
        try:
            with st.spinner("Running local speech transcription…"):
                controller.transcribe(audio_name, audio_data)
        except DemoError as exc:
            st.error(str(exc))
    result = controller.transcript
    if result is None:
        return
    if not result.full_transcript.strip():
        st.warning("No speech was detected. Try a clearer recording.")
        return
    columns = st.columns(3)
    columns[0].metric("Speaking duration", f"{result.duration_seconds:.1f} s")
    columns[1].metric("Words", result.word_count)
    columns[2].metric("Words per minute", result.words_per_minute)
    st.caption(f"Recording duration: {result.audio_duration_seconds:.1f} s. Speaking duration and pace come from the speech module.")
    with st.container(border=True):
        st.markdown("**Transcript**")
        st.text(result.full_transcript)
    st.markdown("**Filler words**")
    st.text(", ".join(f"{word}: {count}" for word, count in result.filler_words.items()) or "No configured filler words detected.")
    with st.expander("Timestamped segments"):
        for segment in result.segments:
            st.text(f"{segment.start:.1f}–{segment.end:.1f} s  {segment.text}")


def review_section(controller):
    st.subheader("See what your rehearsal covered")
    st.caption("The full rehearsal is compared with each slide. No automatic slide timing or slide-change detection.")
    ready = controller.transcript is not None and bool(controller.transcript.full_transcript.strip())
    if st.button("Analyze rehearsal", type="primary", disabled=not controller.pages or not ready):
        try:
            with st.spinner("Checking coverage and argument support…"):
                controller.analyze()
        except DemoError as exc:
            st.error(str(exc))
    if controller.review is None:
        st.info("Transcribe your rehearsal, then run the analysis.")
        return
    review = controller.review
    st.markdown("### Presentation coverage")
    for slide in review.coverage:
        with st.container(border=True):
            st.text(f"Slide {slide.slide_number} · {slide.title or 'Untitled slide'}")
            st.progress(slide.coverage_score, text=f"Coverage {slide.coverage_score:.0%}")
            for column, title, items in zip(st.columns(3),
                                           ["Explained", "Partially explained", "Missing"],
                                           [slide.explained_items, slide.partially_explained_items, slide.missing_items]):
                with column:
                    st.markdown(f"**{title}**")
                    for item in items:
                        st.text("• " + item)
                    if not items:
                        st.caption("None")
    st.markdown("### Argument review")
    st.caption("These findings identify opportunities for explanation or evidence. They do not establish that a claim is false.")
    count = sum(len(slide.findings) for slide in review.arguments)
    if not count:
        st.success("No argument weaknesses were flagged by the current deterministic checks.")
    for severity in Severity:
        matches = [(slide.slide_number, finding) for slide in review.arguments
                   for finding in slide.findings if finding.severity == severity]
        if not matches:
            continue
        st.markdown(f"**{severity.value} · {len(matches)} findings**")
        for number, finding in matches:
            with st.container(border=True):
                st.markdown(f"**{label(finding.type)}** · Slide {number}")
                st.text(finding.claim)
                st.caption(finding.reason)
                with st.expander("Evidence and reason"):
                    st.text("Slide: " + finding.evidence.slide_text)
                    st.text("Transcript match: " + (finding.evidence.transcript_match or "No related match"))
                    for excerpt in finding.evidence.supporting_excerpts:
                        st.text("Support attempt: " + excerpt)
                    st.caption(f"Rule: {finding.evidence.rule_id} · Detection confidence: {finding.confidence}")
    st.download_button("Download review JSON", json.dumps(controller.report(), indent=2, ensure_ascii=False),
                       "stageguide-review.json", "application/json")


def judge_section(controller):
    st.subheader("Face the Judge")
    st.caption("Practice answering the strongest grounded questions from your review.")
    if controller.review is None:
        st.info("Complete the review to prepare judge questions.")
        return
    questions = controller.review.questions
    if not questions:
        st.success("No judge questions to ask: the current review produced no findings.")
        return
    by_id = {question.question_id: question for question in questions}
    question_id = st.selectbox("Choose a question", list(by_id),
                               format_func=lambda key: f"{by_id[key].priority.value} · Slide {by_id[key].slide_number} · {by_id[key].question}",
                               key=f"question-{controller.revision}")
    with st.spinner("Preparing a grounded question…"):
        refined = controller.refine(question_id)
    if refined.used_fallback:
        st.caption("Deterministic Judge Mode · local LLM unavailable or refinement not validated")
    else:
        st.caption("Local AI Judge · refinement passed grounding checks")
    with st.chat_message("assistant"):
        st.markdown("**Judge**")
        st.text(refined.question)
    with st.expander("Why this question?"):
        original = by_id[question_id]
        st.text("Source claim: " + original.claim)
        st.caption(original.reason)
        st.text(f"{original.question_id} → {original.evidence_reference.finding_id} → {original.source_finding_type.value}")
        st.json(original.to_dict(), expanded=False)
    form_key = f"answer-{controller.revision}-{question_id}"
    with st.form(form_key):
        answer = st.text_area("Your answer", placeholder="Explain your reasoning, evidence or assumptions…", height=120)
        submitted = st.form_submit_button("Submit answer", type="primary")
    if submitted:
        try:
            with st.spinner("Checking for a grounded follow-up…"):
                controller.answer(question_id, answer)
        except DemoError as exc:
            st.error(str(exc))
    interaction = controller.interactions.get(question_id)
    if interaction is not None:
        with st.chat_message("user"):
            st.markdown("**Presenter · submitted answer**")
            st.text(interaction.presenter_answer)
        follow_up = generated_follow_up(interaction)
        if follow_up:
            with st.chat_message("assistant"):
                st.markdown("**Follow-up**")
                st.text(follow_up)
        else:
            st.caption("Answer saved for this session. No validated local-model follow-up is available.")
        st.download_button("Download interaction JSON", json.dumps(interaction.to_dict(), indent=2, ensure_ascii=False),
                           "stageguide-interaction.json", "application/json")


def main():
    st.set_page_config(page_title="StageGuide · Rehearsal Studio", page_icon="◒", layout="wide")
    st.markdown("""<style>
    .block-container {max-width: 1160px; padding-top: 2.4rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #e5eaf2;}
    h1 {letter-spacing: -0.045em;}
    [data-testid="stText"] {font-family: inherit; line-height: 1.65;}
    </style>""", unsafe_allow_html=True)
    status_panel = st.sidebar.container()
    controller = controller_and_settings()
    st.caption("LOCAL REHEARSAL WORKSPACE")
    st.title("StageGuide")
    st.write("An On-Device AI Coach for Presentation Rehearsal and Argument Practice")
    st.caption("Bring your deck. Rehearse your story. Strengthen what you can defend.")
    st.divider()
    upload, rehearse, review, judge = st.tabs(["1 · Upload", "2 · Rehearse", "3 · Review", "4 · Face the Judge"])
    with upload:
        source = upload_section(controller)
    with rehearse:
        rehearsal_section(controller, source)
    with review:
        review_section(controller)
    with judge:
        judge_section(controller)
    render_status(controller, status_panel)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger(__name__).exception("StageGuide demo UI failure")
        st.error("This action could not be completed. Reload the page or check your local backend settings.")
