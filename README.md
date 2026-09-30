# StageGuide

StageGuide helps you rehearse presentations on your computer.
It compares slide text with your speech and shows claims that need evidence.

**Presentation → Rehearsal audio → Review → Judge questions**

## Functions

- Read PPTX and PDF files.
- Convert local audio to text with timestamps.
- Show slide content that the speech explains, partly explains, or does not explain.
- Identify claims that need evidence, clarification, or validation.
- Ask judge questions from these findings.
- Use an optional local language model to change question wording and ask one follow-up question.

Fixed rules produce the findings and the original judge questions.
The language model cannot replace these findings.
If model output fails validation, StageGuide returns the original question.

## Current status

The current version includes Milestones 1–7.
The development system is a Mac with an Apple M4 and 16 GiB of memory.

| Component | Model | Execution |
|---|---|---|
| Speech | Whisper tiny.en | CPU |
| Local AI Judge | Qwen3-4B-Instruct-2507, Q4_K_M | llama.cpp with Metal on the verified Mac |
| Qualcomm backends | Planned | Not available |

The app shows the actual backend status.
It does not use cloud inference, a database, or a camera.
It does not detect slide changes automatically.
The demo compares the complete transcript with each slide.

## Install on macOS

Use Python 3.9 or later.
Run these commands from the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[demo,speech-cpu,test]'
```

If the speech model is absent, download its files:

```sh
hf download Systran/faster-whisper-tiny.en \
  model.bin config.json tokenizer.json vocabulary.txt \
  --local-dir models/whisper-tiny.en
```

This command needs an internet connection.
The app uses local files and does not download models automatically.

## Start the app

```sh
source .venv/bin/activate
streamlit run app.py
```

1. Open [StageGuide](http://127.0.0.1:8501).
2. Upload a PPTX or PDF file.
3. Upload a rehearsal audio file. Use WAV for the first test.
4. Select **Transcribe audio**.
5. Open **Review**.
6. Select **Analyze rehearsal**.
7. Open **Face the Judge**.
8. Select a question.
9. Enter your answer.
10. Select **Submit answer**.

For the supplied example, select **Sample demo** in the Upload tab.
Then select **Run complete sample demo**.
The sample uses a local PDF and a synthetic voice recording.

To stop the app, press **Ctrl+C** in its terminal.

## Enable the local AI Judge

Without a configured model, StageGuide uses the original judge questions.
It does not show a generated follow-up when validation fails or the model is unavailable.

1. If Xcode Command Line Tools are absent, install them.
2. Install the optional runtime:

   ```sh
   CMAKE_ARGS='-DGGML_METAL=on' CMAKE_BUILD_PARALLEL_LEVEL=4 \
     python -m pip install -e '.[judge-local]' --no-cache-dir
   ```

3. Download the GGUF file from the URL in [model details](sample_data/judge_model_provenance.json).
4. Compare its SHA-256 checksum with the value in that file.
5. Put the model in `models/qwen3-4b-instruct-2507/`.
6. Set the model path:

   ```sh
   export STAGEGUIDE_JUDGE_MODEL="$PWD/models/qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
   ```

7. Restart the app.

The verified model file is approximately 2.50 GB and uses the Apache 2.0 license.
Its source, revision, and checksum are in the model details file.
The default setting selects Metal when the runtime supports it on Apple Silicon.
The app reports Metal acceleration only after the runtime confirms model offload.
Set `STAGEGUIDE_JUDGE_ACCELERATOR=cpu` to use the CPU.

## Run tests

Run the tests without model inference:

```sh
python -m pytest -q
```

With the judge model path set, include the real-model integration test:

```sh
STAGEGUIDE_RUN_LOCAL_JUDGE=1 python -m pytest -q
```

Last results: **451 passed, 1 skipped** without model inference; **452 passed** with model inference.
Tests do not download model files.

See the [recorded model output](sample_data/real_local_judge_demo.json) for the questions, source findings, and measured times.
See the [UI test results](sample_data/real_judge_ui_demo.json) for the complete sample workflow.

## Project files

| Path | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `stageguide/presentation/` | Presentation parser |
| `stageguide/speech/` | Speech backends and transcript data |
| `stageguide/alignment/` | Slide and speech comparison |
| `stageguide/arguments/` | Argument findings |
| `stageguide/judge/` | Question planner, model backends, and validation |
| `stageguide/demo/` | Connections between the interface and backend modules |
| `tests/` | Automated tests |
| `sample_data/` | Demo files and recorded results |

## Upload with Git

Commit the source files, tests, documentation, configuration, and demo files.
The [`.gitignore`](.gitignore) file excludes environments, installed packages, model weights, secrets, caches, and build files.

Do not commit credentials in `.env.example` or other example files.
Use Git for uploads. Manual folder and ZIP uploads do not apply `.gitignore`.
