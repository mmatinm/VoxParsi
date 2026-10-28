"""VoxParsi — Persian speech-to-text."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from voxparsi import audio as audio_io
from voxparsi import correct as llm_correct
from voxparsi import dataset, experiment, metrics, report, stt

st.set_page_config(page_title="VoxParsi", page_icon="🎙️", layout="wide")

UPLOAD_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "webm"]

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1180px; }
      .rtl {
        direction: rtl; text-align: right;
        font-size: 1.08rem; line-height: 2;
        font-family: "Vazirmatn", "Noto Naskh Arabic", "Tahoma", serif;
      }
      .card {
        border: 1px solid rgba(128,128,128,.2); border-radius: .6rem;
        padding: .8rem 1rem; background: rgba(128,128,128,.05);
      }
      .muted { color: rgba(128,128,128,.9); font-size: .84rem; }
      h1 { letter-spacing: -.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def stt_long_reference(speaker: str, target_seconds: float) -> tuple[Path, str, float]:
    utterances = grouped_corpus()[speaker]
    reference = audio_io.build_speaker_reference(
        [u.wav_path for u in utterances],
        target_seconds=target_seconds,
        labels=[u.utterance_id for u in utterances],
    )
    text_by_id = {u.utterance_id: u.text for u in utterances}
    text = " ".join(text_by_id[label] for label in reference.sources)
    path = materialize_upload(audio_io.to_wav_bytes(reference.samples, reference.sample_rate), ".wav")
    return path, text, reference.duration_seconds


@st.cache_resource(show_spinner=False)
def get_asr(key: str, device: str) -> stt.Transcriber:
    return stt.load_model(key, device)


def corpus() -> list[dataset.Utterance]:
    return list(dataset.load_corpus())


def grouped_corpus() -> dict[str, list[dataset.Utterance]]:
    return dataset.by_speaker()


@st.cache_data(show_spinner=False)
def materialize_upload(data: bytes, suffix: str) -> Path:
    return audio_io.write_temp_audio(data, suffix)


def rtl(text: str) -> None:
    st.markdown(f'<div class="card rtl">{text}</div>', unsafe_allow_html=True)


with st.sidebar:
    st.header("Settings")
    device = stt.pick_device()
    st.caption(f"ASR device: **{device}**")

    specs = stt.available_models()
    spec = st.selectbox(
        "Speech-to-text model",
        specs,
        format_func=lambda s: f"{s.label}" + ("  ·  downloaded" if s.is_cached else f"  ·  {s.download_mb} MB"),
    )
    st.caption(spec.note)

    with st.expander("Decoding options"):
        beam_size = st.slider("Beam size", 1, 8, 5)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        if temperature > 0:
            st.caption("Whisper drops beam search once temperature is above 0.")
        initial_prompt = st.text_input("Initial prompt (optional)", "")

    options = stt.TranscriptionOptions(
        beam_size=beam_size,
        temperature=temperature,
        initial_prompt=initial_prompt or None,
    )

    normalize_scoring = st.toggle(
        "Normalise before scoring",
        value=True,
        help="Fold Persian/Arabic spelling variants and punctuation before measuring WER/CER.",
    )

    st.divider()
    st.subheader("LLM post-correction")
    openrouter_key = st.text_input(
        "OpenRouter API key",
        type="password",
        help="Used only for this session.",
    ).strip()
    st.markdown("[Get an OpenRouter API key](https://openrouter.ai/keys)")
    effective_api_key = openrouter_key or os.environ.get(llm_correct.API_KEY_ENV, "")
    configured = llm_correct.is_configured(effective_api_key)
    use_correction = st.toggle(
        "Correct transcripts with an LLM",
        value=False,
        disabled=not configured,
        help="Send each transcript to OpenRouter for a conservative proofreading pass.",
    )
    correction_model = st.text_input(
        "OpenRouter model",
        llm_correct.DEFAULT_MODEL,
        disabled=not configured,
    )
    if configured:
        st.caption("OpenRouter key found for this session.")
    else:
        st.caption("Add a key to enable post-correction.")

st.title("🎙️ VoxParsi")
st.caption("Persian speech-to-text with Whisper")

transcribe_tab, experiment_tab = st.tabs(["Speech-to-Text", "Experiments & Report"])

with transcribe_tab:
    source = st.radio(
        "Audio source",
        [
            "Upload a file",
            f"Pick from voxforge-fa ({len(corpus())} utterances)",
            "Build a 30–60 s speaker recording",
        ],
        horizontal=True,
        key="stt_source",
    )

    audio_path: Path | None = None
    utterance: dataset.Utterance | None = None
    long_reference_text = ""

    if source.startswith("Upload"):
        upload = st.file_uploader("Audio file", type=UPLOAD_TYPES, key="stt_upload")
        if upload is not None:
            audio_path = materialize_upload(upload.getvalue(), Path(upload.name).suffix)
    elif source.startswith("Pick"):
        left, right = st.columns(2)
        speaker = left.selectbox("Speaker", sorted(grouped_corpus()), key="stt_speaker")
        utterance = right.selectbox(
            "Utterance",
            grouped_corpus()[speaker],
            format_func=lambda u: u.utterance_id,
            key="stt_utterance",
        )
        audio_path = utterance.wav_path
    else:
        left, right = st.columns(2)
        long_speaker = left.selectbox("Speaker", sorted(grouped_corpus()), key="stt_long_speaker")
        long_target = right.slider("Target length (s)", 30, 60, int(audio_io.REFERENCE_TARGET_SEC), 5, key="stt_long_target")
        try:
            audio_path, long_reference_text, duration = stt_long_reference(long_speaker, long_target)
            st.caption(f"Concatenated reference: {duration:.1f} s")
        except Exception as error:
            st.error(f"Could not build the recording: {error}")

    if audio_path is not None:
        st.audio(str(audio_path))

    if st.button("Transcribe", type="primary", disabled=audio_path is None):
        try:
            with st.spinner(f"Loading {spec.label}…"):
                model = get_asr(spec.key, device)
            with st.spinner("Transcribing…"):
                result = stt.transcribe(model, audio_path, options)
            st.session_state["stt_text"] = result.text
            st.session_state["stt_corrected"] = None

            if use_correction:
                with st.spinner("Correcting transcript…"):
                    try:
                        st.session_state["stt_corrected"] = llm_correct.correct(
                            result.text,
                            model=correction_model,
                            api_key=effective_api_key,
                        ).corrected
                    except llm_correct.CorrectionUnavailable as error:
                        st.warning(f"Correction skipped: {error}")
        except Exception as error:
            st.error(f"Transcription failed: {error}")

    hypothesis = st.session_state.get("stt_text", "")
    corrected = st.session_state.get("stt_corrected")

    if hypothesis:
        st.subheader("Transcription")
        rtl(hypothesis)
        if corrected:
            st.markdown("**After LLM correction**")
            rtl(corrected)
        st.download_button(
            "Download transcript (.txt)",
            data=(corrected or hypothesis).encode("utf-8"),
            file_name="transcript.txt",
            mime="text/plain",
        )

    st.subheader("Evaluation")
    if source.startswith("Build") and long_reference_text:
        st.markdown("**Concatenated reference text**")
        reference = st.text_area(
            "Reference text",
            value=long_reference_text,
            height=140,
            help="The reference transcript is assembled in the same order as the concatenated audio.",
            key=f"long_ref_{long_speaker}_{long_target}",
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        reference = st.text_area(
            "Reference text",
            value=utterance.text if utterance else "",
            height=90,
            help="Ground-truth transcript.",
            key=f"ref_{utterance.label if utterance else 'upload'}",
        )
    st.caption("WER and CER are edit rates: lower is better. They compare the transcript to the reference, after optional normalization.")

    if reference.strip() and hypothesis.strip():
        word = metrics.wer(reference, hypothesis, normalize=normalize_scoring)
        char = metrics.cer(reference, hypothesis, normalize=normalize_scoring)

        columns = st.columns(4 if corrected else 2)
        columns[0].metric("Whisper WER", f"{word.rate:.1%}")
        columns[1].metric("Whisper CER", f"{char.rate:.1%}")
        if corrected:
            fixed_word = metrics.wer(reference, corrected, normalize=normalize_scoring)
            fixed_char = metrics.cer(reference, corrected, normalize=normalize_scoring)
            columns[2].metric("LLM WER", f"{fixed_word.rate:.1%}", f"{word.rate - fixed_word.rate:+.1%}")
            columns[3].metric("LLM CER", f"{fixed_char.rate:.1%}", f"{char.rate - fixed_char.rate:+.1%}")

        st.markdown(
            f'<p class="muted">Words — {word.hits} correct, {word.substitutions} substituted, '
            f"{word.deletions} deleted, {word.insertions} inserted "
            f"(reference: {word.reference_length} words).</p>",
            unsafe_allow_html=True,
        )
        if normalize_scoring:
            with st.expander("Normalised text actually compared"):
                st.markdown("**Reference**")
                rtl(metrics.normalize_text(reference))
                st.markdown("**Hypothesis**")
                rtl(metrics.normalize_text(hypothesis))
                if corrected:
                    st.markdown("**LLM output**")
                    rtl(metrics.normalize_text(corrected))
    elif hypothesis.strip():
        st.caption("Add a reference transcript to score this transcription.")

with experiment_tab:
    speakers = sorted(grouped_corpus())
    selected = st.multiselect("Speakers", speakers, default=speakers[:3], key="exp_speakers")
    use_long_reference = st.checkbox("Build 30–60 s speaker references for STT evaluation", value=False)
    target_seconds = st.slider("Reference length (s)", 30, 60, 45, 5, disabled=not use_long_reference)

    if st.button("Run experiment", disabled=not selected):
        try:
            with st.spinner("Running speech-to-text evaluation…"):
                model = get_asr(spec.key, device)
                speaker_map = {speaker: grouped_corpus()[speaker] for speaker in selected}
                utterances = [
                    u
                    for speaker in selected
                    for u in sorted(
                        speaker_map[speaker],
                        key=lambda item: (item.utterance_id.lower(), item.wav_path.name),
                    )
                ]
                if use_long_reference:
                    utterances = [
                        u
                        for speaker in selected
                        for u in sorted(
                            speaker_map[speaker],
                            key=lambda item: (item.utterance_id.lower(), item.wav_path.name),
                        )
                    ]
                run = experiment.run_stt(
                    model,
                    utterances,
                    options=options,
                    normalize=normalize_scoring,
                    use_llm_correction=use_correction,
                    correction_model=correction_model,
                    correction_api_key=effective_api_key or None,
                    long_reference_seconds=target_seconds if use_long_reference else None,
                )
                st.session_state["stt_run"] = run
        except Exception as error:
            st.error(f"Experiment failed: {error}")

    stt_run = st.session_state.get("stt_run")
    if stt_run is None:
        st.caption("Select speakers and run the STT experiment.")
        st.stop()

    st.divider()
    st.subheader("Speech-to-text results")
    columns = st.columns(4)
    columns[0].metric("Corpus WER", f"{stt_run.corpus_wer:.1%}")
    columns[1].metric("Corpus CER", f"{stt_run.corpus_cer:.1%}")
    columns[2].metric("Median WER", f"{stt_run.median_wer:.1%}")
    columns[3].metric("Utterances", len(stt_run.rows))

    if stt_run.correction_delta is not None:
        delta = stt_run.correction_delta
        st.metric(
            "Corpus WER after LLM correction",
            f"{stt_run.corrected_corpus_wer:.1%}",
            f"{delta:+.1%}",
            delta_color="normal" if delta >= 0 else "inverse",
        )
        st.caption("A negative delta means correction hurt the transcript on average.")

    st.markdown("**Per speaker** (worst first)")
    st.dataframe(
        [
            {"speaker": s, "utterances": n, "WER": f"{w:.1%}", "CER": f"{c:.1%}"}
            for s, n, w, c in stt_run.per_speaker()
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Examples (worst 5 by WER)"):
        for row in sorted(stt_run.rows, key=lambda r: -r.wer)[:5]:
            st.markdown(f"`{row.label}` — WER {row.wer:.1%} · CER {row.cer:.1%}")
            st.markdown("Reference")
            rtl(row.reference)
            st.markdown("Heard")
            rtl(row.hypothesis or "—")
            if row.corrected:
                st.markdown("Corrected")
                rtl(row.corrected)
            st.divider()

    st.divider()
    st.subheader("Download the report")
    cols = st.columns(3)
    cols[0].download_button("Report (.md)", report.markdown(stt_run).encode("utf-8"), "report.md", "text/markdown")
    cols[1].download_button("Data (.json)", report.to_json(stt_run).encode("utf-8"), "report.json", "application/json")
    cols[2].download_button("STT rows (.csv)", report.stt_csv(stt_run).encode("utf-8"), "speech_to_text.csv", "text/csv")
