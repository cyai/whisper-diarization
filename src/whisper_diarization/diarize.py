# File: src/whisper_diarization/diarize.py

import os
import re
import torch
import torchaudio
import faster_whisper

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from .helpers import (
    cleanup,
    create_config,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

mtypes = {"cpu": "int8", "cuda": "float16"}

def diarize(
    audio_path: str,
    whisper_model: str = "medium.en",
    device: str = None,
    batch_size: int = 8,
    language: str = None,
    do_stemming: bool = True,
    suppress_numerals: bool = False,
    output_dir: str = None,
):
    """
    Runs full Whisper→ForcedAlign→NeMo diarization pipeline on `audio_path`.
    Returns a list of dicts: [{"start": ms, "end": ms, "speaker": int, "text": str}, ...].
    If output_dir is given, also writes .txt and .srt there.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    language = process_language_arg(language, whisper_model)

    # 1) Optional source separation
    if do_stemming:
        cmd = f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" ' \
              f'-o temp_outputs --device "{device}"'
        if os.system(cmd) == 0:
            base = os.path.splitext(os.path.basename(audio_path))[0]
            audio_to_process = os.path.join("temp_outputs", "htdemucs", base, "vocals.wav")
        else:
            audio_to_process = audio_path
    else:
        audio_to_process = audio_path

    # 2) Whisper ASR
    wm = faster_whisper.WhisperModel(whisper_model,
                                     device=device,
                                     compute_type=mtypes[device])
    wp = faster_whisper.BatchedInferencePipeline(wm)
    waveform = faster_whisper.decode_audio(audio_to_process)
    suppress_tokens = (
        find_numeral_symbol_tokens(wm.hf_tokenizer)
        if suppress_numerals else [-1]
    )
    if batch_size > 0:
        segments, info = wp.transcribe(
            waveform, language,
            suppress_tokens=suppress_tokens,
            batch_size=batch_size,
        )
    else:
        segments, info = wm.transcribe(
            waveform, language,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )

    full_text = "".join(seg.text for seg in segments)

    # free ASR resources
    del wm, wp
    torch.cuda.empty_cache()

    # 3) Forced Alignment
    align_model, align_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    emissions, stride = generate_emissions(
        align_model,
        torch.from_numpy(waveform).to(align_model.dtype).to(device),
        batch_size=batch_size,
    )
    del align_model
    torch.cuda.empty_cache()

    tokens_star, text_star = preprocess_text(
        full_text, romanize=True,
        language=langs_to_iso[info.language],
    )
    aln_segments, scores, blank = get_alignments(
        emissions, tokens_star, align_tokenizer
    )
    spans = get_spans(tokens_star, aln_segments, blank)
    word_timestamps = postprocess_results(text_star, spans, stride, scores)

    # 4) NeMo MSDD diarization requires a mono WAV
    tmpdir = "temp_outputs"
    os.makedirs(tmpdir, exist_ok=True)
    mono_path = os.path.join(tmpdir, "mono_file.wav")
    torchaudio.save(
        mono_path,
        torch.from_numpy(waveform).unsqueeze(0).float(),
        16000, channels_first=True
    )

    diar_model = NeuralDiarizer(cfg=create_config(tmpdir)).to(device)
    diar_model.diarize()
    del diar_model
    torch.cuda.empty_cache()

    # 5) Read RTTM → build speaker intervals
    rttm_path = os.path.join(tmpdir, "pred_rttms", "mono_file.rttm")
    speaker_intervals = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.split()
            start_ms = int(float(parts[5]) * 1000)
            dur_ms   = int(float(parts[8]) * 1000)
            spk_id   = int(parts[11].split("_")[-1])
            speaker_intervals.append([start_ms, start_ms + dur_ms, spk_id])

    # 6) Map words → speakers
    wsm = get_words_speaker_mapping(word_timestamps, speaker_intervals, "start")

    # 7) Optional punctuation
    if info.language in punct_model_langs:
        punct = PunctuationModel(model="kredor/punctuate-all")
        words = [w["word"] for w in wsm]
        labels = punct.predict(words, chunk_size=230)
        # apply only sentence-ending punctuation
        for wd, (lbl, mark) in zip(wsm, labels):
            if mark in ".?!":
                wd["word"] = wd["word"].rstrip(".") + mark

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_intervals)

    # 8) Build final list
    result = []
    for sent, spk in ssm:
        result.append({
            "start_ms": int(sent["start"] * 1000),
            "end_ms":   int(sent["end"]   * 1000),
            "speaker":  spk,
            "text":     sent["text"]
        })

    # 9) Optionally write .txt and .srt
    if output_dir:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        txt_path = os.path.join(output_dir, f"{base}.txt")
        srt_path = os.path.join(output_dir, f"{base}.srt")
        os.makedirs(output_dir, exist_ok=True)
        with open(txt_path, "w", encoding="utf-8-sig") as f:
            get_speaker_aware_transcript(ssm, f)
        with open(srt_path, "w", encoding="utf-8-sig") as srt:
            write_srt(ssm, srt)

    # 10) Cleanup temp files
    cleanup(tmpdir)

    return result
