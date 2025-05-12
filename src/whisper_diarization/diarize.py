# src/whisper_diarization/diarize.py

import logging
import os
import re
import shutil
import tempfile

import faster_whisper
import torch
import torchaudio

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
    write_srt,  # this returns the SRT text if given a file-like
)

def diarize(
    audio_path: str,
    language: str = None,
    whisper_model: str = "medium.en",
    batch_size: int = 8,
    device: str = None,
    stemming: bool = True,
    suppress_numerals: bool = False,
):
    """
    Perform diarization + transcription on `audio_path` and return:
      - `plain_text`: the full speaker-aware transcript as a single string
      - `srt_text`: the SRT-format subtitles as a single string
    """

    # 1) Setup
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    language = process_language_arg(language, whisper_model)
    mtypes = {"cpu": "int8", "cuda": "float16"}

    # use a unique temp dir for all intermediates
    tmpdir = tempfile.mkdtemp(prefix="whisper_diar_temp_")

    try:
        # 2) Optional source separation
        if stemming:
            ret = os.system(
                f'python -m demucs.separate -n htdemucs --two-stems=vocals '
                f'"{audio_path}" -o "{tmpdir}" --device "{device}"'
            )
            if ret == 0:
                base = os.path.splitext(os.path.basename(audio_path))[0]
                vocal_target = os.path.join(tmpdir, "htdemucs", base, "vocals.wav")
            else:
                logging.warning("Source splitting failed; using original audio.")
                vocal_target = audio_path
        else:
            vocal_target = audio_path

        # 3) Whisper ASR
        asr_model = faster_whisper.WhisperModel(
            whisper_model, device=device, compute_type=mtypes[device]
        )
        asr_pipe = faster_whisper.BatchedInferencePipeline(asr_model)
        waveform = faster_whisper.decode_audio(vocal_target)

        suppress_tokens = (
            find_numeral_symbol_tokens(asr_model.hf_tokenizer)
            if suppress_numerals
            else [-1]
        )

        if batch_size > 0:
            segments, info = asr_pipe.transcribe(
                waveform, language,
                suppress_tokens=suppress_tokens,
                batch_size=batch_size,
            )
        else:
            segments, info = asr_model.transcribe(
                waveform, language,
                suppress_tokens=suppress_tokens,
                vad_filter=True,
            )

        full_text = "".join(seg.text for seg in segments)

        # free ASR
        del asr_model, asr_pipe
        torch.cuda.empty_cache()

        # 4) Forced alignment
        aln_model, aln_tokenizer = load_alignment_model(
            device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        emissions, stride = generate_emissions(
            aln_model,
            torch.from_numpy(waveform)
            .to(aln_model.dtype)
            .to(aln_model.device),
            batch_size=batch_size,
        )
        del aln_model
        torch.cuda.empty_cache()

        tokens_starred, text_starred = preprocess_text(
            full_text, romanize=True, language=langs_to_iso[info.language]
        )
        aln_segs, scores, blank = get_alignments(
            emissions, tokens_starred, aln_tokenizer
        )
        spans = get_spans(tokens_starred, aln_segs, blank)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        # 5) Prepare mono for NeMo diarization
        mono_path = os.path.join(tmpdir, "mono_file.wav")
        torchaudio.save(
            mono_path,
            torch.from_numpy(waveform).unsqueeze(0).float(),
            16000,
            channels_first=True,
        )

        # 6) NeMo MSDD diarization
        diar_model = NeuralDiarizer(cfg=create_config(tmpdir)).to(device)
        diar_model.diarize()
        del diar_model
        torch.cuda.empty_cache()

        # 7) Parse RTTM → speaker intervals
        rttm_file = os.path.join(tmpdir, "pred_rttms", "mono_file.rttm")
        speaker_ts = []
        with open(rttm_file) as f:
            for line in f:
                parts = line.split()
                # skip invalid entries
                if parts[5] == "<NA>":
                    continue
                start = float(parts[5]) * 1000
                end = start + float(parts[8]) * 1000
                spk = int(parts[11].split("_")[-1])
                speaker_ts.append([int(start), int(end), spk])

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        # 8) Punctuation (optional)
        if info.language in punct_model_langs:
            punct = PunctuationModel(model="kredor/punctuate-all")
            words = [w["word"] for w in wsm]
            labels = punct.predict(words, chunk_size=230)
            ending = ".?!"
            is_acro = lambda x: re.fullmatch(r"\b(?:[A-Za-z]\.){2,}", x)
            for wd, lbl in zip(wsm, labels):
                if lbl[1] in ending and not is_acro(wd["word"]):
                    wd["word"] = wd["word"].rstrip(".") + lbl[1]

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        # 9) Build outputs in memory
        # Plain text: speaker-aware transcript
        from io import StringIO
        txt_buf = StringIO()
        get_speaker_aware_transcript(ssm, txt_buf)
        plain_text = txt_buf.getvalue()

        # SRT: subtitles
        srt_buf = StringIO()
        write_srt(ssm, srt_buf)
        srt_text = srt_buf.getvalue()

        return plain_text, srt_text

    finally:
        # 10) Cleanup all temps
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
