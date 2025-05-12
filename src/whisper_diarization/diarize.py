import logging
import os
import re

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
    write_srt,
)

def diarize(
    audio_path: str, 
    language: str = None, 
    whisper_model: str = "medium.en", 
    batch_size: int = 8, 
    device: str = None, 
    stemming: bool = True, 
    suppress_numerals: bool = False
):
    """
    Perform diarization and transcription on an audio file.
    
    Args:
        audio_path (str): Path to the input audio file.
        language (str, optional): Language spoken in the audio. 
            Defaults to None (auto-detect).
        whisper_model (str, optional): Name of the Whisper model to use. 
            Defaults to "medium.en".
        batch_size (int, optional): Batch size for batched inference. 
            Defaults to 8. Set to 0 for original whisper longform inference.
        device (str, optional): Device to use for processing. 
            Defaults to cuda if available, else cpu.
        stemming (bool, optional): Whether to perform source separation. 
            Defaults to True.
        suppress_numerals (bool, optional): Whether to suppress numerical digits. 
            Defaults to False.
    
    Returns:
        tuple: Paths to generated transcript (.txt) and subtitle (.srt) files
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model type mapping
    mtypes = {"cpu": "int8", "cuda": "float16"}
    
    # Process language argument
    language = process_language_arg(language, whisper_model)
    
    # Stem audio if required
    if stemming:
        return_code = os.system(
            f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o temp_outputs --device "{device}"'
        )
        
        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. "
                "Use stemming=False to disable it."
            )
            vocal_target = audio_path
        else:
            vocal_target = os.path.join(
                "temp_outputs",
                "htdemucs",
                os.path.splitext(os.path.basename(audio_path))[0],
                "vocals.wav",
            )
    else:
        vocal_target = audio_path
    
    # Transcribe the audio file
    whisper_model_obj = faster_whisper.WhisperModel(
        whisper_model, device=device, compute_type=mtypes[device]
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model_obj)
    audio_waveform = faster_whisper.decode_audio(vocal_target)
    
    # Suppress tokens for numerals if required
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model_obj.hf_tokenizer)
        if suppress_numerals
        else [-1]
    )
    
    # Transcribe
    if batch_size > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            batch_size=batch_size,
        )
    else:
        transcript_segments, info = whisper_model_obj.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )
    
    full_transcript = "".join(segment.text for segment in transcript_segments)
    
    # Clear GPU VRAM
    del whisper_model_obj, whisper_pipeline
    torch.cuda.empty_cache()
    
    # Forced Alignment
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    emissions, stride = generate_emissions(
        alignment_model,
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device),
        batch_size=batch_size,
    )
    
    del alignment_model
    torch.cuda.empty_cache()
    
    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[info.language],
    )
    
    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )
    
    spans = get_spans(tokens_starred, segments, blank_token)
    
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    
    # Convert audio to mono for NeMo compatibility
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    torchaudio.save(
        os.path.join(temp_path, "mono_file.wav"),
        torch.from_numpy(audio_waveform).unsqueeze(0).float(),
        16000,
        channels_first=True,
    )
    
    # Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(device)
    msdd_model.diarize()
    
    del msdd_model
    torch.cuda.empty_cache()
    
    # Reading timestamps <> Speaker Labels mapping
    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])
    
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    
    if info.language in punct_model_langs:
        # Restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")
        
        words_list = list(map(lambda x: x["word"], wsm))
        
        labled_words = punct_model.predict(words_list, chunk_size=230)
        
        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        
        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)
        
        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word
    
    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language."
            " Using the original punctuation."
        )
    
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
    
    # Generate output files
    txt_path = f"{os.path.splitext(audio_path)[0]}.txt"
    srt_path = f"{os.path.splitext(audio_path)[0]}.srt"
    
    with open(txt_path, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)
    
    with open(srt_path, "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)
    
    # Cleanup temporary files
    cleanup(temp_path)
    
    return txt_path, srt_path

# Example usage (commented out)
# if __name__ == "__main__":
#     txt_file, srt_file = diarize("/path/to/your/audio/file.wav")
#     print(f"Generated transcript: {txt_file}")
#     print(f"Generated subtitles: {srt_file}")