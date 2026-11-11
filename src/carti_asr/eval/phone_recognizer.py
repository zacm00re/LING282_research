"""OPTIONAL: what did Carti actually pronounce?

Runs a phone-level recognizer (wav2vec2 fine-tuned to output espeak IPA,
facebook/wav2vec2-lv-60-espeak-cv-ft) on each clip and compares its phone
string with the *canonical* phones of the reference lyric (via phonemizer).

The resulting "pronunciation distance" per style is an acoustic, model-based
estimate of how far each affectation drifts from dictionary pronunciation.
If baby voice has a much larger distance than normal voice, and Whisper's WER
tracks that distance clip-by-clip, that supports "Whisper errs because the
phones really are different" (a perception/encoder story).

It is a second model with its own biases, so treat it as supporting evidence and
spot-check a sample by hand (see docs/phonetic_transcription.md).

    python -m carti_asr.eval.phone_recognizer --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jiwer
import librosa
import pandas as pd
import torch
from transformers import AutoModelForCTC, AutoProcessor

from carti_asr.config import add_common_args, load_config
from carti_asr.eval.metrics import to_phones
from carti_asr.eval.normalize import normalize

PHONE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


@torch.no_grad()
def recognize_phones(paths, sr=16000):
    proc = AutoProcessor.from_pretrained(PHONE_MODEL)
    model = AutoModelForCTC.from_pretrained(PHONE_MODEL).eval()
    out = []
    for p in paths:
        y, _ = librosa.load(p, sr=sr)
        logits = model(proc(y, sampling_rate=sr, return_tensors="pt").input_values).logits
        out.append(proc.batch_decode(logits.argmax(-1))[0])
    return out


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    df = pd.read_csv(Path(cfg["paths"]["manifests_dir"]) / f"{args.split}.csv")
    df = df[df.condition == "original"]
    df["canonical_phones"] = to_phones([normalize(t) for t in df.text])
    df["heard_phones"] = recognize_phones(list(df.audio_path))
    df["pron_distance"] = [jiwer.wer(c, h) if c.strip() else None
                           for c, h in zip(df.canonical_phones, df.heard_phones)]
    out = Path(cfg["paths"]["results_dir"]) / "phones"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"pron_distance_{args.split}.csv", index=False)
    print(df.groupby("style")["pron_distance"].describe())


if __name__ == "__main__":
    main()
