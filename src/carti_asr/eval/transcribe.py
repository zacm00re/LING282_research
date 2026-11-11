"""Run Whisper on a manifest and score it.

Zero-shot:
    python -m carti_asr.eval.transcribe --split test --tag zeroshot
Fine-tuned checkpoint:
    python -m carti_asr.eval.transcribe --split test --model checkpoints/baby_encoder_only_s13 --tag baby_enc_s13

Writes:
    results/<tag>/predictions_<split>.jsonl   one line per clip (ref, hyp, normalized forms)  [git-ignored]
    results/<tag>/scores_<split>.csv          WER/CER/PER per style x condition                [committed]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from carti_asr.config import add_common_args, load_config, seed_everything
from carti_asr.eval.metrics import phone_error_rate, word_scores
from carti_asr.eval.normalize import normalize


@torch.no_grad()
def transcribe(paths: list[str], model, processor, sr: int, num_beams: int, batch_size: int = 16):
    device = model.device
    hyps = []
    for i in range(0, len(paths), batch_size):
        audio = [librosa.load(p, sr=sr)[0] for p in paths[i:i + batch_size]]
        feats = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device, model.dtype)
        ids = model.generate(feats, language="en", task="transcribe", num_beams=num_beams)
        hyps += processor.batch_decode(ids, skip_special_tokens=True)
    return [h.strip() for h in hyps]


def score_table(df: pd.DataFrame, with_per: bool = True) -> pd.DataFrame:
    rows = []
    for (style, cond), g in df.groupby(["style", "condition"]):
        s = word_scores(list(g.ref_norm), list(g.hyp_norm)).as_dict()
        if with_per:
            s["per"] = phone_error_rate(list(g.ref_norm), list(g.hyp_norm))
        rows.append(dict(style=style, condition=cond, n_clips=len(g), **s))
    return pd.DataFrame(rows)


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--model", default=None, help="HF id or local checkpoint; defaults to config")
    parser.add_argument("--tag", required=True, help="results subfolder name")
    parser.add_argument("--drop-adlibs", action="store_true")
    parser.add_argument("--no-per", action="store_true", help="skip PER (if espeak-ng is missing)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])

    model_id = args.model or cfg["model"]["name"]
    processor = WhisperProcessor.from_pretrained(cfg["model"]["name"])
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    df = pd.read_csv(Path(cfg["paths"]["manifests_dir"]) / f"{args.split}.csv")
    df["hyp"] = transcribe(list(df.audio_path), model, processor,
                           cfg["audio"]["sample_rate"], cfg["eval"]["num_beams"])
    df["ref_norm"] = df.text.map(lambda t: normalize(t, args.drop_adlibs))
    df["hyp_norm"] = df.hyp.map(lambda t: normalize(t, args.drop_adlibs))

    out = Path(cfg["paths"]["results_dir"]) / args.tag; out.mkdir(parents=True, exist_ok=True)
    with open(out / f"predictions_{args.split}.jsonl", "w") as f:
        for r in df.to_dict("records"):
            f.write(json.dumps(r) + "\n")
    scores = score_table(df, with_per=not args.no_per)
    scores.to_csv(out / f"scores_{args.split}.csv", index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
