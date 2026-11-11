"""Fine-tune whisper-small on one style with part of the network frozen.

Regimes (the core of the "where does Whisper err?" question):
  encoder_only  train the audio encoder, freeze the text decoder
                -> gains here mean the errors were in *hearing* the sounds
  decoder_only  train the decoder (incl. cross-attention), freeze the encoder
                -> gains here mean the sounds were encoded fine but mapped to
                   the wrong words (language-model prior, vocabulary, spelling)
  full          train everything (upper bound)
  none          no training (= zero-shot; handy for sanity checks)

Run one job per (style, regime, seed). The normal-voice runs are the control:
part of any improvement is just "learning Carti / learning rap / learning
Demucs artifacts". The effect of the affectation is the improvement on
baby/deep MINUS the improvement on normal under the same regime.

    python -m carti_asr.train.finetune --style baby --regime encoder_only --seed 13
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import librosa
import pandas as pd
import torch
from datasets import Dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

from carti_asr.config import add_common_args, load_config, seed_everything
from carti_asr.eval.metrics import word_scores
from carti_asr.eval.normalize import normalize


def freeze_for_regime(model: WhisperForConditionalGeneration, regime: str) -> None:
    enc, dec = model.model.encoder, model.model.decoder
    train_enc = regime in ("encoder_only", "full")
    train_dec = regime in ("decoder_only", "full")
    for p in enc.parameters():
        p.requires_grad = train_enc
    for p in dec.parameters():
        p.requires_grad = train_dec
    # proj_out is tied to the decoder token embeddings, so it follows the decoder.
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[{regime}] trainable params: {n_train/1e6:.1f}M / {n_all/1e6:.1f}M")


def load_split(cfg, split: str, style: str) -> Dataset:
    df = pd.read_csv(Path(cfg["paths"]["manifests_dir"]) / f"{split}.csv")
    df = df[(df["style"] == style) & (df["condition"] == "original")]
    return Dataset.from_pandas(df[["audio_path", "text"]].reset_index(drop=True))


def make_preprocess(processor, sr):
    def _prep(batch):
        audio = librosa.load(batch["audio_path"], sr=sr)[0]
        batch["input_features"] = processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
        # Train on lightly normalized text so the model is not penalized for
        # Genius punctuation/casing it can never predict from audio.
        batch["labels"] = processor.tokenizer(normalize(batch["text"])).input_ids
        return batch
    return _prep


@dataclass
class WhisperCollator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features],
                                              return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (ids[:, 0] == self.decoder_start_token_id).all():
            ids = ids[:, 1:]       # the model prepends it itself
        batch["labels"] = ids
        return batch


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--style", required=True, choices=["normal", "baby", "deep"])
    parser.add_argument("--regime", required=True, choices=["encoder_only", "decoder_only", "full", "none"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    seed = args.seed if args.seed is not None else cfg["seed"]
    seed_everything(seed)
    ft, sr = cfg["finetune"], cfg["audio"]["sample_rate"]

    processor = WhisperProcessor.from_pretrained(cfg["model"]["name"],
                                                 language=cfg["model"]["language"], task=cfg["model"]["task"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model"]["name"])
    model.generation_config.language = cfg["model"]["language"]
    model.generation_config.task = cfg["model"]["task"]
    model.generation_config.forced_decoder_ids = None
    freeze_for_regime(model, args.regime)

    prep = make_preprocess(processor, sr)
    train_ds = load_split(cfg, "train", args.style).map(prep, remove_columns=["audio_path", "text"])
    dev_ds = load_split(cfg, "dev", args.style).map(prep, remove_columns=["audio_path", "text"])

    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": word_scores([normalize(r) for r in refs], [normalize(h) for h in hyps]).wer}

    run_name = f"{args.style}_{args.regime}_s{seed}"
    out_dir = Path(cfg["paths"]["checkpoints_dir"]) / run_name
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        run_name=run_name,
        seed=seed,
        per_device_train_batch_size=ft["per_device_batch_size"],
        per_device_eval_batch_size=ft["per_device_batch_size"],
        gradient_accumulation_steps=ft["gradient_accumulation_steps"],
        learning_rate=float(ft["learning_rate"]),
        warmup_steps=ft["warmup_steps"],
        max_steps=ft["max_steps"],
        eval_strategy="steps",
        eval_steps=ft["eval_steps"],
        save_strategy="steps",
        save_steps=ft["eval_steps"],
        save_total_limit=1,
        load_best_model_at_end=True,       # early stopping on dev WER: important with ~4 min of train audio
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=bool(ft["fp16"]) and torch.cuda.is_available(),
        logging_steps=10,
        report_to=["tensorboard"],
        logging_dir=str(Path("runs") / run_name),
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=WhisperCollator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics, processing_class=processor,
    )
    if args.regime != "none":
        trainer.train()
    trainer.save_model(str(out_dir / "best"))
    processor.save_pretrained(str(out_dir / "best"))
    print(f"Saved best checkpoint to {out_dir/'best'}")
    print(f"Next: python -m carti_asr.eval.transcribe --split test --model {out_dir/'best'} --tag {run_name}")


if __name__ == "__main__":
    main()
