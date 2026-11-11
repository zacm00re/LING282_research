# All code is tentative and just a proof-of-concept -- everything will be reviewed and refactored when experiments are run

# Whisper and vocal affectations: Playboi Carti's normal, baby, and deep voices

LING 282 term project.

## Research question

Playboi Carti uses three distinct vocal styles: a high-pitched **baby voice**, a
low-pitched **deep voice**, and a **normal voice** close to ordinary speech. The
affectations change pitch, but they also change articulation (lisping, slurring,
truncation). One artist is picked to keep dialect, recording, and vocal mixing as constant as possible and the vocal style is the only independent variable.

> What about each vocal affectation makes Whisper's transcription worse, and where
> in the model (encoder/decorder) do those errors happen?

## Methodology

| Step | What | Why |
|---|---|---|
| Data | Songs where one style is used throughout; vocals isolated with Demucs; lines labeled in Audacity with the sung lyric | Clean, aligned (audio, text) pairs per style |
| Split | 60/20/20 train/dev/test **by song** within each style | No lines from a test song leak into training |
| H1: zero-shot | `openai/whisper-small` WER, CER, PER on each style | Does each affectation hurt, and by how much? |
| Pitch control | Normal-voice test clips shifted to the median F0 of baby / deep voice with **formant-preserving** rubberband | If shifted-normal WER is close to normal WER, pitch alone is not the problem |
| H2: where | Fine-tune **encoder only** vs **decoder only**, 3 seeds each, per style; normal voice is the control | Encoder gains = acoustic/phonetic errors; decoder gains = word-choice/language-prior errors |
| Error analysis | Sub/del/ins breakdown,  manual IPA on a sample 

Main quantities reported:

- **WER gap** for style *X*: WER(X) − WER(normal), with 95% CIs from a song-level bootstrap.
- **Pitch share** of the gap: (WER(normal shifted to X) − WER(normal)) / (WER(X) − WER(normal)).
- **Affect-specific gain** for regime *R*: gain(X, R) − gain(normal, R), so that generic
  "learning Carti / learning rap mixes" improvements are subtracted out.

## Repository structure

```
configs/default.yaml        all paths, model, training and pitch settings (one place to change things)
data/
  metadata/songs.csv        song list: id, title, style, source URL, lyrics URL      [committed]
  labels/<song_id>.txt      Audacity label tracks: start, end, sung lyric           [committed]
  raw/ stems/ clips/        audio (downloaded, Demucs vocals, 16 kHz clips)         [git-ignored]
  shifted/ manifests/       pitch-shifted clips, CSV manifests                      [git-ignored]
src/carti_asr/
  config.py                 config loading + seeding
  data/download.py          1. download audio listed in songs.csv (yt-dlp)
  data/separate.py          2. Demucs vocal stems
  data/segment.py           3. cut labeled spans -> 16 kHz mono clips + manifest
  data/split.py             4. song-level 60/20/20 split
  data/pitch.py             5. measure F0 per style; formant-preserving pitch shift
  eval/normalize.py         text normalization (Genius headers, ad-libs, spellings)
  eval/metrics.py           WER/CER (jiwer), PER (phonemizer + espeak-ng), error alignment
  eval/transcribe.py        run Whisper (zero-shot or fine-tuned) on a split and score it
  eval/phone_recognizer.py  optional: wav2vec2 phone recognizer to estimate actual pronunciation
  train/finetune.py         fine-tune with encoder or decoder frozen
  analysis/compare.py       hypothesis tables, bootstrap CIs, pitch share, affect-specific gain
  analysis/error_analysis.py  error types, confusions, manual-review samples
  analysis/plots.py         figures for presentation/writeup
scripts/                    shell entry points (setup, data pipeline, submit all jobs, smoke test)
jobs/                       SLURM scripts for BlueHive (Demucs, zero-shot, fine-tune array, analysis)
results/                    small CSV summaries + figures [committed]; per-clip predictions [ignored]
tests/                      fast unit tests (no audio/GPU needed)
docs/                       labeling + phonetic transcription notes
```