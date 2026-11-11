# Manual phonetic transcription protocol

Used for the sampled error clips written by `analysis/error_analysis.py`
(`results/zeroshot/manual_review_<style>.csv`).

- **Level of detail:** broad (phonemic) transcription for every word, plus narrow
  diacritics only for features that plausibly explain the error and that differ
  between styles. A short list keeps it consistent:
  - fronting / lisping of sibilants: [s̪], [θ] for /s/
  - vowel quality shifts: note the vowel actually produced
  - nasalization: [ ̃ ]
  - creaky voice (deep voice): [ ̰ ]
  - deletion of codas / whole syllables: mark the canonical segment in parentheses
  - devoicing: [ ̥ ]
- **There is no single gold standard.** Narrow transcription is a judgment.
  Report agreement instead: re-transcribe 20% of the sample a week later (or have a
  second person do it) and report percent agreement on the features above.
- Tag each reviewed error with one cause: `phonetic` (the sounds are genuinely
  different), `masking` (beat/ad-lib/Demucs artifact), `lexical` (sounds are clear
  but the word is rare or slang), or `label` (my reference is wrong).
