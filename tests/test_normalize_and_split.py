"""Fast checks that run without audio or a GPU:  python -m pytest -q tests"""
import pandas as pd

from carti_asr.data.split import split_songs
from carti_asr.eval.normalize import normalize


def test_normalize_genius_and_adlibs():
    raw = "[Chorus: Playboi Carti]\nI'm runnin' it up (What?) gon' get it"
    assert normalize(raw) == "i'm running it up what gonna get it"
    assert normalize(raw, drop_adlibs=True) == "i'm running it up gonna get it"


def test_split_is_by_song_and_covers_everything():
    durs = pd.Series({f"s{i}": 60.0 + i for i in range(10)})
    assign = split_songs(durs, {"train": 0.6, "dev": 0.2, "test": 0.2}, seed=0)
    assert set(assign) == set(durs.index)
    assert set(assign.values()) == {"train", "dev", "test"}
    train_share = durs[[s for s, k in assign.items() if k == "train"]].sum() / durs.sum()
    assert 0.5 <= train_share <= 0.7
