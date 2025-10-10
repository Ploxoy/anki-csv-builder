import io
import json
import zipfile

import pytest

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.export_csv import generate_csv
from core.export_anki import build_anki_package, HAS_GENANKI


@pytest.fixture
def sample_cards():
    return [
        {
            "L2_word": "begrijpen",
            "L2_cloze": "Ik {{c1::begrijp}} deze zin.",
            "L1_sentence": "I understand this sentence.",
            "L2_collocations": "goed begrijpen; slecht begrijpen; beter begrijpen",
            "L2_definition": "iets volledig snappen",
            "L1_gloss": "understand",
            "L1_hint": "",
            "AudioSentence": "",
            "AudioWord": "",
        },
        {
            "L2_word": "ruimen",
            "L2_cloze": "We {{c1::ruimen}} het huis.",
            "L1_sentence": "We clean the house.",
            "L2_collocations": "opruimen; huis opruimen; spullen opruimen",
            "L2_definition": "iets netjes maken",
            "L1_gloss": "tidy",
            "L1_hint": "",
            "AudioSentence": "",
            "AudioWord": "",
        },
    ]


def test_generate_csv_basic(sample_cards):
    l1_meta = {"label": "EN", "csv_translation": "Translation", "csv_gloss": "Gloss"}
    csv_text = generate_csv(
        sample_cards,
        l1_meta,
        delimiter="|",
        line_terminator="\n",
        include_header=True,
        include_extras=True,
        anki_field_header=True,
        extras_meta={"level": "B1", "profile": "balanced", "model": "gpt", "L1": "EN"},
    )
    lines = csv_text.strip().split("\n")
    assert lines[0].startswith("L2_word|L2_cloze")
    assert len(lines) == 3
    assert "AudioSentence" in lines[0]
    assert lines[0].count("|") == 12  # 13 columns -> 12 delimiters
    assert "begrijpen" in lines[1]
    assert lines[-1].endswith("EN")


@pytest.mark.skipif(not HAS_GENANKI, reason="genanki not installed")
def test_build_anki_package(sample_cards):
    package_bytes = build_anki_package(
        sample_cards,
        l1_label="EN",
        guid_policy="stable",
        run_id="123",
        model_id=1234567890,
        model_name="Test Model",
        deck_id=987654321,
        deck_name="Test Deck",
        front_template="<div>{{cloze:L2_cloze}}</div>",
        back_template="<div>{{cloze:L2_cloze}}</div>",
        css="",
        tags_meta={"level": "B1", "profile": "balanced", "model": "gpt", "L1": "EN"},
        media_files={"sentence_sample.mp3": b"fake-bytes"},
    )

    assert isinstance(package_bytes, bytes)
    assert len(package_bytes) > 0

    with zipfile.ZipFile(io.BytesIO(package_bytes)) as zf:
        names = zf.namelist()
        assert "collection.anki2" in names
        assert "media" in names
        media_mapping = json.loads(zf.read("media").decode("utf-8"))
        assert "sentence_sample.mp3" in media_mapping.values()
