# scripts/check_wordfreq.py
# Quick check whether 'wordfreq' is installed and how it scores words for 'nl' (Dutch) and 'en' (English).
# Usage: python scripts/check_wordfreq.py

from __future__ import annotations
import importlib
import sys

WORDS = [
    "natuurkunde", "huis", "man",       # Dutch words
    "physics", "computer", "science",   # English words
    "физика", "дом",                    # Russian words (should be low)
    "de man", "het huis",               # with Dutch articles
    "New York"                          # multi-word / proper name
]

def main():
    try:
        wf = importlib.import_module("wordfreq")
        from wordfreq import zipf_frequency  # type: ignore
        print("wordfreq available:", wf.__version__ if hasattr(wf, "__version__") else "(version unknown)")
    except Exception as e:
        print("wordfreq is NOT available in this environment.")
        print("To install: pip install wordfreq")
        sys.exit(1)

    langs = ["nl", "en"]
    print(f"\nChecking zipf_frequency for languages: {langs}\n")
    for w in WORDS:
        scores = {}
        for lang in langs:
            try:
                z = zipf_frequency(w, lang)
            except Exception:
                z = None
            scores[lang] = z
        # Print results
        print(f"{w!r:15} -> nl: {scores['nl']!s:5} | en: {scores['en']!s:5}", end="")
        # quick heuristic: compare
        nz = scores['nl']
        ez = scores['en']
        verdict = ""
        if nz is None and ez is None:
            verdict = " (no data)"
        elif nz is None:
            verdict = " (no nl data)"
        elif ez is None:
            verdict = " (no en data)"
        else:
            diff = (ez or 0.0) - (nz or 0.0)
            if (ez or 0.0) >= 2.0 and diff >= 1.5:
                verdict = " => likely EN"
            elif (nz or 0.0) >= 2.0 and -diff >= 1.0:
                verdict = " => likely NL"
            else:
                verdict = " => ambiguous/close"
        print(verdict)
    print("\nNotes:")
    print("- zipf >= ~2.0 indicates reasonably common word in that language.")
    print("- thresholds used above are heuristic; adjust based on your data.")
    print("- If you deploy to Streamlit, add 'wordfreq' to requirements.txt if you want consistent behavior in cloud.")
    
if __name__ == '__main__':
    main()