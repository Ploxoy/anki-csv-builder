# scripts/check_natuurkunde.py
# Check why 'natuurkunde' is accepted/rejected by is_probably_dutch_word

from core.sanitize_validate import is_probably_dutch_word
try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None

word = "natuurkunde"
ok, reason = is_probably_dutch_word(word)
print(f"word: {word!r} => ok={ok}, reason={reason}")

if zipf_frequency is not None:
    print("wordfreq scores (if available):")
    print("  nl:", zipf_frequency(word, "nl"))
    print("  en:", zipf_frequency(word, "en"))