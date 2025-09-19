"""Compatibility entrypoint for Streamlit app.

The actual UI now lives in ``app/app.py``. Keeping this file ensures
existing commands like ``streamlit run anki_csv_builder.py`` continue to work
while we transition to the new structure.
"""
from importlib import import_module


def main() -> None:
    """Delegate execution to ``app.app``."""
    import_module("app.app")


if __name__ == "__main__":
    main()
