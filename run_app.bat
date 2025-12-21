@echo off
REM Launch Anki CSV Builder outside the dev environment
setlocal

REM Change to the project directory
cd /d D:\github\anki\dev\anki-csv-builder

REM Activate virtual environment if present
if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
) else (
    echo [info] .venv not found. Ensure dependencies are installed.
)

REM Start the Streamlit app
streamlit run app/app.py

endlocal
