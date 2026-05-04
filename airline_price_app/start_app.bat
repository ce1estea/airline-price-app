@echo off
setlocal ENABLEDELAYEDEXPANSION

echo ---------------------------------------------
echo     Airline Price Predictor - Auto Setup
echo ---------------------------------------------
echo.

:: -------------------------------------------------------
:: 1. Check if Python exists
:: -------------------------------------------------------
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed.
    echo Please install Python >= 3.9 from https://www.python.org/downloads/
    pause
    exit /b
)

echo Python detected.
echo.

:: -------------------------------------------------------
:: 2. Create virtual environment if missing
:: -------------------------------------------------------
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
)
echo Virtual environment ready.
echo.

:: -------------------------------------------------------
:: 3. Activate venv
:: -------------------------------------------------------
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Could not activate virtual environment.
    pause
    exit /b
)
echo Virtual environment activated.
echo.

:: -------------------------------------------------------
:: 4. Ensure requirements.txt exists
:: -------------------------------------------------------
if not exist "requirements.txt" (
    echo Creating requirements.txt ...
    (
        echo flask
        echo pandas
        echo matplotlib
        echo scikit-learn
        echo joblib
    ) > requirements.txt
)
echo requirements.txt ready.
echo.

:: -------------------------------------------------------
:: 5. Install required packages
:: -------------------------------------------------------
echo Installing dependencies (this may take a minute)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b
)
echo Dependencies installed.
echo.

:: -------------------------------------------------------
:: 6. Launch web app (Chrome → Firefox → Edge → Default)
:: -------------------------------------------------------
echo Opening browser...

:: Try Chrome first
where chrome >nul 2>nul
if %errorlevel%==0 (
    echo Chrome found. Opening in Chrome...
    start "" chrome --new-tab http://127.0.0.1:5000
) else (
    :: Try Firefox second
    where firefox >nul 2>nul
    if %errorlevel%==0 (
        echo Chrome not found. Opening in Firefox...
        start "" firefox -new-tab http://127.0.0.1:5000
    ) else (
        :: Try Edge third
        where msedge >nul 2>nul
        if %errorlevel%==0 (
            echo Chrome/Firefox not found. Opening in Edge...
            start "" msedge --new-tab http://127.0.0.1:5000
        ) else (
            echo No Chrome/Firefox/Edge detected. Opening default browser...
            start http://127.0.0.1:5000
        )
    )
)

echo Starting Flask app...
python app.py

:: When user closes the app, Flask exits and the batch file ends
echo Application closed.
exit
