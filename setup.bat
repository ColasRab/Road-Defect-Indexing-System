@echo off

:: Create virtual environment if it doesn't exist
if not exist venv (
    python -m venv venv
    echo Virtual environment created.
)

:: Activate venv
call venv\Scripts\activate

:: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo Setup complete. Run "venv\Scripts\activate" to use the environment.
