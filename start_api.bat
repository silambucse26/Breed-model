@echo off
echo Starting Cattle and Buffalo Breed Classification API on http://localhost:8000 ...
python -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
