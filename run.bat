@echo off
echo Starting DermaVision Project...

echo [1] Starting FastAPI Backend on port 8000...
start "DermaVision Backend" cmd /k "cd backend && python -m uvicorn app:app --reload --port 8000"

echo [2] Starting Frontend UI on port 3000...
start "DermaVision Frontend" cmd /k "cd frontend && python -m http.server 3000"

echo Launching browser...
start http://localhost:3000

echo Both servers have been launched in new windows!
pause
