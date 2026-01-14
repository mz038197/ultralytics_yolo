@echo off
echo ======= Setup Python Environment =======
echo.

echo [1/3] Install Python...
if not exist venv (
    py -3 -m venv venv
)

echo [2/3] Activate Python Environment...
call venv\Scripts\activate
echo [2/3] Activate Python Environment! 

echo [3/3] Install Python Packages... (Optional: Install all packages)
pip install -r requirements.txt
echo [3/3] Install Python Packages! Ready to use VS Code!

pause
exit /b
