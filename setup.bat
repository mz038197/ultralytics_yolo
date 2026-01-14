@echo off
echo ======= 準備建立課程基本環境與模組 =======
echo.

echo [1/3] 建立Python虛擬環境...
if not exist venv (
    py -3 -m venv venv
)

echo [2/3] 啟動Python虛擬環境中...
call venv\Scripts\activate
echo [2/3] 啟動Python虛擬環境完成! 

echo [3/3] 安裝Python模組中... (過程將花費1分鐘左右)
pip install -r requirements.txt
echo [3/3] 安裝Python模組完成! 請用 VS Code 打開此資料夾進行課程嚕!

pause
exit /b
