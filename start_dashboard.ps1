# 🚀 Script برای اجرای داشبورد i-Drill
# این اسکریپت بک‌اند و فرانت‌اند را همزمان اجرا می‌کند

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   اجرای داشبورد i-Drill" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# مسیر پروژه
$projectRoot = $PSScriptRoot
$backendPath = Join-Path $projectRoot "src\backend"
$frontendPath = Join-Path $projectRoot "frontend"

# بررسی وجود مسیرها
if (-not (Test-Path $backendPath)) {
    Write-Host "❌ خطا: مسیر بک‌اند پیدا نشد: $backendPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $frontendPath)) {
    Write-Host "❌ خطا: مسیر فرانت‌اند پیدا نشد: $frontendPath" -ForegroundColor Red
    exit 1
}

Write-Host "✅ مسیرها بررسی شدند" -ForegroundColor Green
Write-Host ""

# اجرای بک‌اند در پنجره جدید
Write-Host "🚀 در حال راه‌اندازی بک‌اند..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; Write-Host '=== Backend Server (پورت 8001) ===' -ForegroundColor Cyan; python start_server.py" -WindowStyle Normal

Start-Sleep -Seconds 3

# اجرای فرانت‌اند در پنجره جدید
Write-Host "🚀 در حال راه‌اندازی فرانت‌اند..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; Write-Host '=== Frontend Dashboard (پورت 3000) ===' -ForegroundColor Cyan; npm run dev" -WindowStyle Normal

Write-Host ""
Write-Host "✅ هر دو سرور راه‌اندازی شدند!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 داشبورد: http://localhost:3000" -ForegroundColor Cyan
Write-Host "📚 API Docs: http://localhost:8001/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  توجه: پنجره‌های PowerShell را باز نگه دارید!" -ForegroundColor Yellow
Write-Host ""
Write-Host "برای توقف سرورها، پنجره‌ها را ببندید یا Ctrl+C را بزنید." -ForegroundColor Gray

