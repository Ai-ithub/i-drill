# 🚀 Script برای راه‌اندازی مجدد داشبورد i-Drill
# این اسکریپت فرانت‌اند و بک‌اند را راه‌اندازی مجدد می‌کند

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   راه‌اندازی مجدد داشبورد i-Drill" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# مسیر پروژه
$projectRoot = $PSScriptRoot
$backendPath = Join-Path $projectRoot "src\backend"
$frontendPath = Join-Path $projectRoot "frontend"

# پاک کردن cache فرانت‌اند
Write-Host "🧹 پاک کردن cache فرانت‌اند..." -ForegroundColor Yellow
if (Test-Path "$frontendPath\node_modules\.vite") {
    Remove-Item -Recurse -Force "$frontendPath\node_modules\.vite" -ErrorAction SilentlyContinue
    Write-Host "✅ Cache پاک شد" -ForegroundColor Green
} else {
    Write-Host "⚠️  Cache از قبل پاک شده است" -ForegroundColor Yellow
}

Write-Host ""

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

# راهنمایی برای اجرای دستی
Write-Host "⚠️  لطفاً:" -ForegroundColor Yellow
Write-Host "1. اگر پنجره‌های PowerShell فرانت‌اند و بک‌اند باز هستند، آنها را ببندید (Ctrl+C)" -ForegroundColor White
Write-Host "2. سپس دوباره اجرا کنید:" -ForegroundColor White
Write-Host ""
Write-Host "   برای بک‌اند:" -ForegroundColor Cyan
Write-Host "   cd `"$backendPath`"" -ForegroundColor Gray
Write-Host "   python start_server.py" -ForegroundColor Gray
Write-Host ""
Write-Host "   برای فرانت‌اند:" -ForegroundColor Cyan
Write-Host "   cd `"$frontendPath`"" -ForegroundColor Gray
Write-Host "   npm run dev" -ForegroundColor Gray
Write-Host ""
Write-Host "3. مرورگر را باز کنید و Hard Refresh کنید (Ctrl+Shift+R)" -ForegroundColor White
Write-Host "4. یا Cache مرورگر را پاک کنید" -ForegroundColor White
Write-Host ""
Write-Host "📊 داشبورد: http://localhost:3001/dashboard" -ForegroundColor Cyan
Write-Host "📚 API Docs: http://localhost:8001/docs" -ForegroundColor Cyan
Write-Host ""

