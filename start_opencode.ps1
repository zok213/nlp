# NLP Exam Solver - OpenCode Launcher
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$OllamaPath = "C:\Users\VH\AppData\Local\Programs\Ollama"
$env:Path = "$OllamaPath;$env:Path"
$env:OLLAMA_HOST = "http://localhost:11434"
$env:CODE_MODEL = "exam-r1-solver"

Write-Host "Starting Ollama..." -NoNewline
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Write-Host " Ready" -ForegroundColor Green

Write-Host "`nOpenCode Commands:" -ForegroundColor Cyan
Write-Host "  /shell python exam_solver.py `"Your question`"" -ForegroundColor White
Write-Host "  /shell python exam_solver_pipeline.py image.jpg" -ForegroundColor White
Write-Host "  /shell python exam_solver.py --interactive" -ForegroundColor White

if (Get-Command opencode -ErrorAction SilentlyContinue) {
    Start-Process -FilePath "opencode" -WorkingDirectory $ProjectRoot
}
