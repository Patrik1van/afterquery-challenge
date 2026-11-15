# Run full judge evaluation over tasks/beam-steering-s000..s004 on Modal
# Runs oracle, nop, stop-early, and 3x opencode (Gemini 3.8-Flash), then prints summary table.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot

# 1. Set PYTHONPATH
$env:PYTHONPATH = $repoRoot

# 2. Load API keys from .env if present (never print)
$envFile = Join-Path $repoRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

Write-Host "=================================================================="
Write-Host "CERN TT20 Beam Steering Benchmark — Harbor Modal Evaluation"
Write-Host "Tasks: tasks/beam-steering-s000..s004"
Write-Host "Concurrency: 5"
Write-Host "=================================================================="

# 3. Oracle (Bayes reference)
Write-Host "`n>>> [1/4] Running Oracle Agent (Bayes reference) on tasks s000..s004..."
harbor run -p tasks -i "beam-steering-s*" -n 5 -e modal -a oracle --env-file .env

# 4. Nop (Do-nothing baseline)
Write-Host "`n>>> [2/4] Running Nop Agent (Do-nothing baseline) on tasks s000..s004..."
harbor run -p tasks -i "beam-steering-s*" -n 5 -e modal -a nop --env-file .env

# 5. Stop-Early baseline
Write-Host "`n>>> [3/4] Running Stop-Early Agent baseline on tasks s000..s004..."
harbor run -p tasks -i "beam-steering-s*" -n 5 -e modal -a agents.baselines:StopEarlyAgent --env-file .env

# 6. OpenCode (Gemini 3.8-Flash with -k 3)
Write-Host "`n>>> [4/4] Running OpenCode + Gemini 3.8-Flash (k=3) on tasks s000..s004..."
harbor run -p tasks -i "beam-steering-s*" -n 5 -e modal -a opencode -m gemini/gemini-3.8-flash --api-key-name GEMINI_API_KEY -k 3 --env-file .env

# 7. Export trajectories for webapp replay
Write-Host "`n>>> Exporting trajectories..."
python scripts/export_trajectories.py

# 8. Print results summary table
Write-Host "`n>>> Generating Results Table..."
python scripts/summarize_results.py
