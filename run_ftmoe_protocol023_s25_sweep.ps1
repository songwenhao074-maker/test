# Round 2A - sequential sweep driver for the S2.5 calibration candidates.
# Each candidate runs in its own process so that no candidate's injected
# parameter table can leak into the next one.  Failures are recorded, not
# swallowed.
param(
    [string]$Python = "D:\Anaconda\envs\dynmoe\python.exe",
    [string]$Prefix = "c_",
    [int]$Steps = 320
)
$ErrorActionPreference = "Continue"
$root = "F:\PreGANPlus-master"
$candidates = & $Python "$root\calibrate_ftmoe_protocol023_s25.py" --list |
    ForEach-Object { ($_ -split '\s+')[0] } |
    Where-Object { $_ -like "$Prefix*" }
Write-Host "sweep candidates: $($candidates -join ', ') steps=$Steps"
foreach ($c in $candidates) {
    $started = Get-Date
    Write-Host "--- $c started $($started.ToString('HH:mm:ss')) ---"
    & $Python "$root\calibrate_ftmoe_protocol023_s25.py" --candidate $c --steps $Steps 2>&1 |
        Select-Object -Last 3
    $secs = [int]((Get-Date) - $started).TotalSeconds
    Write-Host "--- $c finished $((Get-Date).ToString('HH:mm:ss')) after $secs s ---"
}
Write-Host "SWEEP DONE"
