# Round 2A guard: a "calibrated" stream must actually differ from the
# registered one.  Twice during round 2A a calibration was silently discarded
# (a shared-family table being overwritten by the per-regime dict, then the
# phase switch resetting the probability to the registered value), and both
# times the giveaway was a byte-identical result.  This script makes that
# failure mode impossible to miss.
param(
    [string]$Python = "D:\Anaconda\envs\dynmoe\python.exe"
)
$ErrorActionPreference = "Continue"
$root = "F:\PreGANPlus-master"
Set-Location $root
$streamRoot = "artifacts\ftmoe_online\protocol_023\development_streams"
$pairs = @(
    @{ name = "compute_first"; registered = "single_compute_first_seed700_steps1200"; frozen = $true },
    @{ name = "memory_first"; registered = "single_memory_first_seed700_steps1200"; frozen = $false },
    @{ name = "io_first"; registered = "single_io_first_seed700_steps1200"; frozen = $false }
)
$fail = 0
foreach ($pair in $pairs) {
    $r = Join-Path $streamRoot $pair.registered
    $c = Join-Path $streamRoot "$($pair.registered)_calibrated"
    if (-not (Test-Path (Join-Path $c "stream.npz"))) {
        Write-Host "$($pair.name): calibrated stream missing"; $fail = 1; continue
    }
    $rh = (Get-FileHash (Join-Path $r "stream.npz") -Algorithm SHA256).Hash
    $ch = (Get-FileHash (Join-Path $c "stream.npz") -Algorithm SHA256).Hash
    $rm = Get-Content (Join-Path $r "manifest.json") -Raw | ConvertFrom-Json
    $cm = Get-Content (Join-Path $c "manifest.json") -Raw | ConvertFrom-Json
    $same = ($rh -eq $ch)
    $rEvents = $rm.task_level_summary.n_onsets
    $cEvents = $cm.task_level_summary.n_onsets
    if ($pair.frozen) {
        # regime A is frozen: its calibrated arm must be byte-identical
        $verdict = if ($same) { "identical-OK" } else { "DIFFERS-BAD" }
        if (-not $same) { $fail = 1 }
    } else {
        $verdict = if ($same) { "IDENTICAL-BAD" } else { "distinct" }
        if ($same) { $fail = 1 }
    }
    Write-Host ("{0,-14} {1,-14} onsets {2} -> {3}  cascade events {4} -> {5}" -f `
        $pair.name, $verdict, $rEvents, $cEvents, `
        $rm.per_regime_summary.A.gate.cascade_events_registered, `
        $cm.per_regime_summary.A.gate.cascade_events_registered)
    if (-not $pair.frozen -and ($cEvents -le $rEvents)) {
        Write-Host "  WARNING: $($pair.name) calibrated arm has no more onsets than the registered arm"
    }
}
if ($fail -ne 0) { Write-Host "CALIBRATION GUARD: FAIL"; exit 1 }
Write-Host "CALIBRATION GUARD: PASS (A byte-identical as required; B/C genuinely re-calibrated)"
exit 0
