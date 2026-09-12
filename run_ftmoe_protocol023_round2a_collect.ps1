# Round 2A - collect the four calibrated streams, then verify each one.
#
# The round-1 streams are never touched: everything this script writes carries
# the '_calibrated' tag suffix (directive section 6).  The verification step is
# the same independent stream verifier round 1 used; a non-zero exit stops the
# run so no gate can be computed on an unverified stream.
param(
    [string]$Python = "D:\Anaconda\envs\dynmoe\python.exe",
    [string]$Selected = "artifacts\ftmoe_online\protocol_023\round2a\data_calibration\selected_generator.json",
    [string]$Suffix = "_calibrated",
    [string[]]$Only = @()
)
$ErrorActionPreference = "Continue"
$root = "F:\PreGANPlus-master"
Set-Location $root
$streamRoot = "artifacts\ftmoe_online\protocol_023\development_streams"

$jobs = @(
    @{ mode = "dev"; regime = $null },
    @{ mode = "single"; regime = "compute_first" },
    @{ mode = "single"; regime = "memory_first" },
    @{ mode = "single"; regime = "io_first" }
)

foreach ($job in $jobs) {
    $label = if ($job.mode -eq "dev") { "dev" } else { "single_$($job.regime)" }
    if ($Only.Count -gt 0 -and ($Only -notcontains $label)) { continue }
    $tag = if ($job.mode -eq "dev") { "dev_seed700_steps2880$Suffix" } else { "single_$($job.regime)_seed700_steps1200$Suffix" }
    $target = Join-Path $streamRoot $tag
    if (Test-Path (Join-Path $target "stream.npz")) {
        Write-Host "--- $label already collected: $tag (skipping collection, verifying) ---"
    } else {
        $started = Get-Date
        Write-Host "--- $label collecting at $($started.ToString('HH:mm:ss')) -> $tag ---"
        $arguments = @("prepare_ftmoe_protocol023_stream.py", "--mode", $job.mode,
                       "--output-root", $streamRoot, "--tag-suffix", $Suffix,
                       "--calibration-parameters", $Selected)
        if ($job.regime) { $arguments += @("--regime", $job.regime) }
        & $Python @arguments 2>&1 | Select-Object -Last 4
        $code = $LASTEXITCODE
        $secs = [int]((Get-Date) - $started).TotalSeconds
        Write-Host "--- $label collection exit=$code after $secs s ---"
        if ($code -ne 0) {
            Write-Host "--- STOP: collection of $label failed; no gate may be computed ---"
            exit 1
        }
    }
    Write-Host "--- verifying $tag (tag suffix '$Suffix') ---"
    & $Python "verify_ftmoe_protocol023_stream.py" $target --tag-suffix $Suffix 2>&1 | Select-Object -Last 2
    $code = $LASTEXITCODE
    Write-Host "--- $label verifier exit=$code ---"
    if ($code -ne 0) {
        Write-Host "--- STOP: verification of $label failed ---"
        exit 1
    }
}
Write-Host "COLLECTION DONE"
exit 0
