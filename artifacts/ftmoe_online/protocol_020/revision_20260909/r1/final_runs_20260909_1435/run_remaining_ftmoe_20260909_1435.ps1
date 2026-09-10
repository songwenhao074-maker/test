Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Set-Location 'F:\PreGANPlus-master'
$py = 'D:\Anaconda\envs\dynmoe\python.exe'
$formal = 'F:\PreGANPlus-master\artifacts\ftmoe_online\protocol_020\revision_20260909\r1\final_runs_20260909_1435'

if (-not (Test-Path -LiteralPath $py)) {
    throw "Python executable missing: $py"
}

$ramText = (& $py -c "import psutil; print(psutil.virtual_memory().available / 1024**3)").Trim()
$ramGiB = [double]$ramText
Write-Output ("RAM_AVAILABLE_GIB={0:N3}" -f $ramGiB)
if ($ramGiB -lt 3.0) {
    throw "Available RAM below 3 GiB: $ramGiB"
}

$env:FTMOE020_RAM_GUARD_GIB = '3.0'
$env:OMP_NUM_THREADS = '3'
$env:MKL_NUM_THREADS = '3'
$env:OPENBLAS_NUM_THREADS = '3'

$runs = @(
    [pscustomobject]@{ Method = 'C-residual-off'; ReplaySeed = 500 },
    [pscustomobject]@{ Method = 'C-residual-on'; ReplaySeed = 500 },
    [pscustomobject]@{ Method = 'A'; ReplaySeed = 501 },
    [pscustomobject]@{ Method = 'C-legacy'; ReplaySeed = 501 },
    [pscustomobject]@{ Method = 'C-residual-off'; ReplaySeed = 501 },
    [pscustomobject]@{ Method = 'C-residual-on'; ReplaySeed = 501 }
)

foreach ($run in $runs) {
    $stem = "{0}_dev{1}_m1" -f $run.Method, $run.ReplaySeed
    $output = Join-Path $formal $stem
    $log = Join-Path $formal ("{0}.stdout_stderr.log" -f $stem)

    if (Test-Path -LiteralPath $output) {
        throw "Refusing to overwrite existing output directory: $output"
    }
    if (Test-Path -LiteralPath $log) {
        throw "Refusing to overwrite existing log: $log"
    }

    $args = @(
        'run_ftmoe_protocol020_r1.py',
        '--method', $run.Method,
        '--model-seed', '1',
        '--replay-seed', [string]$run.ReplaySeed,
        '--checkpoint-path', 'artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt',
        '--stream', ("artifacts/ftmoe_online/protocol_020/drift_streams/dev_seed{0}_steps2000" -f $run.ReplaySeed),
        '--output', $output
    )

    Write-Output ("LAUNCH method={0} replay_seed={1} output={2} log={3}" -f $run.Method, $run.ReplaySeed, $output, $log)
    & $py @args > $log 2>&1
    $exitCode = $LASTEXITCODE
    Add-Content -LiteralPath $log -Value ("`nEXITCODE={0}" -f $exitCode)
    Write-Output ("DONE method={0} replay_seed={1} exitcode={2} output={3} log={4}" -f $run.Method, $run.ReplaySeed, $exitCode, $output, $log)

    if ($exitCode -ne 0) {
        throw "Run failed: $stem exitcode=$exitCode; see $log"
    }
    $summary = Join-Path $output 'summary.json'
    if (-not (Test-Path -LiteralPath $summary)) {
        throw "Run completed without summary.json: $summary"
    }
}

Write-Output 'ALL_SIX_RUNS_COMPLETE'
