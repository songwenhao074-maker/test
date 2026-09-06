$ErrorActionPreference = 'Stop'
$cleanupRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path.TrimEnd('\')
$planPath = Join-Path $PSScriptRoot 'cleanup_20260905\plan.json'
$cleanupPlan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($cleanupRoot -ne $cleanupPlan.root) { throw 'Workspace root mismatch' }

function Get-CheckedCleanupPath([string]$relative) {
    if ([IO.Path]::IsPathRooted($relative)) { throw "Expected relative path: $relative" }
    $target = [IO.Path]::GetFullPath((Join-Path $cleanupRoot $relative))
    if (-not $target.StartsWith($cleanupRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Outside workspace: $target"
    }
    if ($relative -match '(^|[\\/])\.(git|agents|codex|claude)([\\/]|$)') {
        throw "Protected application metadata: $relative"
    }
    $ancestor = $target
    while ($ancestor -and $ancestor -ne $cleanupRoot) {
        if (Test-Path -LiteralPath $ancestor) {
            $item = Get-Item -LiteralPath $ancestor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse point is not allowed: $ancestor"
            }
        }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
    return $target
}

if ((Get-FileHash -LiteralPath $cleanupPlan.archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $cleanupPlan.archive_sha256) {
    throw 'Archive hash mismatch'
}
# Validate every target before the first deletion. No recursive deletes are used.
foreach ($entry in $cleanupPlan.delete_files) {
    $target = Get-CheckedCleanupPath $entry.path
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw "Missing target: $target" }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
        throw "Target changed: $target"
    }
}
foreach ($entry in $cleanupPlan.delete_files) {
    $target = Get-CheckedCleanupPath $entry.path
    Remove-Item -LiteralPath $target -Force
}
$emptyDirectoriesRemoved = @()
foreach ($relative in $cleanupPlan.empty_directories) {
    $target = Get-CheckedCleanupPath $relative
    if ((Test-Path -LiteralPath $target -PathType Container) -and @(Get-ChildItem -LiteralPath $target -Force).Count -eq 0) {
        Remove-Item -LiteralPath $target -Force
        $emptyDirectoriesRemoved += $relative
    }
}
$receipt = @{
    root = $cleanupRoot
    deletedFiles = @($cleanupPlan.delete_files).Count
    emptyDirectoriesRemoved = $emptyDirectoriesRemoved
    archivePreserved = $cleanupPlan.archive
}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup_20260905\deletion_receipt.json') -Encoding UTF8
$receipt | ConvertTo-Json -Depth 8
