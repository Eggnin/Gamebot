param(
    [ValidateSet('natural','wall','encircled','d841six')][string]$Layout='natural',
    [string]$Seed='3JY1 6FLR',
    [ValidateRange(1,60)][int]$Seconds=60,
    [switch]$DodgeOnly,
    [string]$Output=''
)
$ErrorActionPreference='Stop'
$engineRoot=Split-Path -Parent $PSScriptRoot
$isaacRoot='D:/Isaac'
$runner=Join-Path $isaacRoot 'research/external-validation/session-damage-quick.mjs'
$worker=Join-Path $engineRoot 'src/engine_worker.mjs'
$runRoot=Join-Path $engineRoot 'runs/frowning-gaper'
if(!$Output) {
    $stamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $Output=Join-Path $isaacRoot "validation/gaper-$Layout-$stamp"
}
$arguments=@($runner,'--ids=10.0.0','--workers=1',"--seconds=$Seconds", "--seed=$Seed",
    "--layout=$Layout",'--diagnostics=true',"--output=$Output", "--worker=$worker", "--run-root=$runRoot",
    "--no-fire=$($DodgeOnly.ToString().ToLowerInvariant())")
Push-Location $isaacRoot
try {
    & node @arguments
    if($LASTEXITCODE -ne 0) { throw "Training run failed; see $Output" }
} finally { Pop-Location }
