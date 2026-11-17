param(
    [ValidateSet('test','collect','train','validate','all')][string]$Stage='all',
    [string]$Python='python',
    [string]$LegacyRoot='..',
    [string]$Dataset='..\agent4_v312b_idm_dataset',
    [string]$Baseline='..\agent4_v313_model\agent4_v313_steering.joblib',
    [string]$Collection='..\agent4_v315_data',
    [string]$ModelDir='..\agent4_v315_model',
    [string]$Report='..\agent4_v315_validation.json',
    [switch]$PublishSummary
)
$ErrorActionPreference='Stop'
$ScriptRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot=Split-Path -Parent $ScriptRoot
Push-Location $RepoRoot
try {
    function Invoke-Python([string[]]$argList) {
        & $Python @argList
        if ($LASTEXITCODE -ne 0) { throw "Python failed (exit $LASTEXITCODE)" }
    }
    if ($Stage -in @('test','all')) {
        Invoke-Python -argList @('agent4_recovery_v315\test_v315.py')
    }
    if ($Stage -in @('collect','all')) {
        $argsList=@('agent4_recovery_v315\collect_v315.py','--legacy-root',$LegacyRoot,
                    '--dataset',$Dataset,'--baseline',$Baseline,'--out-dir',$Collection)
        if (Test-Path (Join-Path $Collection 'collection_manifest.json')) { $argsList += '--resume' }
        Invoke-Python -argList $argsList
    }
    if ($Stage -in @('train','all')) {
        Invoke-Python -argList @('agent4_recovery_v315\train_v315.py','--legacy-root',$LegacyRoot,
                        '--dataset',$Dataset,'--baseline',$Baseline,'--collection',$Collection,
                        '--out-dir',$ModelDir)
    }
    if ($Stage -in @('validate','all')) {
        $argsList=@('agent4_recovery_v315\eval_v315.py','--legacy-root',$LegacyRoot,
                    '--dataset',$Dataset,'--model',"$ModelDir\agent4_v315_recovery.joblib",
                    '--mode','validation','--out',$Report)
        if (Test-Path $Report) { $argsList += '--resume' }
        Invoke-Python -argList $argsList
        if ($PublishSummary) {
            Invoke-Python -argList @('agent4_recovery_v315\publish_summary_v315.py',
                            '--evaluation',$Report,'--output','reports/v315_validation_summary.json')
            Write-Host 'Summary ready. Review it, then: git add reports/v315_validation_summary.json; git commit -m "V3.15 validation result"; git push'
        }
    }
} finally { Pop-Location }
