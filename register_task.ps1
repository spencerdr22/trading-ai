# register_task.ps1
# Run ONCE as Administrator to register a Windows Task Scheduler job.
# The task wakes the PC from sleep at 9:20 AM ET and starts trading.
#
# Prerequisites:
#   1. Enable wake timers in Windows Power Settings:
#      Control Panel → Power Options → Change plan settings
#      → Change advanced power settings → Sleep
#      → Allow wake timers → Enable
#
# Usage:
#   Right-click PowerShell → "Run as Administrator"
#   .\register_task.ps1

$TaskName   = "MES-Paper-Trading"
$ProjectDir = "C:\Users\spenc\Documents\trading-ai"
$PythonExe  = "$ProjectDir\.venv\Scripts\python.exe"
$Script     = "$ProjectDir\start_trading.py"

# Remove existing task if present
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task: $TaskName"
}

# Action: run python start_trading.py
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument $Script `
    -WorkingDirectory $ProjectDir

# Trigger: every weekday at 9:20 AM Eastern
# (start_trading.py will wait the remaining 10 min until 9:30 market open)
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "9:20AM"

# CRITICAL: enable wake from sleep
$Trigger.RepetitionDuration = $null
$Trigger.Enabled = $true

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -WakeToRun `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 10)

$Principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName   $TaskName `
    -Action     $Action `
    -Trigger    $Trigger `
    -Settings   $Settings `
    -Principal  $Principal `
    -Description "Wakes PC and starts MES paper trading at 9:20 AM ET weekdays"

Write-Host ""
Write-Host "==================================================="
Write-Host "  Task registered: $TaskName"
Write-Host "  Fires: Mon-Fri at 9:20 AM (your local time)"
Write-Host "  WakeToRun: ENABLED"
Write-Host "==================================================="
Write-Host ""
Write-Host "IMPORTANT — also enable Wake Timers in power settings:"
Write-Host "  Run this command to enable them automatically:"
Write-Host ""
Write-Host "  powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1"
Write-Host "  powercfg /setactive SCHEME_CURRENT"
Write-Host ""
Write-Host "Then you can sleep your PC normally."
Write-Host ""
Write-Host "To remove this task later:"
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
