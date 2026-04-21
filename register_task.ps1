# Self-elevate if not already running as Administrator
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]"Administrator")) {
    Start-Process powershell "-ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

$ProjectDir   = "C:\Users\spenc\Documents\trading-ai"
$PythonExe    = "$ProjectDir\.venv\Scripts\python.exe"
$AutostartPy  = "$ProjectDir\autostart.py"
$DashPy       = "$ProjectDir\app\monitor\dashboard.py"
$StreamlitExe = "$ProjectDir\.venv\Scripts\streamlit.exe"
$CurrentUser  = (Get-CimInstance Win32_ComputerSystem).UserName

Write-Host ""
Write-Host "============================================================"
Write-Host "  Trading-AI Task Scheduler Registration"
Write-Host "  User: $CurrentUser"
Write-Host "============================================================"
Write-Host ""

if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: Python not found at $PythonExe"
    Write-Host "Run install_deps.bat first."
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path $AutostartPy)) {
    Write-Host "ERROR: autostart.py not found at $AutostartPy"
    Read-Host "Press Enter to exit"
    exit 1
}

# --- Task 1: Daily trading session Mon-Fri 9:20 AM ---

$TaskName1 = "TradingAI-Daily"

if (Get-ScheduledTask -TaskName $TaskName1 -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName1 -Confirm:$false
    Write-Host "  Removed existing task: $TaskName1"
}

$Action1 = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$AutostartPy`"" `
    -WorkingDirectory $ProjectDir

$Trigger1 = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "9:20AM"

$Settings1 = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 10) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName1 `
    -Action $Action1 `
    -Trigger $Trigger1 `
    -Settings $Settings1 `
    -RunLevel Highest `
    -Force | Out-Null

Write-Host "  [OK] $TaskName1 -- Mon-Fri 9:20 AM (WakeToRun=true)"

# --- Task 2: Dashboard on every login ---

$TaskName2 = "TradingAI-Dashboard"

if (Get-ScheduledTask -TaskName $TaskName2 -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName2 -Confirm:$false
    Write-Host "  Removed existing task: $TaskName2"
}

$DashArgs = "run `"$DashPy`" --server.port 8501 --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none"

$Action2 = New-ScheduledTaskAction `
    -Execute $StreamlitExe `
    -Argument $DashArgs `
    -WorkingDirectory $ProjectDir

$Trigger2 = New-ScheduledTaskTrigger -AtLogOn

$Settings2 = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName2 `
    -Action $Action2 `
    -Trigger $Trigger2 `
    -Settings $Settings2 `
    -RunLevel Limited `
    -Force | Out-Null

Write-Host "  [OK] $TaskName2 -- starts at every login, restarts on crash"

# --- Enable wake timers ---

Write-Host ""
Write-Host "  Enabling wake timers..."
& powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
& powercfg /setdcvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
& powercfg /setactive SCHEME_CURRENT
Write-Host "  [OK] Wake timers enabled"

# --- Verify ---

Write-Host ""
Write-Host "  Verifying..."
$t1 = Get-ScheduledTask -TaskName $TaskName1 -ErrorAction SilentlyContinue
$t2 = Get-ScheduledTask -TaskName $TaskName2 -ErrorAction SilentlyContinue

if ($t1) {
    Write-Host "  [OK] $TaskName1 state: $($t1.State)"
} else {
    Write-Host "  [FAIL] $TaskName1 not registered"
}

if ($t2) {
    Write-Host "  [OK] $TaskName2 state: $($t2.State)"
} else {
    Write-Host "  [FAIL] $TaskName2 not registered"
}

# --- Time zone note ---

$tz = [System.TimeZoneInfo]::Local
Write-Host ""
Write-Host "  PC time zone: $($tz.DisplayName)"

# --- Done ---

Write-Host ""
Write-Host "============================================================"
Write-Host "  DONE."
Write-Host ""
Write-Host "  TradingAI-Daily    : Mon-Fri 9:20 AM, wakes PC from sleep"
Write-Host "  TradingAI-Dashboard: Starts at every login on port 8501"
Write-Host ""
Write-Host "  Verify: taskschd.msc"
Write-Host ""
Write-Host "  You can now put the PC to sleep. It will wake at 9:20 AM."
Write-Host "============================================================"
Write-Host ""
Read-Host "Press Enter to close"
