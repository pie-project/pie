@echo off
rem pie installer - served at https://pie-project.org/install.cmd
rem
rem   curl -fsSL https://pie-project.org/install.cmd -o install.cmd && install.cmd && del install.cmd

setlocal

where powershell.exe >nul 2>nul
if errorlevel 1 (
  echo error: powershell.exe was not found. Install PowerShell or run install.ps1 directly.
  exit /b 1
)

if "%PIE_INSTALL_PS1_URL%"=="" set "PIE_INSTALL_PS1_URL=https://pie-project.org/install.ps1"

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $u=$env:PIE_INSTALL_PS1_URL; iex ((New-Object Net.WebClient).DownloadString($u))"
exit /b %ERRORLEVEL%
