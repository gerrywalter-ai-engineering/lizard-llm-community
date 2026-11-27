@echo off
rem Lizard LLM Community Edition - one-click setup.
rem Checks Node.js, then installs the two native engines (prebuilt, checksum-verified).
setlocal
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js 22 or newer is required.
  echo Install it with:  winget install OpenJS.NodeJS.LTS
  echo or download it from https://nodejs.org, then run install.cmd again.
  pause
  exit /b 1
)
node scripts\setup.mjs %*
if errorlevel 1 (
  pause
  exit /b 1
)
echo.
set /p START="Start Lizard now? [Y/n] "
if /i "%START%"=="n" exit /b 0
call "%~dp0start.cmd"
