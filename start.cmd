@echo off
rem Lizard LLM Community Edition - start the app and open it in the browser.
setlocal
cd /d "%~dp0"
node src\cli.js app %*
