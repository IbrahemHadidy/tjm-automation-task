@echo off
 Self-elevate to Admin
nul 2&1 %SYSTEMROOT%system32cacls.exe %SYSTEMROOT%system32configsystem
if '%errorlevel%' NEQ '0' ( goto UACPrompt ) else ( goto gotAdmin )
UACPrompt
echo Set UAC = CreateObject^(Shell.Application^)  %temp%getadmin.vbs
echo UAC.ShellExecute %~s0, , , runas, 1  %temp%getadmin.vbs
%temp%getadmin.vbs & exit B
gotAdmin
if exist %temp%getadmin.vbs ( del %temp%getadmin.vbs )
pushd %~dp0

 Execute Script
call .venvScriptsactivate
vlm-lab
pause
