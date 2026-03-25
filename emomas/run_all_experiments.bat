@echo off
chcp 65001 >nul
echo ===============================================
echo 🚀 MASTER EMO-MAS EXPERIMENT RUNNER
echo ===============================================
echo.

echo This will run all 5 experiment suites:
echo 1. table1_7b.bat    - Qwen-7b on all datasets
echo 2. table1_gpt.bat   - GPT-4o-mini on all datasets  
echo 3. table2_ssad.bat  - GPT-4o-mini vs emotional debtors on student
echo 4. table2_ssd.bat   - GPT-4o-mini vs emotional debtors on medical
echo 5. table3_disaster.bat - Qwen models vs emotional debtors on disaster
echo.

set start_time=%time%
echo Master run started at: %date% %start_time%
echo.

set scripts=table1_7b.bat table1_gpt.bat table2_ssad.bat table2_ssd.bat table3_disaster.bat
set count=0

for %%s in (%scripts%) do (
    set /a count+=1
    echo ===============================================
    echo 🏃 RUNNING SCRIPT %count% of 5: %%s
    echo ===============================================
    echo.
    
    if exist "%%s" (
        echo Starting %%s...
        echo.
        call "%%s"
        echo.
        echo ✅ Completed: %%s
    ) else (
        echo ❌ Script not found: %%s
    )
    
    echo.
    echo Pausing 5 seconds before next script...
    echo.
    timeout /t 5 /nobreak >nul
)

set end_time=%time%
call :CalculateDuration "%start_time%" "%end_time%" total_duration

echo ===============================================
echo 🏁 MASTER RUN COMPLETED
echo ===============================================
echo.
echo 📊 SUMMARY:
echo   Total scripts: 5
echo   Run duration: %total_duration%
echo   Started: %date% %start_time%
echo   Completed: %date% %end_time%
echo.
echo 📁 Output directories created:
if exist table1_7b echo   - table1_7b/
if exist table1_gpt echo   - table1_gpt/
if exist table2_ssad echo   - table2_ssad/
if exist table2_ssd echo   - table2_ssd/
if exist table3_disaster echo   - table3_disaster/
echo.
echo 🎯 All experiments completed!
echo ===============================================
pause
goto :EOF

:CalculateDuration
setlocal
set "start=%~1"
set "end=%~2"
set "start=%start:"=%"
set "end=%end:"=%"

for /f "tokens=1-3 delims=:.," %%a in ("%start%") do set /a start_secs=((%%a*60)+%%b)*60+%%c
for /f "tokens=1-3 delims=:.," %%a in ("%end%") do set /a end_secs=((%%a*60)+%%b)*60+%%c

if %end_secs% lss %start_secs% set /a end_secs+=86400
set /a diff_secs=end_secs-start_secs

if %diff_secs% lss 3600 (
    set /a diff_minutes=diff_secs/60
    set /a diff_secs=diff_secs %% 60
    set "duration=%diff_minutes%:%02d%diff_secs%"
) else (
    set /a diff_hours=diff_secs/3600
    set /a diff_secs=diff_secs %% 3600
    set /a diff_minutes=diff_secs/60
    set /a diff_secs=diff_secs %% 60
    set "duration=%diff_hours%:%02d%diff_minutes%:%02d%diff_secs%"
)

endlocal & set "%~3=%duration%"
goto :EOF