@echo off
chcp 65001 >nul
echo ===============================================
echo EmoMAS Experiment Suite
echo ===============================================
echo.

set scenarios=10
set iterations=1
set model_creditor=gpt-4o-mini
set model_debtor=gpt-4o-mini
set out_dir=table2_ssd
set dataset=student

set creditor_types=vanilla prompt bayesian  rl_agents gametheory gpt coherence
set debtor_types=pressure victim threat

echo EXPERIMENT CONFIGURATION:
echo   Scenarios per experiment: %scenarios%
echo   Creditor Model: %model_creditor%
echo   Debtor Model: %model_debtor%
echo   Dataset: %dataset%
echo   Output Directory: %out_dir%
echo.

set /a completed=0
set /a total=0

for %%i in (%creditor_types%) do (
    for %%j in (%debtor_types%) do (
        set /a total+=1
    )
)

echo Starting %total% experiments...
echo.

for %%c in (%creditor_types%) do (
    for %%d in (%debtor_types%) do (
        set /a completed+=1
        echo ===============================================
        echo EXPERIMENT %completed%/%total%
        echo   Creditor Model: %%c
        echo   Debtor Strategy: %%d
        echo   Dataset: %dataset%
        echo ===============================================
        
        python experiments/run_all_datasets.py --model_type %%c --dataset_type %dataset% --model_creditor "%model_creditor%" --model_debtor "%model_debtor%" --scenarios %scenarios% --iterations %iterations% --debtor_model_type %%d --out_dir "%out_dir%"
        
        if errorlevel 1 (
            echo [FAILED] %%c with %%d on %dataset%
        ) else (
            echo [SUCCESS] %%c with %%d on %dataset%
        )
        echo.
        timeout /t 2 /nobreak > nul
    )
)

echo All experiments completed!
echo Results in: %out_dir%
pause