@echo off
echo ===============================================
echo 🧪 COMPREHENSIVE EMOMAS EXPERIMENT SUITE
echo ===============================================
echo.

set scenarios=10
set iterations=1
set model_creditor=qwen-1.5b
set model_debtor=gpt-4o-mini
set debtor_model_type=vanilla
set out_dir=table1_4b_mas

set model_types= bayesian gpt
set datasets=disaster student medical

echo 📊 EXPERIMENT CONFIGURATION:
echo   Scenarios per experiment: %scenarios%
echo   Creditor Model: %model_creditor%
echo   Output Directory: %out_dir%
echo.

set /a completed=0
set /a total=0

for %%i in (%model_types%) do (
    for %%j in (%datasets%) do (
        set /a total+=1
    )
)

echo Starting %total% experiments...
echo.

for %%m in (%model_types%) do (
    for %%d in (%datasets%) do (
        set /a completed+=1
        echo ===============================================
        echo 🔬 EXPERIMENT !completed!/%total%
        echo   Model Type: %%m
        echo   Dataset: %%d
        echo ===============================================
        
        python experiments/run_all_datasets.py --model_type %%m --dataset_type %%d --model_creditor "%model_creditor%" --model_debtor "%model_debtor%" --scenarios %scenarios% --iterations %iterations% --debtor_model_type %debtor_model_type% --out_dir "%out_dir%"
        
        if errorlevel 1 (
            echo ❌ FAILED: %%m on %%d
        ) else (
            echo ✅ SUCCESS: %%m on %%d
        )
        echo.
        timeout /t 2 /nobreak > nul
    )
)

echo All experiments completed!