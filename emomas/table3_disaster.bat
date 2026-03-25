@echo off
echo Starting disaster dataset experiments...
echo.

rem Configuration
set scenarios=4
set iterations=1
set debtor_model=qwen-1.5b
set output_dir=table3_disaster_qwen-1.5b

rem Create output directory
if not exist "%output_dir%" mkdir "%output_dir%"

rem Run experiments for both qwen models
for %%m in ( "qwen-1.5b") do (
    for %%c in (  "gpt") do (
        for %%d in ("victim") do (
            echo Running: %%m - %%c - %%d - disaster
            echo.
            
            python experiments/run_all_datasets.py --model_type %%c --dataset_type disaster --model_creditor %%m --model_debtor "%debtor_model%" --scenarios %scenarios% --iterations %iterations% --debtor_model_type %%d --out_dir "%output_dir%"
            
            if errorlevel 1 (
                echo [ERROR] Failed: %%m - %%c - %%d - disaster
            ) else (
                echo [OK] Success: %%m - %%c - %%d - disaster
            )
            echo.
            timeout /t 2 /nobreak >nul
        )
    )
)

echo All disaster experiments completed!
echo Results saved to: %output_dir%
pause