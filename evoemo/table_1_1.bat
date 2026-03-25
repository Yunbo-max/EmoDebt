@REM @echo off
@REM REM Comprehensive Two-Stage Experiment Suite: All Methods vs Vanilla
@REM REM Training Stage: Train each method against vanilla opponent
@REM REM Testing Stage: Test trained models against vanilla

@REM echo ================================================================================
@REM echo 🚀 COMPREHENSIVE TWO-STAGE EXPERIMENT SUITE: ALL METHODS vs VANILLA
@REM echo ================================================================================
@REM echo Sequential Training and Testing: Train each method → Test immediately
@REM echo Methods: vanilla, qlearning, dqn, evolutionary, hierarchical
@REM echo ================================================================================

@REM REM =============================================================================
@REM REM METHOD 1: VANILLA vs VANILLA (BASELINE)
@REM REM =============================================================================

@REM echo.
@REM echo 📊 1/5 - VANILLA vs VANILLA (BASELINE)
@REM echo =============================================================================

@REM echo 🎯 Training: Vanilla vs Vanilla
@REM python experiments/run_training_testing.py --stage train ^
@REM   --creditor_method vanilla --debtor_method vanilla ^
@REM   --creditor_model deepseek-chat --debtor_model deepseek-chat ^
@REM   --scenarios 80 --train_seed 42

@REM if %errorlevel% neq 0 (
@REM     echo ❌ Vanilla training failed
@REM     pause
@REM     exit /b 1
@REM )

@REM echo 🥊 Testing: Vanilla vs Vanilla
@REM python experiments/run_training_testing.py --stage test ^
@REM   --creditor_method_test vanilla --debtor_method_test vanilla ^
@REM   --creditor_model deepseek-chat --debtor_model deepseek-chat ^
@REM   --scenarios 10 --test_seed 42 --iterations 5

@REM if %errorlevel% neq 0 (
@REM     echo ❌ Vanilla testing failed
@REM     pause
@REM     exit /b 1
@REM )

@REM echo ✅ Vanilla vs Vanilla completed successfully!

@REM REM =============================================================================
@REM REM METHOD 2: Q-LEARNING vs VANILLA
@REM REM =============================================================================

@REM echo.
@REM echo 📊 2/5 - Q-LEARNING vs VANILLA
@REM echo =============================================================================

@REM echo 🎯 Training: Q-Learning vs Vanilla
@REM python experiments/run_training_testing.py --stage train ^
@REM   --creditor_method qlearning --debtor_method vanilla ^
@REM   --creditor_model deepseek-chat --debtor_model deepseek-chat ^
@REM   --scenarios 80 --train_seed 42 ^
@REM   --episodes 240 --episodes_per_scenario 3 ^
@REM   --learning_rate 0.15 --discount_factor 0.9 ^
@REM   --exploration_rate 0.8 --exploration_decay 0.995 ^
@REM   --temperature 1.2

@REM if %errorlevel% neq 0 (
@REM     echo ❌ Q-Learning training failed
@REM     pause
@REM     exit /b 1
@REM )

@REM echo 🥊 Testing: Q-Learning vs Vanilla
@REM python experiments/run_training_testing.py --stage test ^
@REM   --creditor_method_test qlearning --debtor_method_test vanilla ^
@REM   --creditor_model deepseek-chat --debtor_model deepseek-chat ^
@REM   --scenarios 10 --test_seed 42 --iterations 1

@REM if %errorlevel% neq 0 (
@REM     echo ❌ Q-Learning testing failed
@REM     pause
@REM     exit /b 1
@REM )

@REM echo 🥊 Testing: Vanilla vs Q-Learning (reverse)
@REM python experiments/run_training_testing.py --stage test ^
@REM   --creditor_method_test vanilla --debtor_method_test qlearning ^
@REM   --creditor_model deepseek-chat --debtor_model deepseek-chat ^
@REM   --scenarios 10 --test_seed 42 --iterations 1

@REM if %errorlevel% neq 0 (
@REM     echo ❌ Vanilla vs Q-Learning testing failed
@REM     pause
@REM     exit /b 1
@REM )

@REM echo ✅ Q-Learning vs Vanilla completed successfully!

REM =============================================================================
REM METHOD 3: DQN vs VANILLA
REM =============================================================================

echo.
echo 📊 3/5 - DQN vs VANILLA
echo =============================================================================

echo 🎯 Training: DQN vs Vanilla
python experiments/run_training_testing.py --stage train ^
  --creditor_method dqn --debtor_method vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 80 --train_seed 42 ^
  --episodes 240 --episodes_per_scenario 3 ^
  --learning_rate 5e-4 --discount_factor 0.95 ^
  --exploration_rate 0.9 --exploration_decay 0.995 ^
  --batch_size 32 --replay_buffer_size 5000

if %errorlevel% neq 0 (
    echo ❌ DQN training failed
    pause
    exit /b 1
)

echo 🥊 Testing: DQN vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test dqn --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ DQN testing failed
    pause
    exit /b 1
)

echo 🥊 Testing: Vanilla vs DQN (reverse)
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test vanilla --debtor_method_test dqn ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Vanilla vs DQN testing failed
    pause
    exit /b 1
)

echo ✅ DQN vs Vanilla completed successfully!

REM =============================================================================
REM METHOD 4: EVOLUTIONARY vs VANILLA
REM =============================================================================

echo.
echo 📊 4/5 - EVOLUTIONARY vs VANILLA
echo =============================================================================

echo 🎯 Training: Evolutionary vs Vanilla
python experiments/run_training_testing.py --stage train ^
  --creditor_method evolutionary --debtor_method vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 80 --train_seed 42 ^
  --generations 10 --population_size 24 ^
  --mutation_rate 0.15 --crossover_rate 0.8

if %errorlevel% neq 0 (
    echo ❌ Evolutionary training failed
    pause
    exit /b 1
)

echo 🥊 Testing: Evolutionary vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test evolutionary --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Evolutionary testing failed
    pause
    exit /b 1
)

echo 🥊 Testing: Vanilla vs Evolutionary (reverse)
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test vanilla --debtor_method_test evolutionary ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Vanilla vs Evolutionary testing failed
    pause
    exit /b 1
)

echo ✅ Evolutionary vs Vanilla completed successfully!

REM =============================================================================
REM METHOD 5: HIERARCHICAL vs VANILLA
REM =============================================================================

echo.
echo 📊 5/5 - HIERARCHICAL vs VANILLA
echo =============================================================================

echo 🎯 Training: Hierarchical vs Vanilla
python experiments/run_training_testing.py --stage train ^
  --creditor_method hierarchical --debtor_method vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 80 --train_seed 42 ^
  --generations 10 --negotiations_per_gen 24 ^
  --learning_rate 0.6 --mutation_rate 0.12 --crossover_rate 0.8

if %errorlevel% neq 0 (
    echo ❌ Hierarchical training failed
    pause
    exit /b 1
)

echo 🥊 Testing: Hierarchical vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test hierarchical --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Hierarchical testing failed
    pause
    exit /b 1
)

echo 🥊 Testing: Vanilla vs Hierarchical (reverse)
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test vanilla --debtor_method_test hierarchical ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Vanilla vs Hierarchical testing failed
    pause
    exit /b 1
)

echo ✅ Hierarchical vs Vanilla completed successfully!

echo.
echo ================================================================================
echo 🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!
echo ================================================================================
echo.
echo 📊 EXPERIMENT SUMMARY:
echo   Sequential Train-Test Approach: 5 complete method evaluations
echo     ✅ Vanilla vs Vanilla (train → test)
echo     ✅ Q-Learning vs Vanilla (train → bidirectional test)
echo     ✅ DQN vs Vanilla (train → bidirectional test)
echo     ✅ Evolutionary vs Vanilla (train → bidirectional test)  
echo     ✅ Hierarchical vs Vanilla (train → bidirectional test)
echo.
echo   Testing Stage: 9 head-to-head comparisons completed
echo     ✅ Vanilla vs Vanilla (baseline)
echo     ✅ Q-Learning vs Vanilla + Vanilla vs Q-Learning
echo     ✅ DQN vs Vanilla + Vanilla vs DQN
echo     ✅ Evolutionary vs Vanilla + Vanilla vs Evolutionary
echo     ✅ Hierarchical vs Vanilla + Vanilla vs Hierarchical
echo.
echo 📁 RESULTS LOCATIONS:
echo   Training Results: results/training/
echo   Trained Models: trained_models/
echo   Testing Results: results/testing/
echo.
echo ⏱️ Total Runtime: %TIME%
echo ================================================================================

pause