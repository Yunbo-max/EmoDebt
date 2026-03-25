@echo off
REM Complete Tournament: All Advanced Methods vs All Advanced Methods
REM Methods: qlearning, dqn, evolutionary, hierarchical
REM Sequential Structure: Train → Test → Train → Test (alternating)
REM Training: 12 sessions | Testing: 16 games (4×4 matrix)

@REM echo ================================================================================
@REM echo 🏆 COMPLETE TOURNAMENT: ALL ADVANCED METHODS vs ALL ADVANCED METHODS
@REM echo ================================================================================
@REM echo Sequential Structure: Train → Test → Train → Test (alternating approach)
@REM echo Methods: qlearning, dqn, evolutionary, hierarchical (no vanilla)
@REM echo Training Sessions: 12 | Testing Games: 16
@REM echo ================================================================================

REM =============================================================================
REM SESSION 1: Q-LEARNING vs DQN
REM =============================================================================

echo.
echo 📊 SESSION 1/12: Q-LEARNING vs DQN
echo =============================================================================

echo 🎯 Training 1/12: Q-Learning vs Vanilla
python experiments/run_training_testing.py --stage train ^
  --creditor_method qlearning --debtor_method vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 80 --train_seed 42 ^
  --episodes 240 --episodes_per_scenario 3 ^
  --learning_rate 0.15 --discount_factor 0.9 ^
  --exploration_rate 0.8 --exploration_decay 0.995 ^
  --temperature 1.2

if %errorlevel% neq 0 (
    echo ❌ Q-Learning training failed
    pause
    exit /b 1
)

echo 🥊 Testing 1/16: Q-Learning vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test qlearning --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Q-Learning vs DQN testing failed
    pause
    exit /b 1
)

echo ✅ Session 1 completed successfully!

REM =============================================================================
REM SESSION 2: Q-LEARNING vs EVOLUTIONARY
REM =============================================================================

echo.
echo 📊 SESSION 2/12: Q-LEARNING vs EVOLUTIONARY
echo =============================================================================

echo 🎯 Training 2/12: Q-Learning vs Vanilla
python experiments/run_training_testing.py --stage train ^
  --creditor_method qlearning --debtor_method vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 80 --train_seed 42 ^
  --episodes 240 --episodes_per_scenario 3 ^
  --learning_rate 0.15 --discount_factor 0.9 ^
  --exploration_rate 0.8 --exploration_decay 0.995 ^
  --temperature 1.2

if %errorlevel% neq 0 (
    echo ❌ Q-Learning training failed
    pause
    exit /b 1
)

echo 🥊 Testing 2/16: Q-Learning vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test qlearning --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Q-Learning vs Evolutionary testing failed
    pause
    exit /b 1
)

echo ✅ Session 2 completed successfully!

REM =============================================================================
REM SESSION 3: Q-LEARNING vs HIERARCHICAL
REM =============================================================================

echo.
echo 📊 SESSION 3/12: Q-LEARNING vs HIERARCHICAL
echo =============================================================================

echo 🎯 Training 3/12: Q-Learning vs Vanilla
python experiments/run_training_testing.py --stage train ^
  --creditor_method qlearning --debtor_method vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 80 --train_seed 42 ^
  --episodes 240 --episodes_per_scenario 3 ^
  --learning_rate 0.15 --discount_factor 0.9 ^
  --exploration_rate 0.8 --exploration_decay 0.995 ^
  --temperature 1.2

if %errorlevel% neq 0 (
    echo ❌ Q-Learning training failed
    pause
    exit /b 1
)

echo 🥊 Testing 3/16: Q-Learning vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test qlearning --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Q-Learning vs Hierarchical testing failed
    pause
    exit /b 1
)

echo ✅ Session 3 completed successfully!

REM =============================================================================
REM SESSION 4: DQN vs Q-LEARNING
REM =============================================================================

echo.
echo 📊 SESSION 4/12: DQN vs Q-LEARNING
echo =============================================================================

echo 🎯 Training 4/12: DQN vs Vanilla
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

echo 🥊 Testing 4/16: DQN vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test dqn --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ DQN vs Q-Learning testing failed
    pause
    exit /b 1
)

echo ✅ Session 4 completed successfully!

REM =============================================================================
REM SESSION 5: DQN vs EVOLUTIONARY
REM =============================================================================

echo.
echo 📊 SESSION 5/12: DQN vs EVOLUTIONARY
echo =============================================================================

echo 🎯 Training 5/12: DQN vs Vanilla
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

echo 🥊 Testing 5/16: DQN vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test dqn --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ DQN vs Evolutionary testing failed
    pause
    exit /b 1
)

echo ✅ Session 5 completed successfully!

REM =============================================================================
REM SESSION 6: DQN vs HIERARCHICAL
REM =============================================================================

echo.
echo 📊 SESSION 6/12: DQN vs HIERARCHICAL
echo =============================================================================

echo 🎯 Training 6/12: DQN vs Vanilla
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

echo 🥊 Testing 6/16: DQN vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test dqn --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ DQN vs Hierarchical testing failed
    pause
    exit /b 1
)

echo ✅ Session 6 completed successfully!
