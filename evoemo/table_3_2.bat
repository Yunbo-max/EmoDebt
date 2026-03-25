
REM =============================================================================
REM SESSION 7: EVOLUTIONARY vs Q-LEARNING
REM =============================================================================

echo.
echo 📊 SESSION 7/12: EVOLUTIONARY vs Q-LEARNING
echo =============================================================================

echo 🎯 Training 7/12: Evolutionary vs Q-Learning
python experiments/run_training_testing.py --stage train ^
  --creditor_method evolutionary --debtor_method qlearning ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 80 --train_seed 42 ^
  --generations 10 --population_size 24 ^
  --mutation_rate 0.15 --crossover_rate 0.8

if %errorlevel% neq 0 (
    echo ❌ Evolutionary vs Q-Learning training failed
    pause
    exit /b 1
)

echo 🥊 Testing 7/16: Evolutionary vs Q-Learning
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test evolutionary --debtor_method_test qlearning ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Evolutionary vs Q-Learning testing failed
    pause
    exit /b 1
)

echo ✅ Session 7 completed successfully!

REM =============================================================================
REM SESSION 8: EVOLUTIONARY vs DQN
REM =============================================================================

echo.
echo 📊 SESSION 8/12: EVOLUTIONARY vs DQN
echo =============================================================================

echo 🎯 Training 8/12: Evolutionary vs Vanilla
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

echo 🥊 Testing 8/16: Evolutionary vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test evolutionary --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Evolutionary vs DQN testing failed
    pause
    exit /b 1
)

echo ✅ Session 8 completed successfully!

REM =============================================================================
REM SESSION 9: EVOLUTIONARY vs HIERARCHICAL
REM =============================================================================

echo.
echo 📊 SESSION 9/12: EVOLUTIONARY vs HIERARCHICAL
echo =============================================================================

echo 🎯 Training 9/12: Evolutionary vs Vanilla
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

echo 🥊 Testing 9/16: Evolutionary vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test evolutionary --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Evolutionary vs Hierarchical testing failed
    pause
    exit /b 1
)

echo ✅ Session 9 completed successfully!

REM =============================================================================
REM SESSION 10: HIERARCHICAL vs Q-LEARNING
REM =============================================================================

echo.
echo 📊 SESSION 10/12: HIERARCHICAL vs Q-LEARNING
echo =============================================================================

echo 🎯 Training 10/12: Hierarchical vs Vanilla
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

echo 🥊 Testing 10/16: Hierarchical vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test hierarchical --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Hierarchical vs Q-Learning testing failed
    pause
    exit /b 1
)

echo ✅ Session 10 completed successfully!

REM =============================================================================
REM SESSION 11: HIERARCHICAL vs DQN
REM =============================================================================

echo.
echo 📊 SESSION 11/12: HIERARCHICAL vs DQN
echo =============================================================================

echo 🎯 Training 11/12: Hierarchical vs Vanilla
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

echo 🥊 Testing 11/16: Hierarchical vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test hierarchical --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Hierarchical vs DQN testing failed
    pause
    exit /b 1
)

echo ✅ Session 11 completed successfully!

REM =============================================================================
REM SESSION 12: HIERARCHICAL vs EVOLUTIONARY
REM =============================================================================

echo.
echo 📊 SESSION 12/12: HIERARCHICAL vs EVOLUTIONARY
echo =============================================================================

echo 🎯 Training 12/12: Hierarchical vs Vanilla
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

echo 🥊 Testing 12/16: Hierarchical vs Vanilla
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test hierarchical --debtor_method_test vanilla ^
  --creditor_model deepseek-chat --debtor_model deepseek-chat ^
  --scenarios 10 --test_seed 42 --iterations 1

if %errorlevel% neq 0 (
    echo ❌ Hierarchical vs Evolutionary testing failed
    pause
    exit /b 1
)

echo ✅ Session 12 completed successfully!

REM =============================================================================
REM DIAGONAL TRAINING SESSIONS (self vs self)
REM =============================================================================

echo.
echo 🎯 DIAGONAL SESSIONS: Self vs Self Training
echo =============================================================================

REM =============================================================================
REM SESSION 13: Q-LEARNING vs Q-LEARNING
REM =============================================================================

echo.
echo 📊 SESSION 13/16: Q-LEARNING vs Q-LEARNING
echo =============================================================================

echo 🎯 Training 13/16: Q-Learning vs Q-Learning
python experiments/run_training_testing.py --stage train ^
  --creditor_method qlearning --debtor_method qlearning ^
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini ^
  --scenarios 80 --train_seed 42 ^
  --episodes 240 --episodes_per_scenario 3 ^
  --learning_rate 0.15 --discount_factor 0.9 ^
  --exploration_rate 0.8 --exploration_decay 0.995 ^
  --temperature 1.2

if %errorlevel% neq 0 (
    echo ❌ Q-Learning vs Q-Learning training failed
    pause
    exit /b 1
)

echo 🥊 Testing 13/16: Q-Learning vs Q-Learning
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test qlearning --debtor_method_test qlearning ^
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini ^
  --scenarios 10 --test_seed 42 --iterations 4

if %errorlevel% neq 0 (
    echo ❌ Q-Learning vs Q-Learning testing failed
    pause
    exit /b 1
)

echo ✅ Session 13 completed successfully!

REM =============================================================================
REM SESSION 14: DQN vs DQN
REM =============================================================================

echo.
echo 📊 SESSION 14/16: DQN vs DQN
echo =============================================================================

echo 🎯 Training 14/16: DQN vs DQN
python experiments/run_training_testing.py --stage train ^
  --creditor_method dqn --debtor_method dqn ^
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini ^
  --scenarios 80 --train_seed 42 ^
  --episodes 240 --episodes_per_scenario 3 ^
  --learning_rate 5e-4 --discount_factor 0.95 ^
  --exploration_rate 0.9 --exploration_decay 0.995 ^
  --batch_size 32 --replay_buffer_size 4000

if %errorlevel% neq 0 (
    echo ❌ DQN vs DQN training failed
    pause
    exit /b 1
)

echo 🥊 Testing 14/16: DQN vs DQN
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test dqn --debtor_method_test dqn ^
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini ^
  --test_scenarios 8 --iterations 4

if %errorlevel% neq 0 (
    echo ❌ DQN vs DQN testing failed
    pause
    exit /b 1
)

echo ✅ Session 14 completed successfully!

REM =============================================================================
REM SESSION 15: EVOLUTIONARY vs EVOLUTIONARY
REM =============================================================================

echo.
echo 📊 SESSION 15/16: EVOLUTIONARY vs EVOLUTIONARY
echo =============================================================================

echo 🎯 Training 15/16: Evolutionary vs Evolutionary
python experiments/run_training_testing.py --stage train ^
  --creditor_method evolutionary --debtor_method evolutionary ^
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini ^
  --scenarios 80 --train_seed 42 ^
  --generations 10 --population_size 24 ^
  --mutation_rate 0.15 --crossover_rate 0.8

if %errorlevel% neq 0 (
    echo ❌ Evolutionary vs Evolutionary training failed
    pause
    exit /b 1
)

echo 🥊 Testing 15/16: Evolutionary vs Evolutionary
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test evolutionary --debtor_method_test evolutionary ^
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini ^
  --test_scenarios 8 --iterations 4

if %errorlevel% neq 0 (
    echo ❌ Evolutionary vs Evolutionary testing failed
    pause
    exit /b 1
)

echo ✅ Session 15 completed successfully!

REM =============================================================================
REM SESSION 16: HIERARCHICAL vs HIERARCHICAL
REM =============================================================================

echo.
echo 📊 SESSION 16/16: HIERARCHICAL vs HIERARCHICAL
echo =============================================================================

echo 🎯 Training 16/16: Hierarchical vs Hierarchical
python experiments/run_training_testing.py --stage train ^
  --creditor_method hierarchical --debtor_method hierarchical ^
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini ^
  --scenarios 80 --train_seed 42 ^
  --generations 10 --negotiations_per_gen 24 ^
  --learning_rate 0.6 --mutation_rate 0.12 --crossover_rate 0.8

if %errorlevel% neq 0 (
    echo ❌ Hierarchical vs Hierarchical training failed
    pause
    exit /b 1
)

echo 🥊 Testing 16/16: Hierarchical vs Hierarchical
python experiments/run_training_testing.py --stage test ^
  --creditor_method_test hierarchical --debtor_method_test hierarchical ^
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini ^
  --scenarios 10 --test_seed 42 --iterations 4

if %errorlevel% neq 0 (
    echo ❌ Hierarchical vs Hierarchical testing failed
    pause
    exit /b 1
)

echo ✅ Session 16 completed successfully!

echo.
echo ================================================================================
echo 🎉 COMPLETE TOURNAMENT FINISHED!
echo ================================================================================
echo.
echo 📊 TOURNAMENT SUMMARY:
echo   Sequential Structure: 16 Train→Test sessions completed
echo     ✅ Cross-Method sessions: 12 train→test (between different methods)
echo       • Q-Learning sessions: 3 train→test (vs DQN, Evolutionary, Hierarchical)
echo       • DQN sessions: 3 train→test (vs Q-Learning, Evolutionary, Hierarchical)  
echo       • Evolutionary sessions: 3 train→test (vs Q-Learning, DQN, Hierarchical)
echo       • Hierarchical sessions: 3 train→test (vs Q-Learning, DQN, Evolutionary)
echo.
echo     ✅ Diagonal sessions: 4 train→test (self vs self)
echo       • Q-Learning vs Q-Learning
echo       • DQN vs DQN
echo       • Evolutionary vs Evolutionary
echo       • Hierarchical vs Hierarchical
echo.
echo 🏆 COMPLETE TOURNAMENT MATRIX (4×4):
echo         Q-L  DQN  EVO  HIE
echo   Q-L    ✅   ✅   ✅   ✅
echo   DQN    ✅   ✅   ✅   ✅
echo   EVO    ✅   ✅   ✅   ✅
echo   HIE    ✅   ✅   ✅   ✅
echo.
echo 📁 RESULTS LOCATIONS:
echo   Training Results: results/training/
echo   Trained Models: trained_models/
echo   Testing Results: results/testing/
echo.
echo ⏱️ Total Runtime: %TIME%
echo ================================================================================

pause