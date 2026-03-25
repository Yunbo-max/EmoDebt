# Two-Stage Emotional Negotiation System

This system implements a two-stage approach for training and testing emotional negotiation models with **unified argument naming**:

- **AI Methods** (`creditor_method`, `debtor_method`): The negotiation strategies (evolutionary, dqn, hierarchical, qlearning, vanilla)
- **LLM Models** (`creditor_model`, `debtor_model`): The language models (deepseek-chat, gpt-4o-mini, claude-3-sonnet, etc.)

1. **Training Stage**: Train AI methods on scenarios and save their learned emotional policies
2. **Testing Stage**: Load trained AI methods and test them against each other with statistical analysis

## Available AI Methods

1. **Vanilla Method**: Pure LLM-to-LLM negotiation (no emotion optimization)
2. **Evolutionary Method**: Uses evolutionary algorithms to optimize emotional transition matrices
3. **DQN Method**: Deep Q-Network approach for learning emotional strategies
4. **Hierarchical Method**: Hierarchical Bayesian optimization with group-level emotion modeling
5. **Q-Learning Method**: Traditional Q-learning for emotional transition optimization

## Quick Start

### 1. Training Stage

Train AI methods against opponents:

```bash
# Train evolutionary method as creditor vs vanilla debtor using DeepSeek Chat
python experiments/run_training_testing.py --stage train --creditor_method evolutionary --debtor_method vanilla --creditor_model deepseek-chat --debtor_model deepseek-chat --scenarios 20 --generations 10

# Train hierarchical method vs Q-learning using GPT-4o-Mini
python experiments/run_training_testing.py --stage train --creditor_method hierarchical --debtor_method qlearning --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini --scenarios 20 --generations 10

# Train DQN method using mixed LLM models  
python experiments/run_training_testing.py --stage train --creditor_method dqn --debtor_method vanilla --creditor_model deepseek-chat --debtor_model gpt-4o-mini --scenarios 20 --episodes 100
```

Trained models are automatically saved to `trained_models/` directory with timestamps.

### 2. Testing Stage

Test trained AI methods against each other:

```bash
# Evolutionary vs Vanilla (both using DeepSeek Chat)
python experiments/run_training_testing.py --stage test --creditor_method_test evolutionary --debtor_method_test vanilla --creditor_model deepseek-chat --debtor_model deepseek-chat --test_scenarios 10 --iterations 5

# DQN vs Hierarchical (using GPT-4o-Mini)
python experiments/run_training_testing.py --stage test --creditor_method_test dqn --debtor_method_test hierarchical --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini --test_scenarios 10 --iterations 5

# Mixed setup: Q-Learning (DeepSeek) vs Vanilla (GPT)
python experiments/run_training_testing.py --stage test --creditor_method_test qlearning --debtor_method_test vanilla --creditor_model deepseek-chat --debtor_model gpt-4o-mini --test_scenarios 10 --iterations 5
```

### 3. Full Tournament

Run all models against each other:

```bash
# Run complete tournament
python run_tournament.py --dataset_type debt --test_scenarios 10 --iterations 5

# Custom tournament with specific models
python run_tournament.py --models vanilla evolutionary dqn --test_scenarios 5 --iterations 3
```

### 4. Quick Test

Test that everything is working:

```bash
python quick_test.py
```

## Batch Automation

Use the provided batch script to run complete experiments:

```bash
# Windows
run_two_stage_experiments.bat

# Linux/Mac - create similar .sh script
```

## Directory Structure

```
results/
├── training/           # Training results and saved models
│   ├── evolutionary/
│   ├── dqn/
│   ├── hierarchical/
│   └── qlearning/
├── testing/           # Head-to-head testing results
│   ├── vanilla_vs_evolutionary/
│   ├── dqn_vs_hierarchical/
│   └── ...
├── tournament_*/      # Tournament results
└── ...

trained_models/        # Saved model policies
├── evolutionary_20250103_142030.json
├── dqn_20250103_142045.json
└── ...
```

## Key Features

### Training Stage
- **Model-specific parameters**: Each model type supports its own hyperparameters
- **Automatic saving**: Trained policies are automatically saved with timestamps
- **Progress tracking**: Real-time progress updates and performance metrics
- **Statistical analysis**: 95% confidence intervals for all metrics

### Testing Stage
- **Model-agnostic**: Any trained model can compete against any other
- **Scenario randomization**: Different scenarios used for testing vs training
- **Head-to-head analysis**: Direct comparison between model pairs
- **Statistical significance**: Bootstrap confidence intervals and significance tests

### Tournament System
- **Round-robin format**: Every model competes against every other model
- **Both roles**: Each model plays both creditor and debtor roles
- **Leaderboard**: Ranked by average success rate across all matches
- **Head-to-head matrix**: Detailed win/loss tracking between specific models

## Parameters

### Training Parameters
- `--scenarios`: Number of scenarios for training
- `--generations`: Evolutionary generations (evolutionary, hierarchical)
- `--population_size`: Population size for evolutionary models
- `--mutation_rate`: Mutation rate for evolutionary algorithms
- `--crossover_rate`: Crossover rate for evolutionary algorithms
- `--episodes`: Training episodes (DQN, Q-learning)

### Testing Parameters
- `--test_scenarios`: Number of scenarios for testing
- `--iterations`: Iterations per scenario
- `--creditor_model`: Model playing creditor role
- `--debtor_model`: Model playing debtor role

### LLM Parameters
- `--model_creditor`: LLM model for creditor agent (default: gpt-4o-mini)
- `--model_debtor`: LLM model for debtor agent (default: gpt-4o-mini)
- `--max_dialog_len`: Maximum dialog turns per negotiation

## Dataset Support

Supports multiple datasets:
- `debt`: Credit recovery scenarios
- `disaster`: Disaster survivor scenarios  
- `student`: Education/sleep scenarios
- `medical`: Hospital/surgery scenarios

## Output Analysis

All results include:
- **Success rates** with 95% confidence intervals
- **Collection efficiency** (days to resolution)
- **Negotiation efficiency** (rounds to completion)
- **Statistical significance** tests
- **Detailed breakdowns** by scenario type

## Example Workflow

1. **Train all models**:
   ```bash
   python experiments/run_training_testing.py --stage train --model_type evolutionary --scenarios 30
   python experiments/run_training_testing.py --stage train --model_type dqn --scenarios 30
   python experiments/run_training_testing.py --stage train --model_type hierarchical --scenarios 30
   python experiments/run_training_testing.py --stage train --model_type qlearning --scenarios 30
   ```

2. **Run cross-model tests**:
   ```bash
   # Test best learners vs vanilla
   python experiments/run_training_testing.py --stage test --creditor_model evolutionary --debtor_model vanilla --test_scenarios 15
   python experiments/run_training_testing.py --stage test --creditor_model hierarchical --debtor_model vanilla --test_scenarios 15
   
   # Test learners vs learners
   python experiments/run_training_testing.py --stage test --creditor_model evolutionary --debtor_model dqn --test_scenarios 15
   python experiments/run_training_testing.py --stage test --creditor_model hierarchical --debtor_model qlearning --test_scenarios 15
   ```

3. **Run full tournament**:
   ```bash
   python run_tournament.py --test_scenarios 20 --iterations 10
   ```

This creates a comprehensive evaluation of which emotional negotiation strategies work best in different contexts, with rigorous statistical analysis to support the findings.