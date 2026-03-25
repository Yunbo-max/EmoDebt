# Two-Stage Emotional Negotiation System - Complete Usage Guide

This guide shows you how to use method-specific parameters for training and testing different AI negotiation models.

## 🚀 Quick Start

### Basic Training Command Structure
```bash
python experiments/run_training_testing.py --stage train \
  --creditor_method <METHOD> \
  --debtor_method <OPPONENT> \
  --creditor_model <LLM> \
  --debtor_model <LLM> \
  [METHOD_SPECIFIC_PARAMS]
```

### Basic Testing Command Structure  
```bash
python experiments/run_training_testing.py --stage test \
  --creditor_method_test <METHOD1> \
  --debtor_method_test <METHOD2> \
  --creditor_model <LLM> \
  --debtor_model <LLM> \
  [TESTING_PARAMS]
```

## 📋 Available Methods

| Method | Type | Description |
|--------|------|-------------|
| `vanilla` | Baseline | No learning - direct LLM negotiation |
| `evolutionary` | Optimization | Genetic algorithm emotion optimization |
| `hierarchical` | Bayesian | Hierarchical Bayesian emotion learning |
| `dqn` | Reinforcement Learning | Deep Q-Network emotion selection |
| `qlearning` | Reinforcement Learning | Q-Learning emotion strategies |

## 🧬 Evolutionary Method

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--generations` | 10 | Number of evolutionary generations |
| `--population_size` | 20 | Negotiations per generation |
| `--mutation_rate` | 0.1 | Probability of emotion mutation |
| `--crossover_rate` | 0.7 | Genetic crossover probability |
| `--elite_size` | 5 | Top sequences to keep |

### Training Examples
```bash
# Quick test (fast)
python experiments/run_training_testing.py --stage train \
  --creditor_method evolutionary --debtor_method vanilla \
  --creditor_model deepseek-chat --debtor_model deepseek-chat \
  --scenarios 3 --generations 5 --population_size 10

# Standard training  
python experiments/run_training_testing.py --stage train \
  --creditor_method evolutionary --debtor_method vanilla \
  --creditor_model deepseek-chat --debtor_model deepseek-chat \
  --scenarios 20 --generations 15 --population_size 30 \
  --mutation_rate 0.15 --crossover_rate 0.8

# Intensive training (slow but thorough)
python experiments/run_training_testing.py --stage train \
  --creditor_method evolutionary --debtor_method vanilla \
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini \
  --scenarios 50 --generations 25 --population_size 50 \
  --mutation_rate 0.1 --crossover_rate 0.7 --elite_size 10
```

## 🎯 Hierarchical Bayesian Method

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--generations` | 10 | Number of evolutionary generations |
| `--negotiations_per_gen` | 20 | Negotiations per generation (like population_size) |
| `--mutation_rate` | 0.15 | Higher mutation for exploration |
| `--crossover_rate` | 0.7 | Genetic crossover probability |
| `--learning_rate` | 0.6 | Bayesian learning rate (lambda) |

### Training Examples
```bash
# Quick test
python experiments/run_training_testing.py --stage train \
  --creditor_method hierarchical --debtor_method vanilla \
  --creditor_model deepseek-chat --debtor_model deepseek-chat \
  --scenarios 3 --generations 5 --negotiations_per_gen 10

# Standard training
python experiments/run_training_testing.py --stage train \
  --creditor_method hierarchical --debtor_method vanilla \
  --creditor_model deepseek-chat --debtor_model deepseek-chat \
  --scenarios 20 --generations 15 --negotiations_per_gen 25 \
  --learning_rate 0.7 --mutation_rate 0.12

# High-learning training
python experiments/run_training_testing.py --stage train \
  --creditor_method hierarchical --debtor_method qlearning \
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini \
  --scenarios 30 --generations 20 --negotiations_per_gen 40 \
  --learning_rate 0.8 --mutation_rate 0.1 --crossover_rate 0.8
```

## 🧠 DQN (Deep Q-Network) Method

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--episodes` | 200 | Total training episodes |
| `--episodes_per_scenario` | 5 | Episodes per scenario before cycling |
| `--learning_rate` | 1e-4 | Neural network learning rate |
| `--discount_factor` | 0.95 | Future reward discount (gamma) |
| `--exploration_rate` | 1.0 | Initial exploration (epsilon) |
| `--exploration_decay` | 0.995 | Exploration decay rate |
| `--batch_size` | 32 | Training batch size |
| `--replay_buffer_size` | 10000 | Experience replay buffer |

### Training Examples
```bash
# Quick test
python experiments/run_training_testing.py --stage train \
  --creditor_method dqn --debtor_method vanilla \
  --creditor_model deepseek-chat --debtor_model deepseek-chat \
  --scenarios 3 --episodes 50 --episodes_per_scenario 3

# Standard training
python experiments/run_training_testing.py --stage train \
  --creditor_method dqn --debtor_method vanilla \
  --creditor_model deepseek-chat --debtor_model deepseek-chat \
  --scenarios 20 --episodes 300 --episodes_per_scenario 5 \
  --learning_rate 5e-4 --exploration_rate 0.9

# Deep training
python experiments/run_training_testing.py --stage train \
  --creditor_method dqn --debtor_method evolutionary \
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini \
  --scenarios 40 --episodes 500 --episodes_per_scenario 8 \
  --learning_rate 1e-3 --batch_size 64 --replay_buffer_size 20000
```

## 📊 Q-Learning Method

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--episodes` | 100 | Total training episodes |
| `--episodes_per_scenario` | 5 | Episodes per scenario |
| `--learning_rate` | 0.1 | Q-learning rate (alpha) |
| `--discount_factor` | 0.9 | Future reward discount (gamma) |
| `--exploration_rate` | 1.0 | Initial exploration (epsilon) |
| `--exploration_decay` | 0.995 | Exploration decay rate |
| `--temperature` | 1.0 | Softmax exploration temperature |

### Training Examples
```bash
# Quick test
python experiments/run_training_testing.py --stage train \
  --creditor_method qlearning --debtor_method vanilla \
  --creditor_model deepseek-chat --debtor_model deepseek-chat \
  --scenarios 3 --episodes 30 --episodes_per_scenario 3

# Standard training
python experiments/run_training_testing.py --stage train \
  --creditor_method qlearning --debtor_method vanilla \
  --creditor_model deepseek-chat --debtor_model deepseek-chat \
  --scenarios 20 --episodes 150 --learning_rate 0.15 \
  --exploration_rate 0.8 --temperature 1.2

# Thorough training
python experiments/run_training_testing.py --stage train \
  --creditor_method qlearning --debtor_method hierarchical \
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini \
  --scenarios 35 --episodes 250 --episodes_per_scenario 7 \
  --learning_rate 0.2 --discount_factor 0.95
```

## 🎮 Testing Trained Models

### Automatic Model Loading (Recommended)
```bash
# Test most recent evolutionary vs hierarchical
python experiments/run_training_testing.py --stage test \
  --creditor_method_test evolutionary --debtor_method_test hierarchical \
  --creditor_model deepseek-chat --debtor_model deepseek-chat \
  --test_scenarios 15 --iterations 8

# Test trained DQN vs trained Q-Learning
python experiments/run_training_testing.py --stage test \
  --creditor_method_test dqn --debtor_method_test qlearning \
  --creditor_model gpt-4o-mini --debtor_model gpt-4o-mini \
  --test_scenarios 20 --iterations 10
```

### Manual Model File Loading
```bash
# Test specific model versions
python experiments/run_training_testing.py --stage test \
  --creditor_method_test evolutionary --debtor_method_test vanilla \
  --creditor_model_file "trained_models/evolutionary_as_creditor_vs_vanilla_20250103_143025.json" \
  --creditor_model deepseek-chat --debtor_model deepseek-chat \
  --test_scenarios 10 --iterations 5
```

## 📁 File Organization

### Training Output
```
results/training/
├── evolutionary_vs_vanilla/           # Training experiment results  
├── hierarchical_vs_qlearning/         # Another training combo
└── dqn_vs_vanilla/                    # DQN training results

trained_models/                        # Saved trained models
├── evolutionary_as_creditor_vs_vanilla_20250103_143025.json
├── hierarchical_as_creditor_vs_vanilla_20250103_143030.json
└── dqn_as_creditor_vs_vanilla_20250103_143045.json
```

### Testing Output
```
results/testing/
├── evolutionary_vs_hierarchical/      # Head-to-head competition
├── dqn_vs_qlearning/                 # RL vs RL competition
└── vanilla_vs_evolutionary/           # Baseline vs optimized
```

## 🎯 Parameter Tuning Tips

### For Speed (Quick Testing)
- Use fewer `--scenarios` (3-5)
- Use fewer `--generations`/`--episodes` (5-10)
- Use smaller `--population_size` (10-15)

### For Quality (Production)
- Use more `--scenarios` (20-50)
- Use more `--generations`/`--episodes` (15-30)
- Use larger `--population_size` (30-50)

### For Different Opponents
- Train against `vanilla` for baseline learning
- Train against other methods for competitive learning
- Mix opponents for robust training

## 🔧 Advanced Combinations

### Multi-Method Tournament Training
```bash
# Train each method against multiple opponents
python experiments/run_training_testing.py --stage train --creditor_method evolutionary --debtor_method vanilla --scenarios 15 --generations 12
python experiments/run_training_testing.py --stage train --creditor_method evolutionary --debtor_method qlearning --scenarios 15 --generations 12
python experiments/run_training_testing.py --stage train --creditor_method hierarchical --debtor_method vanilla --scenarios 15 --generations 12
python experiments/run_training_testing.py --stage train --creditor_method hierarchical --debtor_method evolutionary --scenarios 15 --generations 12
```

### Cross-Method Testing
```bash
# Test all combinations
python experiments/run_training_testing.py --stage test --creditor_method_test evolutionary --debtor_method_test hierarchical --test_scenarios 10
python experiments/run_training_testing.py --stage test --creditor_method_test hierarchical --debtor_method_test evolutionary --test_scenarios 10
python experiments/run_training_testing.py --stage test --creditor_method_test dqn --debtor_method_test qlearning --test_scenarios 10
```

## ⚠️ Common Issues

### Parameter Mismatches
- `evolutionary` and `hierarchical` use `--generations`
- `dqn` and `qlearning` use `--episodes` 
- Don't mix these parameters

### Model Loading Errors
- If testing fails to find models, check `trained_models/` directory
- Use specific `--creditor_model_file` if auto-loading fails
- Make sure you trained the methods before testing them

### LLM Model Names
- Use exact model names: `deepseek-chat`, `gpt-4o-mini`
- Check your .env file for API keys
- Different models may have different response patterns

This guide gives you everything you need to run sophisticated emotional negotiation experiments! 🚀