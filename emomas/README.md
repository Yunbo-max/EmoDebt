# EmoMAS: Emotional Multi-Agent System

A sophisticated multi-agent system for emotional debt collection negotiations using various machine learning and AI approaches.

## 🎯 Overview

EmoMAS implements multiple advanced models for optimizing emotional strategies in debt collection scenarios:

- **Vanilla Baseline**: Simple two-agent negotiation without emotion optimization
- **Prompt-Based**: LLM negotiation with random emotion selection prompts
- **Bayesian Transition Optimization**: Dynamic emotion selection using Bayesian inference
- **GPT Orchestrator**: LLM-guided emotion coordination 
- **Baseline Evolutionary**: Genetic algorithm approach
- **Q-Learning**: Reinforcement learning baseline
- **DQN**: Deep Q-Network baseline
- **Hierarchical Bayesian**: Multi-level optimization
- **HMM + Game Theory**: Hidden Markov Models with game-theoretic payoffs

## 🏗️ Project Structure

```
emomas/
├── main.py                 # Main entry point
├── README.md              # This file
├── .env                   # Environment variables
├── config/
│   └── scenarios.json     # Negotiation scenarios
├── experiments/           # Experiment runners
│   ├── run_all_datasets.py    # Run on multiple datasets
│   ├── run_baseline.py        # Vanilla baseline model
│   ├── run_promptllm.py       # Prompt-based model
│   ├── run_multiagent.py      # Multi-agent experiments
│   └── run_multiagent_llm.py  # LLM-based experiments
├── llm/                   # LLM integration
│   ├── llm_wrapper.py     # LLM API wrapper
│   ├── negotiator.py      # Single negotiator
│   └── negotiator_multiagent.py # Multi-agent negotiator
├── models/                # ML/AI models
│   ├── base_model.py      # Base emotion model
│   ├── vanilla_model.py   # Vanilla baseline (no optimization)
│   ├── prompt_model.py    # Prompt-based (random emotion prompts)
│   ├── bayesian_multiagent.py    # Bayesian optimization
│   ├── baseline_evolutionary.py  # Evolutionary baseline
│   ├── dqn_baseline.py           # DQN implementation
│   ├── hierarchical_evolutionary.py # Hierarchical model
│   ├── hmm_game_theory.py        # HMM + Game Theory
│   ├── llm_multiagent.py         # LLM multi-agent
│   └── qlearning_baseline.py     # Q-Learning baseline
├── results/               # Experiment results
└── utils/                 # Utility functions
    ├── helpers.py         # General helpers
    └── preprocessing.py   # Data preprocessing
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate
cd emomas

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (OpenAI, etc.)
```

### 2. Basic Usage

```bash
# Run vanilla baseline (simple two-agent negotiation)
python experiments/run_baseline.py --scenarios 5 --iterations 10

# Run multi-dataset comparison with vanilla baseline
python experiments/run_all_datasets.py --dataset_type all --model_type vanilla --iterations 5
```

## 📊 Running Experiments

### Single Model Experiments

#### Vanilla Baseline (Recommended Starting Point)
```bash
# Simple baseline - no emotion optimization
python experiments/run_baseline.py --scenarios 3 --iterations 5

# Or via run_all_datasets
python experiments/run_all_datasets.py \
    --model_type vanilla \
    --iterations 5 \
    --scenarios 3 \
    --dataset_type debt
```

#### Prompt-Based (Random Emotion Prompts)
```bash
# Prompts creditor to use random emotions
python experiments/run_promptllm.py --scenarios 3 --iterations 5

# Or via run_all_datasets
python experiments/run_all_datasets.py \
    --model_type prompt \
    --iterations 5 \
    --scenarios 3 \
    --dataset_type debt
```

#### Bayesian Transition Optimization
```bash
python experiments/run_all_datasets.py \
    --model_type bayesian \
    --iterations 10 \
    --scenarios 5 \
    --dataset_type debt
```

#### GPT Orchestrator
```bash
python experiments/run_all_datasets.py \
    --model_type gpt \
    --iterations 5 \
    --model_creditor gpt-4o-mini \
    --model_debtor gpt-4o-mini
```

#### Baseline Evolutionary
```bash
python experiments/run_all_datasets.py \
    --model_type baseline \
    --iterations 10 \
    --scenarios 3
```

#### Q-Learning Baseline
```bash
python experiments/run_all_datasets.py \
    --model_type qlearning \
    --iterations 5 \
    --scenarios 3
```

#### Deep Q-Network (DQN)
```bash
python experiments/run_all_datasets.py \
    --model_type dqn \
    --iterations 5 \
    --scenarios 3
```

#### Hierarchical Bayesian
```bash
python experiments/run_all_datasets.py \
    --model_type hierarchical \
    --iterations 10 \
    --scenarios 5
```

#### HMM + Game Theory
```bash
python experiments/run_all_datasets.py \
    --model_type hmm \
    --iterations 8 \
    --scenarios 4
```

### Multi-Dataset Experiments

Run on all available datasets:

```bash
python experiments/run_all_datasets.py \
    --dataset_type all \
    --model_type bayesian \
    --iterations 5 \
    --scenarios 5 \
    --max_dialog_len 30
```

Available datasets:
- `debt`: Debt collection scenarios
- `disaster`: Disaster survivor negotiations  
- `student`: Student sleep schedule negotiations
- `medical`: Medical procedure negotiations
- `all`: Run on all datasets

### Advanced Usage

#### Custom Parameters
```bash
python experiments/run_all_datasets.py \
    --model_type bayesian \
    --iterations 15 \
    --scenarios 8 \
    --max_dialog_len 40 \
    --model_creditor gpt-4o \
    --model_debtor gpt-4o-mini \
    --debtor_emotion angry \
    --out_dir results/custom_experiment
```

#### Debug Mode
```bash
python experiments/run_all_datasets.py \
    --model_type hmm \
    --debug \
    --scenarios 2 \
    --iterations 3
```

## 🎛️ Configuration

### Command Line Parameters

#### Common Parameters
- `--model_type`: Model to use (`bayesian`, `gpt`, `baseline`, `qlearning`, `dqn`, `hierarchical`, `hmm`)
- `--dataset_type`: Dataset to use (`debt`, `disaster`, `student`, `medical`, `all`)
- `--iterations`: Number of iterations/generations per scenario
- `--scenarios`: Number of scenarios to test
- `--max_dialog_len`: Maximum dialog turns per negotiation

#### Model Parameters
- `--model_creditor`: LLM for creditor agent (default: `gpt-4o-mini`)
- `--model_debtor`: LLM for debtor agent (default: `gpt-4o-mini`)
- `--debtor_emotion`: Fixed debtor emotion (`happy`, `angry`, `sad`, `fear`, `neutral`)

#### Output Parameters
- `--out_dir`: Output directory for results
- `--debug`: Enable debug output

### Environment Variables (.env)

```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Other LLM providers (if needed)
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Experiment settings
DEFAULT_MODEL=gpt-4o-mini
DEBUG_MODE=false
```

## 📈 Results and Analysis

### Output Structure

Results are saved in the `results/` directory:

```
results/
├── debt_scenarios/              # Dataset-specific results
│   ├── bayesian_transition_TIMESTAMP.json
│   ├── scenarios.json          # Processed scenarios
│   └── summary_TIMESTAMP.txt   # Human-readable summary
├── multi_dataset/               # Cross-dataset comparisons
│   └── cross_dataset_summary_TIMESTAMP.json
└── model_specific/              # Individual model results
    ├── baseline/
    ├── qlearning/
    ├── dqn/
    └── ...
```

### Key Metrics

Each experiment tracks:
- **Success Rate**: Percentage of successful negotiations
- **Collection Rate**: Average amount collected
- **Negotiation Rounds**: Average dialog length
- **Emotion Effectiveness**: Performance per emotion type
- **Model Statistics**: Learning curves, convergence metrics

### Cross-Dataset Comparison

When running `--dataset_type all`, you get:

```
📊 CROSS-DATASET COMPARISON
Dataset         Success Rate    Collection Rate      Avg Rounds
----------------------------------------------------------------
debt           85.2%           0.745                12.3
disaster       72.8%           0.681                15.7
student        91.3%           0.832                8.9
medical        78.4%           0.697                13.2
```

## 🔧 Model Details

### Bayesian Transition Optimization
- Uses Bayesian inference for emotion selection
- Adapts based on negotiation outcomes
- Optimizes transition probabilities

### GPT Orchestrator  
- LLM-guided emotion coordination
- Multi-agent conversation management
- Context-aware response generation

### Baseline Models
- **Evolutionary**: Genetic algorithm optimization
- **Q-Learning**: Tabular reinforcement learning
- **DQN**: Deep reinforcement learning
- **Hierarchical**: Multi-level Bayesian optimization

### HMM + Game Theory
- Hidden Markov Models for state prediction
- Game-theoretic payoff matrices
- Strategic emotion selection

## 🛠️ Development

### Adding New Models

1. Create model in `models/new_model.py`:
```python
def run_new_model_experiment(scenarios, **kwargs):
    # Implementation
    return results
```

2. Add to `models/__init__.py`:
```python
from .new_model import run_new_model_experiment
```

3. Update `run_all_datasets.py` choices and logic

### Adding New Datasets

1. Place CSV file in `data/` directory
2. Update `dataset_paths` in `run_all_datasets.py`
3. Ensure CSV has required columns (amount, context, etc.)

### Custom Scenarios

Edit `config/scenarios.json` to add custom negotiation scenarios:

```json
{
  "scenarios": [
    {
      "id": "custom_1",
      "debt_amount": 5000,
      "debtor_context": "Lost job recently",
      "urgency": "high",
      "relationship": "first_contact"
    }
  ]
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Check the documentation
- Review the example outputs in `results/`

## 🔮 Future Work

- **Reinforcement Learning**: More RL approaches
- **Multi-Modal**: Voice/video emotion recognition
- **Real-Time**: Live negotiation systems
- **Explainability**: Model decision explanations
- **Scalability**: Distributed training support