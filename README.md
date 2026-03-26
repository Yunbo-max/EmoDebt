# EmoDebt: Bayesian-Optimized Emotional Intelligence for Strategic Agent-to-Agent Debt Recovery

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2503.21080-b31b1b.svg)](https://arxiv.org/abs/2503.21080)
[![AAMAS 2026](https://img.shields.io/badge/AAMAS-2026-green.svg)](https://cyprusconferences.org/aamas2026/accepted-research-track/)
[![Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-yellow.svg)](https://huggingface.co/spaces/humanlong/EmoDebt)
[![Paper](https://img.shields.io/badge/🤗%20Hugging%20Face-Paper-orange.svg)](https://huggingface.co/papers/2503.21080)

> **Accepted at AAMAS 2026** — 25th International Conference on Autonomous Agents and Multiagent Systems, May 25-29, 2026, Paphos, Cyprus.

![EmoDebt Framework](Framework.png)

**EmoDebt** is an advanced AI system that leverages Bayesian optimization and emotional intelligence for automated debt collection negotiations. The system uses sophisticated machine learning techniques to optimize creditor-debtor interactions, improving recovery rates while maintaining professional and ethical standards.

## 🌟 Key Features

- 🧠 **Bayesian Emotional Optimization**: Uses Gaussian Processes to learn optimal Markovian transition matrices between 7 emotional states (happy, angry, sad, fear, disgust, surprise, neutral)
- 🎭 **Dynamic Emotional Adaptation**: Creditor agents dynamically adjust negotiation strategies based on debtor's emotional profile and responses
- ⚖️ **Strategic Concession Patterns**: Implements psychologically-grounded negotiation tactics with temporal payment constraints
- 📊 **Online Learning**: Continuously improves emotional strategies through reinforcement learning from negotiation outcomes
- 🤖 **Multi-Model Support**: Compatible with GPT-4o-mini, GPT-5-mini, Claude, and other state-of-the-art LLMs via LangGraph
- 🔄 **Real-time Adaptation**: Adjusts emotional strategies mid-negotiation based on debtor responses
- 📈 **Performance Analytics**: Comprehensive metrics including success rates, collection efficiency, and emotional convergence

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (or other supported LLM provider)
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Yunbo-max/EmoDebt.git
   cd EmoDebt
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   # Required API keys
   OPENAI_API_KEY=your_openai_api_key_here

   # Optional: Other LLM providers
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```

4. **Verify installation**:
   ```bash
   python main.py --model evolutionary --scenarios 1
   ```

## 📚 Usage Guide

### Run EmoDebt Bayesian Optimization

```bash
python experiments/run_emodebt.py \
    --iterations 10 \
    --negotiations 3 \
    --model_creditor gpt-4o-mini \
    --model_debtor gpt-4o-mini \
    --scenarios 50
```

### Run Baseline Experiments

```bash
python experiments/run_baseline.py \
    --generations 10 \
    --population_size 20 \
    --model_creditor gpt-4o-mini \
    --model_debtor gpt-4o-mini
```

### Run Across All Datasets

```bash
python experiments/run_all_datasets.py \
    --model_type bayesian \
    --iterations 10 \
    --scenarios 50 \
    --debtor_emotion neutral
```

**Core Parameters:**

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `--model_creditor` | LLM for creditor agent | `gpt-4o-mini`, `gpt-5-mini`, `claude-3-sonnet` | `gpt-4o-mini` |
| `--model_debtor` | LLM for debtor agent | `gpt-4o-mini`, `gpt-5-mini`, `claude-3-sonnet` | `gpt-4o-mini` |
| `--debtor_emotion` | Debtor emotional profile | `happy`, `angry`, `sad`, `fear`, `disgust`, `surprise`, `neutral` | `neutral` |
| `--iterations` | Bayesian learning cycles | 1-50 | `20` |
| `--scenarios` | Number of debt cases | 1-100 | `3` |

## 📊 Performance Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Success Rate (SR)** | Successful / Total x 100% | % of negotiations reaching agreement |
| **Collection Efficiency (CE)** | Final Timeline / Target Days | Ratio of actual vs. target timeline (lower = better) |
| **Negotiation Speed (NS)** | Total dialogue turns | Efficiency of the negotiation process (lower = better) |

## 🏗️ System Architecture

### Emotional States

The system models 7 distinct emotional states based on Ekman's basic emotions:

| Emotion | Description | Strategic Use |
|---------|-------------|---------------|
| **Happy** | Optimistic, positive tone | Building rapport, encouraging cooperation |
| **Angry** | Firm, assertive approach | Emphasizing urgency, creating pressure |
| **Sad** | Empathetic, understanding | Acknowledging debtor difficulties |
| **Fear** | Cautious, concerned tone | Highlighting consequences |
| **Disgust** | Disappointed, professional | Expressing concern about situation |
| **Surprise** | Engaging, unexpected | Introducing creative solutions |
| **Neutral** | Balanced, fact-focused | Professional baseline approach |

### Psychological Priors (Transition Matrix P⁰)

Initialized from Thornton & Tamir (2017) and Sun et al. (2023):

| From \ To | Happy | Surprise | Angry | Sad | Disgust | Fear | Neutral |
|-----------|-------|----------|-------|-----|---------|------|---------|
| **Happy** | 0.30 | 0.15 | 0.05 | 0.10 | 0.05 | 0.05 | 0.30 |
| **Surprise** | 0.20 | 0.20 | 0.15 | 0.10 | 0.10 | 0.10 | 0.15 |
| **Angry** | 0.10 | 0.10 | 0.25 | 0.15 | 0.15 | 0.10 | 0.15 |
| **Sad** | 0.15 | 0.10 | 0.10 | 0.20 | 0.10 | 0.15 | 0.20 |
| **Disgust** | 0.10 | 0.15 | 0.20 | 0.15 | 0.15 | 0.10 | 0.15 |
| **Fear** | 0.15 | 0.10 | 0.10 | 0.20 | 0.10 | 0.15 | 0.20 |
| **Neutral** | 0.15 | 0.15 | 0.15 | 0.15 | 0.10 | 0.10 | 0.20 |

### Project Structure

```
EmoDebt/
├── main.py                       # Entry point
├── config/scenarios.json          # CRAD dataset (100 scenarios)
├── experiments/
│   ├── run_emodebt.py            # EmoDebt Bayesian optimization
│   ├── run_baseline.py           # Baseline experiments
│   └── run_all_datasets.py       # Cross-dataset evaluation
├── llm/
│   ├── llm_wrapper.py            # LLM interface (GPT, Claude)
│   ├── negotiator.py             # Multi-agent negotiation via LangGraph
│   └── negotiator_multiagent.py  # Multi-agent variant
├── models/
│   ├── base_model.py             # Base emotion model interface
│   └── emodebt_bayesian.py       # Bayesian emotional transition optimizer
├── results/                      # Experiment results (JSON)
└── utils/
    ├── helpers.py                # Utility functions
    └── preprocessing.py          # Data preprocessing
```

## 📚 Citation

If you use EmoDebt in your research, please cite:

```bibtex
@inproceedings{long2026emodebt,
    title={EmoDebt: Bayesian-Optimized Emotional Intelligence for Strategic Agent-to-Agent Debt Recovery},
    author={Long, Yunbo and Liu, Yuhan and Xu, Liming and Brintrup, Alexandra},
    booktitle={Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
    year={2026},
    address={Paphos, Cyprus},
    url={https://arxiv.org/abs/2503.21080}
}
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Yunbo Long](https://github.com/Yunbo-max)**

[⭐ Star this repo](https://github.com/Yunbo-max/EmoDebt) | [🐛 Report Bug](https://github.com/Yunbo-max/EmoDebt/issues) | [💡 Request Feature](https://github.com/Yunbo-max/EmoDebt/issues)

</div>
