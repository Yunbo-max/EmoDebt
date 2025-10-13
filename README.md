# EmoDebt: Bayesian-Optimized Emotional Intelligence for Strategic Agent-to-Agent Debt Recovery

![Debt Negotiation System Diagram](docs/system_diagram.png)

> **Abstract**: This system implements a Bayesian-optimized emotional intelligence framework for automated debt collection negotiations between LLM-powered agents. The creditor agent learns optimal emotional transition strategies through Gaussian Process optimization while adapting to debtor emotional profiles.

## Key Features

- 🧠 **Bayesian Emotional Optimization**: Uses Gaussian Processes to learn Markovian transition matrices between 7 emotional states
- 🎭 **Dynamic Emotional Adaptation**: Creditor agent adjusts strategy based on debtor's emotional profile (happy, angry, sad, etc.)
- ⚖️ **Strategic Concession Patterns**: Implements psychologically-grounded negotiation tactics with temporal constraints
- 📊 **Online Learning**: Continuously improves emotional strategies through negotiation outcomes
- 🤖 **Multi-Model Support**: Compatible with GPT-4o, Claude 3, and other LLMs via LangChain

## Installation
pip install requirements.txt

## Create a config file named `.env` which contains the following lines:
   ```sh
   OPENAI_API_KEY=...
   ```

# Generate debt collection scenarios from CSV data
python debt_prepare.py --debt_csv ./data/credit_recovery_scenarios.csv --n_trial_per_debt 2 --out_fn data/debt_collection_scenarios.json

## Debt recovery negotitaion 
python langgraph_bargain_debt.py --mode bayesian --model_creditor gpt-4o-mini --model_debtor gpt-5-mini --debtor_emotion vanilla --iterations 5 --scenarios 20

### Key Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--mode` | Learning strategy | `bayesian`, `vanilla` |
| `--model_creditor` | LLM for creditor agent | `gpt-4o-mini`, `gpt-5-mini`, etc. |
| `--model_debtor` | LLM for debtor agent | `gpt-4o-mini`, `gpt-5-mini`, etc. |
| `--debtor_emotion` | Fixed emotional profile | `vanilla`,`happy`, `angry`, `sad`, `all` etc. |
| `--iterations` | Learning cycles per scenario | 1-20 |
| `--scenarios` | Number of debt cases | 1-100 |


## 📊 Citation

If you use EmoDebt in your research, please cite:

@article{emodebt2024,
title={EmoDebt: Bayesian-Optimized Emotional Intelligence for Strategic Agent-to-Agent Debt Recovery},
author={Yunbo Long},
journal={arXiv preprint},
year={2024},
url={https://github.com/your-username/emodebt}
}



## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

