# EmoDebt - Negotiation Simulation Framework

A negotiation simulation framework incorporating LLM agents with emotional profiles.

## Installation
pip install requirements.txt

## Create a config file named `.env` which contains the following lines:
   ```sh
   OPENAI_API_KEY=...
   OPENAI_ORGANIZATION=...(optional)
   LLM_SERVER=.... (optional)
   HUGGING_FACE_TOKEN=...
   ```

## Usage
python langgraph_bargain_debt_simple.py --mode bayesian --model_creditor gpt-4o-mini --model_debtor gpt-5-mini --debtor_emotion vanilla --iterations 10 --scenarios 10

