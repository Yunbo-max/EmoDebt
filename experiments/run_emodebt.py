# experiments/run_emodebt.py
#!/usr/bin/env python3
"""Run EmoDebt Bayesian Optimization"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from models.emodebt_bayesian import run_emodebt_experiment
from utils.helpers import load_scenarios

def main():
    parser = argparse.ArgumentParser(description="Run EmoDebt Bayesian Optimization")
    parser.add_argument("--iterations", type=int, default=20, help="BO iterations")
    parser.add_argument("--negotiations", type=int, default=3, help="Negotiations per iteration")
    parser.add_argument("--scenarios", type=int, default=3, help="Number of scenarios")
    parser.add_argument("--model_creditor", default="gpt-4o-mini", help="Creditor model")
    parser.add_argument("--model_debtor", default="gpt-4o-mini", help="Debtor model")
    parser.add_argument("--debtor_emotion", default="neutral", help="Debtor emotion")
    
    args = parser.parse_args()
    
    scenarios = load_scenarios("config/scenarios.json")[:args.scenarios]
    
    results = run_emodebt_experiment(
        scenarios=scenarios,
        iterations=args.iterations,
        negotiations_per_iteration=args.negotiations,
        model_creditor=args.model_creditor,
        model_debtor=args.model_debtor,
        debtor_emotion=args.debtor_emotion
    )
    
    print(f"\n✅ EmoDebt completed! Final success rate: {results['performance']['overall_success_rate']:.1%}")

if __name__ == "__main__":
    main()