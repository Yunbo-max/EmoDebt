#!/usr/bin/env python3
"""
Run Baseline Evolutionary Bayesian Model
Usage: python experiments/run_baseline.py --generations 10 --population_size 20
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from models.baseline_evolutionary import run_baseline_experiment
from utils.helpers import load_scenarios

def main():
    parser = argparse.ArgumentParser(description="Run Baseline Evolutionary Bayesian Model")
    
    # Parameters
    parser.add_argument("--scenarios", type=int, default=3,
                       help="Number of scenarios to test")
    parser.add_argument("--generations", type=int, default=10,
                       help="Number of evolutionary generations")
    parser.add_argument("--population_size", type=int, default=20,
                       help="Negotiations per generation")
    parser.add_argument("--max_dialog_len", type=int, default=30,
                       help="Maximum dialog length per negotiation")
    parser.add_argument("--model_creditor", default="gpt-4o-mini",
                       help="LLM model for creditor agent")
    parser.add_argument("--model_debtor", default="gpt-4o-mini",
                       help="LLM model for debtor agent")
    parser.add_argument("--debtor_emotion", default="neutral",
                       choices=["happy", "surprising", "angry", "sad", "disgust", "fear", "neutral"],
                       help="Fixed debtor emotion for experiments")
    parser.add_argument("--out_dir", default="results/baseline",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # Load scenarios
    scenarios = load_scenarios("config/scenarios.json")
    if not scenarios:
        print("❌ No scenarios found. Please create config/scenarios.json")
        return
    
    test_scenarios = scenarios[:args.scenarios]
    
    print("="*80)
    print("🧬 BASELINE EVOLUTIONARY BAYESIAN MODEL")
    print("="*80)
    print(f"Generations: {args.generations}")
    print(f"Negotiations per generation: {args.population_size}")
    print(f"Scenarios: {len(test_scenarios)}")
    print(f"Creditor Model: {args.model_creditor}")
    print(f"Debtor Model: {args.model_debtor}")
    print(f"Debtor Emotion: {args.debtor_emotion}")
    print("="*80)
    
    # Run baseline experiment
    results = run_baseline_experiment(
        scenarios=test_scenarios,
        generations=args.generations,
        population_size=args.population_size,
        model_creditor=args.model_creditor,
        model_debtor=args.model_debtor,
        debtor_emotion=args.debtor_emotion,
        max_dialog_len=args.max_dialog_len,
        out_dir=args.out_dir
    )
    
    # Print summary
    print("\n📊 BASELINE RESULTS SUMMARY:")
    print("-"*40)
    
    # Extract success rates across generations
    generations = list(results['generation_results'].keys())
    success_rates = []
    avg_days = []
    
    for gen_key in generations:
        gen_data = results['generation_results'][gen_key]
        success_rates.append(gen_data['success_rate'])
        avg_days.append(gen_data['avg_days'])
    
    print(f"Final Success Rate: {success_rates[-1]:.1%}")
    print(f"Average Success Rate: {sum(success_rates)/len(success_rates):.1%}")
    print(f"Best Success Rate: {max(success_rates):.1%}")
    print(f"Final Avg Collection Days: {avg_days[-1]:.1f}")
    print(f"Best Fitness: {results.get('final_stats', {}).get('best_fitness', 0):.3f}")
    
    # Calculate learning trend
    if len(success_rates) >= 3:
        early_avg = sum(success_rates[:2]) / 2
        late_avg = sum(success_rates[-2:]) / 2
        improvement = late_avg - early_avg
        print(f"Learning Improvement: {improvement:+.3f} ({improvement*100:+.1f}%)")
    
    print("\n✅ Baseline experiment completed!")

if __name__ == "__main__":
    main()