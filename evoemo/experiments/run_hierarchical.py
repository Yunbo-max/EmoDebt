#!/usr/bin/env python3
"""
Run Hierarchical Evolutionary Bayesian Model
Usage: python experiments/run_hierarchical.py --generations 10 --negotiations 10
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from models.hierarchical_evolutionary import run_hierarchical_experiment
from utils.helpers import load_scenarios

def main():
    parser = argparse.ArgumentParser(description="Run Hierarchical Evolutionary Bayesian Model")
    
    # Parameters
    parser.add_argument("--scenarios", type=int, default=3,
                       help="Number of scenarios to test")
    parser.add_argument("--generations", type=int, default=10,
                       help="Number of evolutionary generations")
    parser.add_argument("--negotiations", type=int, default=10,
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
    parser.add_argument("--mutation_rate", type=float, default=0.15,
                       help="Mutation rate for evolution")
    parser.add_argument("--crossover_rate", type=float, default=0.7,
                       help="Crossover rate for evolution")
    parser.add_argument("--learning_rate", type=float, default=0.6,
                       help="Bayesian learning rate (lambda)")
    parser.add_argument("--out_dir", default="results/hierarchical",
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
    print("🎯 HIERARCHICAL EVOLUTIONARY BAYESIAN MODEL")
    print("="*80)
    print(f"Generations: {args.generations}")
    print(f"Negotiations per generation: {args.negotiations}")
    print(f"Scenarios: {len(test_scenarios)}")
    print(f"Creditor Model: {args.model_creditor}")
    print(f"Debtor Model: {args.model_debtor}")
    print(f"Debtor Emotion: {args.debtor_emotion}")
    print(f"Evolution Parameters:")
    print(f"  Mutation Rate: {args.mutation_rate}")
    print(f"  Crossover Rate: {args.crossover_rate}")
    print(f"  Learning Rate: {args.learning_rate}")
    print("="*80)
    
    # Run hierarchical experiment
    results = run_hierarchical_experiment(
        scenarios=test_scenarios,
        generations=args.generations,
        negotiations_per_gen=args.negotiations,
        model_creditor=args.model_creditor,
        model_debtor=args.model_debtor,
        debtor_emotion=args.debtor_emotion,
        max_dialog_len=args.max_dialog_len,
        out_dir=args.out_dir
    )
    
    # Print hierarchical learning summary
    print("\n📊 HIERARCHICAL LEARNING SUMMARY:")
    print("-"*40)
    
    # Extract metrics across generations
    generations = list(results['generation_results'].keys())
    success_rates = []
    avg_days = []
    group_entropies = []
    
    for gen_key in generations:
        gen_data = results['generation_results'][gen_key]
        success_rates.append(gen_data['success_rate'])
        avg_days.append(gen_data['avg_days'])
        group_entropy = gen_data['stats']['hierarchical_entropy']['group_entropy']
        group_entropies.append(group_entropy)
    
    print(f"Final Success Rate: {success_rates[-1]:.1%}")
    print(f"Average Success Rate: {sum(success_rates)/len(success_rates):.1%}")
    print(f"Final Avg Collection Days: {avg_days[-1]:.1f}")
    print(f"Final Group Matrix Entropy: {group_entropies[-1]:.3f}")
    print(f"Best Fitness: {results.get('final_stats', {}).get('best_fitness', 0):.3f}")
    
    # Show learned group transitions
    final_matrices = results.get('final_hierarchical_matrices', {})
    if 'group_matrix' in final_matrices:
        print(f"\n🎯 LEARNED GROUP TRANSITIONS:")
        groups = results['hierarchical_structure']['groups']
        group_matrix = final_matrices['group_matrix']
        
        for i, from_group in enumerate(groups):
            for j, to_group in enumerate(groups):
                prob = group_matrix[i][j]
                if prob > 0.15:  # Show significant transitions
                    arrow = "→"
                    print(f"  {from_group:8} {arrow:2} {to_group:8}: {prob:.3f}")
    
    print("\n✅ Hierarchical experiment completed!")

if __name__ == "__main__":
    main()