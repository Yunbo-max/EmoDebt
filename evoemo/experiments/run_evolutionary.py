#!/usr/bin/env python3
"""
Run Baseline Evolutionary Model
Usage: python experiments/run_evolutionary.py --generations 10 --population_size 20
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
    parser = argparse.ArgumentParser(description="Run Baseline Evolutionary Model")
    
    # Parameters
    parser.add_argument("--scenarios", type=int, default=3,
                       help="Number of scenarios to test")
    parser.add_argument("--generations", type=int, default=1,
                       help="Number of evolutionary generations")
    parser.add_argument("--population_size", type=int, default=3,
                       help="Population size (negotiations per generation)")
    parser.add_argument("--max_dialog_len", type=int, default=30,
                       help="Maximum dialog length per negotiation")
    parser.add_argument("--model_creditor", default="gpt-4o-mini",
                       help="LLM model for creditor agent")
    parser.add_argument("--model_debtor", default="gpt-4o-mini",
                       help="LLM model for debtor agent")
    parser.add_argument("--debtor_emotion", default="neutral",
                       choices=["happy", "surprising", "angry", "sad", "disgust", "fear", "neutral"],
                       help="Fixed debtor emotion for experiments")
    parser.add_argument("--elite_size", type=int, default=5,
                       help="Number of elite sequences to keep")
    parser.add_argument("--mutation_rate", type=float, default=0.1,
                       help="Mutation rate for evolution")
    parser.add_argument("--crossover_rate", type=float, default=0.7,
                       help="Crossover rate for evolution")
    parser.add_argument("--out_dir", default="results/baseline_evolutionary",
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
    print("🧬 BASELINE EVOLUTIONARY MODEL")
    print("="*80)
    print(f"Generations: {args.generations}")
    print(f"Population size (negotiations per generation): {args.population_size}")
    print(f"Scenarios: {len(test_scenarios)}")
    print(f"Creditor Model: {args.model_creditor}")
    print(f"Debtor Model: {args.model_debtor}")
    print(f"Debtor Emotion: {args.debtor_emotion}")
    print(f"Evolution Parameters:")
    print(f"  Elite Size: {args.elite_size}")
    print(f"  Mutation Rate: {args.mutation_rate}")
    print(f"  Crossover Rate: {args.crossover_rate}")
    print("="*80)
    
    # Run baseline evolutionary experiment
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
    
    # Print evolutionary learning summary
    print("\n📊 EVOLUTIONARY LEARNING SUMMARY:")
    print("-"*40)
    
    # Extract metrics across generations
    generations = list(results['generation_results'].keys())
    success_rates = []
    avg_days = []
    best_fitnesses = []
    
    for gen_key in generations:
        gen_data = results['generation_results'][gen_key]
        success_rates.append(gen_data['success_rate'])
        avg_days.append(gen_data['avg_days'])
        best_fitnesses.append(gen_data['best_fitness'])
    
    print(f"Final Success Rate: {success_rates[-1]:.1%}")
    print(f"Average Success Rate: {sum(success_rates)/len(success_rates):.1%}")
    print(f"Final Avg Collection Days: {avg_days[-1]:.1f}")
    print(f"Final Best Fitness: {best_fitnesses[-1]:.3f}")
    
    # Show evolution progress
    if len(success_rates) > 1:
        initial_success = success_rates[0]
        final_success = success_rates[-1]
        improvement = final_success - initial_success
        print(f"Success Rate Improvement: {improvement:+.1%}")
        
        initial_fitness = best_fitnesses[0]
        final_fitness = best_fitnesses[-1]
        fitness_improvement = final_fitness - initial_fitness
        print(f"Fitness Improvement: {fitness_improvement:+.3f}")
    
    # Show best learned sequence
    final_result = results.get('final_best_sequence', {})
    best_sequence = final_result.get('sequence')
    if best_sequence:
        print(f"\n🏆 Best Learned Sequence ({len(best_sequence)} emotions):")
        print(f"   {' → '.join(best_sequence)}")
        print(f"   Final Fitness: {final_result.get('fitness', 0):.3f}")
    
    # Show transition matrix summary
    transition_matrix = final_result.get('transition_matrix')
    if transition_matrix:
        print(f"\n📈 Learned Transition Matrix (top probabilities):")
        emotions = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
        for i, from_emotion in enumerate(emotions):
            max_prob_idx = max(range(len(transition_matrix[i])), key=lambda j: transition_matrix[i][j])
            max_prob = transition_matrix[i][max_prob_idx]
            to_emotion = emotions[max_prob_idx]
            if max_prob > 0.2:  # Show significant transitions
                print(f"   {from_emotion:10} → {to_emotion:10}: {max_prob:.3f}")
    
    print("\n✅ Baseline evolutionary experiment completed!")

if __name__ == "__main__":
    main()