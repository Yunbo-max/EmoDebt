#!/usr/bin/env python3
"""
Run Q-Learning Baseline Model
Usage: python experiments/run_qlearning.py --episodes 100 --learning_rate 0.1
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from models.qlearning_baseline import run_qlearning_experiment
from utils.helpers import load_scenarios

def main():
    parser = argparse.ArgumentParser(description="Run Q-Learning Baseline Model")
    
    # Parameters
    parser.add_argument("--scenarios", type=int, default=3,
                       help="Number of scenarios to test")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Total training episodes")
    parser.add_argument("--episodes_per_scenario", type=int, default=5,
                       help="Episodes per scenario before cycling")
    parser.add_argument("--max_dialog_len", type=int, default=30,
                       help="Maximum dialog length per negotiation")
    parser.add_argument("--model_creditor", default="gpt-4o-mini",
                       help="LLM model for creditor agent")
    parser.add_argument("--model_debtor", default="gpt-4o-mini",
                       help="LLM model for debtor agent")
    parser.add_argument("--debtor_emotion", default="neutral",
                       choices=["happy", "surprising", "angry", "sad", "disgust", "fear", "neutral"],
                       help="Fixed debtor emotion for experiments")
    
    # Q-learning parameters
    parser.add_argument("--learning_rate", type=float, default=0.1,
                       help="Q-learning learning rate (alpha)")
    parser.add_argument("--discount_factor", type=float, default=0.9,
                       help="Q-learning discount factor (gamma)")
    parser.add_argument("--exploration_rate", type=float, default=1.0,
                       help="Initial exploration rate (epsilon)")
    parser.add_argument("--exploration_decay", type=float, default=0.995,
                       help="Exploration rate decay")
    parser.add_argument("--min_exploration", type=float, default=0.1,
                       help="Minimum exploration rate")
    parser.add_argument("--use_softmax", action="store_true", default=True,
                       help="Use softmax exploration instead of epsilon-greedy")
    parser.add_argument("--no_softmax", action="store_false", dest="use_softmax",
                       help="Use epsilon-greedy exploration")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for softmax exploration")
    
    parser.add_argument("--out_dir", default="results/qlearning",
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
    print("🎯 Q-LEARNING BASELINE MODEL")
    print("="*80)
    print(f"Total episodes: {args.episodes}")
    print(f"Episodes per scenario: {args.episodes_per_scenario}")
    print(f"Scenarios: {len(test_scenarios)}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Discount factor: {args.discount_factor}")
    print(f"Exploration: {args.exploration_rate} (decay: {args.exploration_decay})")
    print(f"Use softmax: {args.use_softmax}")
    if args.use_softmax:
        print(f"Temperature: {args.temperature}")
    print(f"Creditor Model: {args.model_creditor}")
    print(f"Debtor Model: {args.model_debtor}")
    print(f"Debtor Emotion: {args.debtor_emotion}")
    print("="*80)
    
    # Run Q-learning experiment
    results = run_qlearning_experiment(
        scenarios=test_scenarios,
        episodes=args.episodes,
        episodes_per_scenario=args.episodes_per_scenario,
        model_creditor=args.model_creditor,
        model_debtor=args.model_debtor,
        debtor_emotion=args.debtor_emotion,
        max_dialog_len=args.max_dialog_len,
        out_dir=args.out_dir,
        use_softmax=args.use_softmax,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        exploration_rate=args.exploration_rate,
        exploration_decay=args.exploration_decay
    )
    
    # Print summary
    print("\n📊 Q-LEARNING RESULTS SUMMARY:")
    print("-"*40)
    
    # Extract learning curve
    learning_curve = results.get('learning_curve', [])
    if learning_curve:
        # Calculate performance metrics
        all_rewards = [ep['reward'] for ep in learning_curve]
        all_success = [ep['success'] for ep in learning_curve]
        
        # Split into halves to measure learning
        half = len(learning_curve) // 2
        first_half_rewards = all_rewards[:half]
        second_half_rewards = all_rewards[half:]
        first_half_success = all_success[:half]
        second_half_success = all_success[half:]
        
        print(f"Overall Success Rate: {results.get('overall_success_rate', 0):.1%}")
        print(f"Best Reward: {results.get('best_reward', 0):.2f}")
        print(f"Final Avg Reward: {results.get('final_avg_reward', 0):.2f}")
        
        if first_half_rewards and second_half_rewards:
            first_avg = np.mean(first_half_rewards)
            second_avg = np.mean(second_half_rewards)
            improvement = second_avg - first_avg
            print(f"Reward Improvement: {improvement:+.2f} ({improvement/first_avg*100 if first_avg>0 else 0:+.1f}%)")
        
        if first_half_success and second_half_success:
            first_success_rate = np.mean(first_half_success)
            second_success_rate = np.mean(second_half_success)
            success_improvement = second_success_rate - first_success_rate
            print(f"Success Rate Improvement: {success_improvement:+.3f} ({success_improvement*100:+.1f}%)")
        
        # Q-table statistics
        final_stats = results.get('final_stats', {})
        print(f"\n🎯 Model Statistics:")
        print(f"   Q-table mean: {final_stats.get('q_table_mean', 0):.3f}")
        print(f"   Q-table std: {final_stats.get('q_table_std', 0):.3f}")
        print(f"   Transition entropy: {final_stats.get('transition_entropy', 0):.3f}")
        print(f"   Final exploration: {final_stats.get('exploration_rate', 0):.3f}")
    
    # Show best learned sequence
    best_sequence = results.get('best_sequence', [])
    if best_sequence:
        print(f"\n🏆 Best Learned Sequence ({len(best_sequence)} emotions):")
        print(f"   {' → '.join(best_sequence)}")
    
    print("\n✅ Q-learning experiment completed!")

if __name__ == "__main__":
    main()