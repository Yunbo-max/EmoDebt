#!/usr/bin/env python3
"""
Run DQN (Deep Q-Network) Baseline Model
Usage: python experiments/run_dqn.py --episodes 200 --learning_rate 1e-4
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from models.dqn_baseline import run_dqn_experiment
from utils.helpers import load_scenarios

def main():
    parser = argparse.ArgumentParser(description="Run DQN Baseline Model")
    
    # Parameters
    parser.add_argument("--scenarios", type=int, default=3,
                       help="Number of scenarios to test")
    parser.add_argument("--episodes", type=int, default=200,
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
    
    # DQN parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="DQN learning rate")
    parser.add_argument("--discount_factor", type=float, default=0.95,
                       help="Discount factor (gamma)")
    parser.add_argument("--exploration_rate", type=float, default=1.0,
                       help="Initial exploration rate (epsilon)")
    parser.add_argument("--exploration_decay", type=float, default=0.995,
                       help="Exploration rate decay")
    parser.add_argument("--min_exploration", type=float, default=0.05,
                       help="Minimum exploration rate")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--replay_buffer_size", type=int, default=10000,
                       help="Experience replay buffer size")
    parser.add_argument("--target_update_freq", type=int, default=100,
                       help="Target network update frequency")
    parser.add_argument("--tau", type=float, default=0.01,
                       help="Soft update parameter for target network")
    
    # DQN enhancements
    parser.add_argument("--no_double_dqn", action="store_false", dest="use_double_dqn",
                       help="Disable Double DQN")
    parser.add_argument("--no_dueling", action="store_false", dest="use_dueling",
                       help="Disable Dueling DQN")
    parser.add_argument("--no_per", action="store_false", dest="use_per",
                       help="Disable Prioritized Experience Replay")
    
    parser.add_argument("--device", default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device to run DQN on")
    
    parser.add_argument("--out_dir", default="results/dqn",
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
    print("🧠 DQN (DEEP Q-NETWORK) BASELINE MODEL")
    print("="*80)
    print(f"Total episodes: {args.episodes}")
    print(f"Episodes per scenario: {args.episodes_per_scenario}")
    print(f"Scenarios: {len(test_scenarios)}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Discount factor: {args.discount_factor}")
    print(f"Exploration: {args.exploration_rate} (decay: {args.exploration_decay})")
    print(f"Batch size: {args.batch_size}")
    print(f"Buffer size: {args.replay_buffer_size}")
    print(f"Target update freq: {args.target_update_freq}")
    print(f"Double DQN: {args.use_double_dqn}")
    print(f"Dueling DQN: {args.use_dueling}")
    print(f"PER: {args.use_per}")
    print(f"Device: {args.device}")
    print(f"Creditor Model: {args.model_creditor}")
    print(f"Debtor Model: {args.model_debtor}")
    print(f"Debtor Emotion: {args.debtor_emotion}")
    print("="*80)
    
    # Run DQN experiment
    results = run_dqn_experiment(
        scenarios=test_scenarios,
        episodes=args.episodes,
        episodes_per_scenario=args.episodes_per_scenario,
        model_creditor=args.model_creditor,
        model_debtor=args.model_debtor,
        debtor_emotion=args.debtor_emotion,
        max_dialog_len=args.max_dialog_len,
        out_dir=args.out_dir,
        use_double_dqn=args.use_double_dqn,
        use_dueling=args.use_dueling,
        use_per=args.use_per,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        exploration_rate=args.exploration_rate,
        exploration_decay=args.exploration_decay,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_buffer_size,
        target_update_freq=args.target_update_freq,
        tau=args.tau,
        min_exploration=args.min_exploration
    )
    
    # Print summary
    print("\n📊 DQN RESULTS SUMMARY:")
    print("-"*40)
    
    # Define EMOTIONS here since it's used in the analysis
    EMOTIONS = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
    
    # Extract learning curve
    learning_curve = results.get('learning_curve', [])
    if learning_curve:
        # Calculate performance metrics
        all_rewards = [ep['reward'] for ep in learning_curve]
        all_success = [ep['success'] for ep in learning_curve]
        
        # Split into quarters to measure learning progression
        quarter = len(learning_curve) // 4
        if quarter > 0:
            rewards_by_quarter = []
            success_by_quarter = []
            
            for q in range(4):
                start = q * quarter
                end = (q + 1) * quarter if q < 3 else len(learning_curve)
                quarter_rewards = all_rewards[start:end]
                quarter_success = all_success[start:end]
                
                rewards_by_quarter.append(np.mean(quarter_rewards) if quarter_rewards else 0)
                success_by_quarter.append(np.mean(quarter_success) if quarter_success else 0)
            
            print(f"Reward progression by quarter:")
            for q, (reward, success) in enumerate(zip(rewards_by_quarter, success_by_quarter)):
                print(f"  Q{q+1}: reward={reward:.2f}, success={success:.1%}")
        
        print(f"\nOverall Success Rate: {results.get('overall_success_rate', 0):.1%}")
        print(f"Best Reward: {results.get('best_reward', 0):.2f}")
        print(f"Final Avg Reward: {results.get('final_avg_reward', 0):.2f}")
        
        # Final statistics
        final_stats = results.get('final_stats', {})
        print(f"\n🎯 Final Model Statistics:")
        print(f"   Total episodes: {final_stats.get('total_episodes', 0)}")
        print(f"   Total training steps: {final_stats.get('total_steps', 0)}")
        print(f"   Replay buffer size: {final_stats.get('replay_buffer_size', 0)}")
        print(f"   Final exploration: {final_stats.get('exploration_rate', 0):.3f}")
        
        # Architecture info
        print(f"   Double DQN: {final_stats.get('using_double_dqn', False)}")
        print(f"   Dueling: {final_stats.get('using_dueling', False)}")
        print(f"   PER: {final_stats.get('using_per', False)}")
    
    # Show best learned sequence
    best_sequence = results.get('best_sequence', [])
    if best_sequence:
        print(f"\n🏆 Best Learned Sequence ({len(best_sequence)} emotions):")
        print(f"   {' → '.join(best_sequence)}")
        
        # Analyze the sequence
        print(f"\n🔍 Sequence Analysis:")
        print(f"   Unique emotions: {len(set(best_sequence))}")
        
        # Count emotion frequencies
        from collections import Counter
        emotion_counts = Counter(best_sequence)
        for emotion in EMOTIONS:
            count = emotion_counts.get(emotion, 0)
            if count > 0:
                print(f"   {emotion}: {count} ({count/len(best_sequence):.0%})")
    
    print("\n✅ DQN experiment completed!")

if __name__ == "__main__":
    main()