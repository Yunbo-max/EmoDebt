#!/usr/bin/env python3
"""
Run HMM + Game Theory Model
Usage: python experiments/run_hmm.py --iterations 10 --exploration_rate 0.2
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

from models.hmm_game_theory import run_hmm_experiment
from utils.helpers import load_scenarios

def main():
    parser = argparse.ArgumentParser(description="Run HMM + Game Theory Model")
    
    # Parameters
    parser.add_argument("--scenarios", type=int, default=3,
                       help="Number of scenarios to test")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of iterations to run")
    
    # Model parameters
    parser.add_argument("--model_creditor", type=str, default="gpt-4o-mini",
                       help="Creditor model")
    parser.add_argument("--model_debtor", type=str, default="gpt-4o-mini",
                       help="Debtor model")
    parser.add_argument("--debtor_emotion", type=str, default="neutral",
                       help="Debtor emotion state")
    parser.add_argument("--max_dialog_len", type=int, default=30,
                       help="Maximum dialog length")
    
    # HMM specific parameters
    parser.add_argument("--exploration_rate", type=float, default=0.2,
                       help="Initial exploration rate")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                       help="Learning rate for model updates")
    
    # Output
    parser.add_argument("--out_dir", type=str, default="results/hmm_game_theory",
                       help="Output directory")
    parser.add_argument("--save_stats", action="store_true",
                       help="Save detailed model statistics")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load scenarios
    scenarios = load_scenarios()[:args.scenarios]
    
    print("🎯 HMM + GAME THEORY MODEL")
    print("="*40)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Iterations: {args.iterations}")
    print(f"Exploration Rate: {args.exploration_rate}")
    print(f"Models: {args.model_creditor} vs {args.model_debtor}")
    print(f"Debtor Emotion: {args.debtor_emotion}")
    print(f"Output: {args.out_dir}")
    print("-"*40)
    
    # Run experiment
    results = run_hmm_experiment(
        scenarios=scenarios,
        iterations=args.iterations,
        model_creditor=args.model_creditor,
        model_debtor=args.model_debtor,
        debtor_emotion=args.debtor_emotion,
        max_dialog_len=args.max_dialog_len,
        out_dir=args.out_dir
    )
    
    # Display results
    print("\n📊 FINAL RESULTS")
    print("="*40)
    
    perf = results.get('performance', {})
    print(f"✅ Success Rate: {perf.get('success_rate', 0):.1%}")
    print(f"⏱️  Avg Collection Days: {perf.get('avg_collection_days', 0):.1f}")
    print(f"🔢 Total Negotiations: {perf.get('total_negotiations', 0)}")
    print(f"🎯 Successful: {perf.get('successful_negotiations', 0)}")
    
    # Model statistics
    final_stats = results.get('final_stats', {})
    if final_stats:
        print(f"\n🧠 MODEL STATISTICS")
        print(f"Exploration Rate: {final_stats.get('exploration_rate', 0):.3f}")
        print(f"Transition Entropy: {final_stats.get('transition_entropy', 0):.3f}")
        print(f"Emission Entropy: {final_stats.get('emission_entropy', 0):.3f}")
        print(f"History Length: {final_stats.get('history_length', 0)}")
        print(f"Final Emotion: {final_stats.get('current_emotion', 'unknown')}")
    
    # Save detailed stats if requested
    if args.save_stats and final_stats:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = f"{args.out_dir}/hmm_stats_{timestamp}.json"
        
        with open(stats_file, "w") as f:
            json.dump(final_stats, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        print(f"📊 Detailed stats saved to: {stats_file}")
    
    print("\n✨ HMM + Game Theory experiment completed!")
    
    return results

if __name__ == "__main__":
    main()