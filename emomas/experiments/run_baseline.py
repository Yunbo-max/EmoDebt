#!/usr/bin/env python3
"""
Run Vanilla Baseline Model - Simple Two-Agent Negotiation
Usage: python experiments/run_baseline.py --scenarios 5 --iterations 10
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

from models.vanilla_model import run_vanilla_baseline_experiment
from utils.helpers import load_scenarios

def main():
    parser = argparse.ArgumentParser(description="Run Vanilla Baseline Model")
    
    # Basic parameters
    parser.add_argument("--scenarios", type=int, default=3,
                       help="Number of scenarios to test")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of iterations per scenario")
    
    # Model parameters
    parser.add_argument("--model_creditor", type=str, default="gpt-4o-mini",
                       help="LLM model for creditor agent")
    parser.add_argument("--model_debtor", type=str, default="gpt-4o-mini",
                       help="LLM model for debtor agent")
    parser.add_argument("--debtor_emotion", type=str, default="neutral",
                       choices=["happy", "surprising", "angry", "sad", "disgust", "fear", "neutral"],
                       help="Debtor emotion state")
    parser.add_argument("--max_dialog_len", type=int, default=30,
                       help="Maximum dialog length per negotiation")
    
    # Output parameters
    parser.add_argument("--out_dir", type=str, default="results/vanilla_baseline",
                       help="Output directory for results")
    parser.add_argument("--save_dialogs", action="store_true",
                       help="Save detailed dialog transcripts")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed negotiation progress")
    
    # Experiment parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load scenarios
    scenarios = load_scenarios()[:args.scenarios]
    
    print("🎯 VANILLA BASELINE MODEL")
    print("=" * 50)
    print("📋 EXPERIMENT CONFIGURATION")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Creditor Model: {args.model_creditor}")
    print(f"  Debtor Model: {args.model_debtor}")
    print(f"  Debtor Emotion: {args.debtor_emotion}")
    print(f"  Max Dialog Length: {args.max_dialog_len}")
    print(f"  Output Directory: {args.out_dir}")
    print(f"  Random Seed: {args.seed}")
    print()
    print("🤖 MODEL DETAILS")
    print("  Type: Vanilla Baseline (No Emotion Optimization)")
    print("  Strategy: Always neutral tone, direct LLM negotiation")
    print("  Learning: None (pure baseline)")
    print("-" * 50)
    
    # Run experiment
    results = run_vanilla_baseline_experiment(
        scenarios=scenarios,
        iterations=args.iterations,
        model_creditor=args.model_creditor,
        model_debtor=args.model_debtor,
        debtor_emotion=args.debtor_emotion,
        max_dialog_len=args.max_dialog_len,
        out_dir=args.out_dir
    )
    
    # Display detailed results
    print("\\n📊 DETAILED RESULTS")
    print("=" * 50)
    
    perf = results.get('performance', {})
    analysis = results.get('analysis', {})
    
    print("🎯 PERFORMANCE METRICS")
    print(f"  Success Rate: {perf.get('success_rate', 0):.1%}")
    print(f"  Average Collection Days: {perf.get('avg_collection_days', 0):.1f}")
    print(f"  Average Collection Amount: ${perf.get('avg_collection_amount', 0):.0f}")
    print(f"  Average Negotiation Rounds: {perf.get('avg_negotiation_rounds', 0):.1f}")
    print(f"  Total Negotiations: {perf.get('total_negotiations', 0)}")
    print(f"  Successful: {perf.get('successful_negotiations', 0)}")
    print(f"  Failed: {perf.get('failed_negotiations', 0)}")
    
    print("\\n❌ FAILURE ANALYSIS")
    failure_breakdown = analysis.get('failure_breakdown', {})
    if failure_breakdown:
        for reason, count in failure_breakdown.items():
            percentage = (count / perf.get('failed_negotiations', 1)) * 100
            print(f"  {reason.title()}: {count} ({percentage:.1f}%)")
    else:
        print("  No failures recorded")
    
    print("\\n💬 DIALOG PATTERNS")
    success_patterns = analysis.get('success_patterns', {})
    print(f"  Avg Rounds (Successful): {success_patterns.get('avg_rounds_successful', 0):.1f}")
    print(f"  Avg Rounds (Failed): {success_patterns.get('avg_rounds_failed', 0):.1f}")
    
    # Model statistics
    final_stats = results.get('final_stats', {})
    print("\\n🧠 MODEL STATISTICS")
    print(f"  Model Type: {final_stats.get('model_type', 'unknown')}")
    print(f"  Total Negotiations: {final_stats.get('negotiation_count', 0)}")
    print(f"  Overall Success Rate: {final_stats.get('success_rate', 0):.1%}")
    print(f"  Emotion Strategy: Always Neutral")
    print(f"  Learning Capability: None (Pure Baseline)")
    
    # Comparison baseline info
    print("\\n📈 BASELINE CHARACTERISTICS")
    print("  ✅ Advantages:")
    print("    - Simple and interpretable")
    print("    - No complex optimization")  
    print("    - Fast execution")
    print("    - Consistent behavior")
    print("  ⚠️  Limitations:")
    print("    - No emotion adaptation")
    print("    - No learning from experience") 
    print("    - May miss emotional opportunities")
    print("    - Cannot improve over time")
    
    # Save experiment configuration
    config_file = f"{args.out_dir}/experiment_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_file, 'w') as f:
        json.dump({
            'experiment_type': 'vanilla_baseline',
            'timestamp': datetime.now().isoformat(),
            'configuration': vars(args),
            'scenarios_count': len(scenarios),
            'model_description': 'Vanilla baseline with no emotion optimization'
        }, f, indent=2)
    
    print(f"\\n📁 FILES SAVED")
    print(f"  Main Results: {args.out_dir}/vanilla_baseline_*.json")
    print(f"  Summary: {args.out_dir}/summary_*.txt") 
    print(f"  Configuration: {config_file}")
    
    print("\\n✨ Vanilla Baseline experiment completed!")
    print("\\n💡 TIP: Use these results as baseline to compare against")
    print("     emotion-optimized models to measure improvement.")
    
    return results

if __name__ == "__main__":
    main()