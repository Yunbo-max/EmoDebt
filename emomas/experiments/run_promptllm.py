#!/usr/bin/env python3
"""
Run Prompt-Based Model - LLM Negotiation with Emotion Selection Prompts
Usage: python experiments/run_promptllm.py --scenarios 5 --iterations 10
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

from models.prompt_model import run_prompt_based_experiment
from utils.helpers import load_scenarios

def main():
    parser = argparse.ArgumentParser(description="Run Prompt-Based Model")
    
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
    parser.add_argument("--out_dir", type=str, default="results/prompt_based",
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
    
    print("🎯 PROMPT-BASED MODEL")
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
    print("  Type: Prompt-Based (Random Emotion Selection)")
    print("  Strategy: Creditor prompted to use random emotions")
    print("  Emotions: happy, surprising, angry, sad, disgust, fear, neutral")
    print("  Learning: None (random selection only)")
    print("  Optimization: None (pure prompting)")
    print("-" * 50)
    
    # Run experiment
    results = run_prompt_based_experiment(
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
    final_stats = results.get('final_stats', {})
    
    print("🎯 PERFORMANCE METRICS")
    print(f"  Success Rate: {perf.get('success_rate', 0):.1%}")
    print(f"  Average Collection Days: {perf.get('avg_collection_days', 0):.1f}")
    print(f"  Average Collection Amount: ${perf.get('avg_collection_amount', 0):.0f}")
    print(f"  Average Negotiation Rounds: {perf.get('avg_negotiation_rounds', 0):.1f}")
    print(f"  Total Negotiations: {perf.get('total_negotiations', 0)}")
    print(f"  Successful: {perf.get('successful_negotiations', 0)}")
    print(f"  Failed: {perf.get('failed_negotiations', 0)}")
    
    print("\\n🎭 EMOTION DISTRIBUTION")
    emotion_dist = final_stats.get('emotion_distribution', {})
    total_emotions = sum(emotion_dist.values()) if emotion_dist else 0
    for emotion, count in emotion_dist.items():
        percentage = (count / total_emotions) * 100 if total_emotions > 0 else 0
        print(f"  {emotion.title()}: {count} ({percentage:.1f}%)")
    
    print("\\n✨ EMOTION EFFECTIVENESS")
    emotion_effectiveness = analysis.get('emotion_effectiveness', {})
    if emotion_effectiveness:
        # Sort by success rate
        sorted_emotions = sorted(emotion_effectiveness.items(), 
                               key=lambda x: x[1]['success_rate'], reverse=True)
        for emotion, stats in sorted_emotions:
            print(f"  {emotion.title()}: {stats['success_rate']:.1%} success ({stats['count']} uses)")
    else:
        print("  No emotion effectiveness data available")
    
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
    
    print("\\n🧠 MODEL STATISTICS")
    print(f"  Model Type: {final_stats.get('model_type', 'unknown')}")
    print(f"  Total Negotiations: {final_stats.get('negotiation_count', 0)}")
    print(f"  Overall Success Rate: {final_stats.get('success_rate', 0):.1%}")
    print(f"  Uses Emotion Prompts: {final_stats.get('uses_emotion_prompts', False)}")
    print(f"  Optimization: {not final_stats.get('no_optimization', True)}")
    
    # Comparison info
    print("\\n📈 MODEL CHARACTERISTICS")
    print("  ✅ Advantages:")
    print("    - Uses emotion prompts")
    print("    - Tests all seven emotions")
    print("    - Simple to understand")
    print("    - Fast execution")
    print("  ⚠️  Limitations:")
    print("    - Random emotion selection")
    print("    - No learning from experience")
    print("    - No optimization strategy")
    print("    - May use inappropriate emotions")
    
    # Find best and worst emotions
    if emotion_effectiveness:
        best_emotion = max(emotion_effectiveness.items(), key=lambda x: x[1]['success_rate'])
        worst_emotion = min(emotion_effectiveness.items(), key=lambda x: x[1]['success_rate'])
        
        print("\\n🏆 EMOTION INSIGHTS")
        print(f"  Best Emotion: {best_emotion[0].title()} ({best_emotion[1]['success_rate']:.1%} success)")
        print(f"  Worst Emotion: {worst_emotion[0].title()} ({worst_emotion[1]['success_rate']:.1%} success)")
        
        success_diff = best_emotion[1]['success_rate'] - worst_emotion[1]['success_rate']
        print(f"  Effectiveness Range: {success_diff:.1%} difference")
    
    # Save experiment configuration
    config_file = f"{args.out_dir}/experiment_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_file, 'w') as f:
        json.dump({
            'experiment_type': 'prompt_based',
            'timestamp': datetime.now().isoformat(),
            'configuration': vars(args),
            'scenarios_count': len(scenarios),
            'model_description': 'Prompt-based model with random emotion selection'
        }, f, indent=2)
    
    print(f"\\n📁 FILES SAVED")
    print(f"  Main Results: {args.out_dir}/prompt_based_*.json")
    print(f"  Summary: {args.out_dir}/summary_*.txt") 
    print(f"  Configuration: {config_file}")
    
    print("\\n✨ Prompt-based experiment completed!")
    print("\\n💡 TIP: Compare these results with vanilla baseline")
    print("     to see if emotion prompts provide any benefit.")
    
    return results

if __name__ == "__main__":
    main()