






#!/usr/bin/env python3
"""
Run GPT Orchestrator Experiment
Usage: python experiments/run_multiagent.py --iterations 5 --scenarios 3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from models.llm_multiagent import run_gpt_orchestrator_experiment
from utils.helpers import load_scenarios

def main():
    parser = argparse.ArgumentParser(description="Run GPT Orchestrator Experiment")
    
    # Parameters
    parser.add_argument("--scenarios", type=int, default=3,
                       help="Number of scenarios to test")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of iterations (negotiations per scenario)")
    parser.add_argument("--max_dialog_len", type=int, default=30,
                       help="Maximum dialog length per negotiation")
    parser.add_argument("--model_creditor", default="gpt-4o-mini",
                       help="LLM model for creditor agent")
    parser.add_argument("--model_debtor", default="gpt-4o-mini",
                       help="LLM model for debtor agent")
    parser.add_argument("--debtor_emotion", default="neutral",
                       choices=["happy", "surprising", "angry", "sad", "disgust", "fear", "neutral"],
                       help="Fixed debtor emotion for experiments")
    parser.add_argument("--out_dir", default="results/gpt_orchestrator",
                       help="Output directory for results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output with detailed decisions")
    
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
    print("🤖 GPT-4o-mini ORCHESTRATOR EXPERIMENT")
    print("="*80)
    print(f"Iterations: {args.iterations}")
    print(f"Scenarios: {len(test_scenarios)}")
    print(f"Agents: Game Theory, Online RL, Emotional Coherence")
    print(f"Orchestrator: GPT-4o-mini LLM")
    print(f"Creditor Model: {args.model_creditor}")
    print(f"Debtor Model: {args.model_debtor}")
    print(f"Debtor Emotion: {args.debtor_emotion}")
    print("="*80)
    
    # Run experiment
    results = run_gpt_orchestrator_experiment(
        scenarios=test_scenarios,
        iterations=args.iterations,
        model_creditor=args.model_creditor,
        model_debtor=args.model_debtor,
        debtor_emotion=args.debtor_emotion,
        max_dialog_len=args.max_dialog_len,
        out_dir=args.out_dir
    )
    
    # Extract summary for text file
    summary_stats = results.get('summary_statistics', {})
    
    # Save simple text summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"{args.out_dir}/summary_{timestamp}.txt"
    
    with open(summary_file, "w", encoding='utf-8') as f:
        f.write("GPT ORCHESTRATOR EXPERIMENT SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Success Rate: {summary_stats.get('success_rate', 0):.1%}\n")
        
        coll = summary_stats.get('collection_rate', {})
        f.write(f"Collection Rate: {coll.get('mean', 0):.3f} ± {coll.get('std', 0):.3f}\n")
        
        rounds = summary_stats.get('negotiation_rounds', {})
        f.write(f"Negotiation Rounds: {rounds.get('mean', 0):.1f} ± {rounds.get('std', 0):.1f}\n\n")
        
        counts = summary_stats.get('counts', {})
        f.write(f"Total Negotiations: {counts.get('total_negotiations', 0)}\n")
        f.write(f"Successful: {counts.get('successful_negotiations', 0)}\n")
        f.write(f"Failed: {counts.get('failed_negotiations', 0)}\n\n")
        
        f.write("AGENT PERFORMANCE:\n")
        agent_perf = summary_stats.get('agent_performance', {})
        for agent, perf in agent_perf.items():
            f.write(f"  {agent}: {perf.get('accuracy', 0):.1%}\n")
    
    print(f"\n📝 Summary saved to: {summary_file}")
    print("✅ GPT Orchestrator experiment completed!")

if __name__ == "__main__":
    main()