#!/usr/bin/env python3
"""
Main entry point for EvoEmo Debt Collection System
Usage: python main.py --model evolutionary --generations 10
"""

import argparse
import json
import os
from datetime import datetime
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.evolutionary_bayesian import EvolutionaryBayesianOptimizer, run_evolutionary_experiment
from models.hmm_game_theory import HMMGameTheoryModel, run_hmm_experiment
from llm.llm_wrapper import LLMWrapper
from utils.helpers import load_scenarios, save_results

def main():
    parser = argparse.ArgumentParser(description="EvoEmo Debt Collection System")
    
    # Model selection
    parser.add_argument("--model", choices=["evolutionary", "hmm", "both"], default="evolutionary",
                       help="Model to use: evolutionary (Evolutionary Bayesian), hmm (HMM+Game Theory), both")
    
    # Common parameters
    parser.add_argument("--scenarios", type=int, default=3,
                       help="Number of scenarios to test")
    parser.add_argument("--max_dialog_len", type=int, default=30,
                       help="Maximum dialog length per negotiation")
    parser.add_argument("--out_dir", default="results",
                       help="Output directory for results")
    
    # Evolutionary parameters
    parser.add_argument("--generations", type=int, default=10,
                       help="Number of evolutionary generations (evolutionary model only)")
    parser.add_argument("--population_size", type=int, default=20,
                       help="Population size (evolutionary model only)")
    
    # HMM parameters
    parser.add_argument("--hmm_iterations", type=int, default=5,
                       help="Number of HMM learning iterations (hmm model only)")
    
    # Model parameters
    parser.add_argument("--model_creditor", default="gpt-4o-mini",
                       help="LLM model for creditor agent")
    parser.add_argument("--model_debtor", default="gpt-4o-mini",
                       help="LLM model for debtor agent")
    parser.add_argument("--debtor_emotion", default="neutral",
                       choices=["happy", "surprising", "angry", "sad", "disgust", "fear", "neutral"],
                       help="Fixed debtor emotion for experiments")
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # Load scenarios
    scenarios = load_scenarios("config/scenarios.json")
    if not scenarios:
        print("❌ No scenarios found. Creating sample scenarios...")
        scenarios = create_sample_scenarios()
        with open("config/scenarios.json", "w") as f:
            json.dump(scenarios, f, indent=2)
    
    # Select scenarios for experiment
    test_scenarios = scenarios[:args.scenarios]
    
    print("="*80)
    print("🎭 EvoEmo Debt Collection System")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Scenarios: {len(test_scenarios)}")
    print(f"Creditor Model: {args.model_creditor}")
    print(f"Debtor Model: {args.model_debtor}")
    print(f"Debtor Emotion: {args.debtor_emotion}")
    print("="*80)
    
    results = {}
    
    # Run selected model(s)
    if args.model in ["evolutionary", "both"]:
        print("\n🧬 Running Evolutionary Bayesian Model...")
        evolutionary_results = run_evolutionary_experiment(
            scenarios=test_scenarios,
            generations=args.generations,
            population_size=args.population_size,
            model_creditor=args.model_creditor,
            model_debtor=args.model_debtor,
            debtor_emotion=args.debtor_emotion,
            max_dialog_len=args.max_dialog_len,
            out_dir=args.out_dir
        )
        results["evolutionary"] = evolutionary_results
        print("✅ Evolutionary Bayesian model completed!")
    
    if args.model in ["hmm", "both"]:
        print("\n🎯 Running HMM + Game Theory Model...")
        hmm_results = run_hmm_experiment(
            scenarios=test_scenarios,
            iterations=args.hmm_iterations,
            model_creditor=args.model_creditor,
            model_debtor=args.model_debtor,
            debtor_emotion=args.debtor_emotion,
            max_dialog_len=args.max_dialog_len,
            out_dir=args.out_dir
        )
        results["hmm"] = hmm_results
        print("✅ HMM + Game Theory model completed!")
    
    # Save combined results
    if args.model == "both":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"{args.out_dir}/comparison_results_{timestamp}.json"
        
        # Compare results
        comparison = {
            "evolutionary_success_rate": results["evolutionary"].get("final_stats", {}).get("success_rate", 0),
            "hmm_success_rate": results["hmm"].get("final_stats", {}).get("success_rate", 0),
            "evolutionary_avg_days": results["evolutionary"].get("final_stats", {}).get("avg_collection_days", 0),
            "hmm_avg_days": results["hmm"].get("final_stats", {}).get("avg_collection_days", 0),
        }
        
        print("\n📊 MODEL COMPARISON:")
        print(f"   Evolutionary Bayesian - Success Rate: {comparison['evolutionary_success_rate']:.1%}")
        print(f"   HMM + Game Theory    - Success Rate: {comparison['hmm_success_rate']:.1%}")
        print(f"   Evolutionary Bayesian - Avg Days: {comparison['evolutionary_avg_days']:.1f}")
        print(f"   HMM + Game Theory    - Avg Days: {comparison['hmm_avg_days']:.1f}")
        
        with open(result_file, "w") as f:
            json.dump({"comparison": comparison, "detailed_results": results}, f, indent=2)
        
        print(f"\n💾 Comparison results saved to: {result_file}")
    
    print("\n🎉 All experiments completed!")

def create_sample_scenarios():
    """Create sample debt collection scenarios"""
    return [
        {
            "id": "debt_001",
            "product": {"type": "debt_collection", "amount": 15000},
            "seller": {"target_price": 30, "min_price": 20, "max_price": 60},
            "buyer": {"target_price": 90, "min_price": 60, "max_price": 120},
            "metadata": {
                "outstanding_balance": 15000,
                "creditor_name": "ABC Collections",
                "debtor_name": "John Doe",
                "cash_flow_situation": "Irregular income",
                "business_impact": "High impact on credit score",
                "recovery_stage": "Early"
            }
        },
        {
            "id": "debt_002",
            "product": {"type": "debt_collection", "amount": 25000},
            "seller": {"target_price": 45, "min_price": 30, "max_price": 75},
            "buyer": {"target_price": 120, "min_price": 90, "max_price": 150},
            "metadata": {
                "outstanding_balance": 25000,
                "creditor_name": "XYZ Bank",
                "debtor_name": "Jane Smith",
                "cash_flow_situation": "Steady income",
                "business_impact": "Medium impact",
                "recovery_stage": "Mid"
            }
        },
        {
            "id": "debt_003",
            "product": {"type": "debt_collection", "amount": 5000},
            "seller": {"target_price": 15, "min_price": 10, "max_price": 30},
            "buyer": {"target_price": 60, "min_price": 30, "max_price": 90},
            "metadata": {
                "outstanding_balance": 5000,
                "creditor_name": "Quick Loans",
                "debtor_name": "Bob Johnson",
                "cash_flow_situation": "Poor cash flow",
                "business_impact": "Low impact",
                "recovery_stage": "Late"
            }
        }
    ]

if __name__ == "__main__":
    main()