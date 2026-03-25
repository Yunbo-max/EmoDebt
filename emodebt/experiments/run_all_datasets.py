#!/usr/bin/env python3
"""
Run Bayesian Transition Optimization on all four datasets
Usage: python experiments/run_all_datasets.py --dataset_type debt --iterations 10 --scenarios 5
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import from your codebase
from models.bayesian_multiagent import run_bayesian_transition_experiment
from models.llm_multiagent import run_gpt_orchestrator_experiment
from models.baseline_evolutionary import run_baseline_experiment
from models.qlearning_baseline import run_qlearning_experiment
from models.dqn_baseline import run_dqn_experiment
from models.hierarchical_evolutionary import run_hierarchical_experiment
from models.hmm_game_theory import run_hmm_experiment
from utils.preprocessing import preprocess_all_scenarios

def run_experiment_on_dataset(
    csv_path: str,
    dataset_type: str,
    model_type: str = "bayesian",  # "bayesian" or "gpt"
    iterations: int = 5,
    n_scenarios: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    max_dialog_len: int = 30,
    base_out_dir: str = "results"
):
    """Run experiment on a specific dataset"""
    
    # Create dataset-specific output directory
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = f"{base_out_dir}/{dataset_type}_{dataset_name}"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Preprocess the CSV into scenarios
    print(f"\n📊 Processing {dataset_type} dataset: {csv_path}")
    scenarios = preprocess_all_scenarios(
        csv_path=csv_path,
        scenario_type=dataset_type,
        output_path=f"{out_dir}/scenarios.json",  # Save processed scenarios
        n_scenarios=n_scenarios
    )
    
    print(f"✅ Created {len(scenarios)} scenarios")
    
    # Choose which experiment to run
    if model_type == "bayesian":
        print(f"🧠 Running Bayesian Transition Optimization")
        results = run_bayesian_transition_experiment(
            scenarios=scenarios,
            iterations=iterations,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    elif model_type == "gpt":
        print(f"🤖 Running GPT Orchestrator")
        results = run_gpt_orchestrator_experiment(
            scenarios=scenarios,
            iterations=iterations,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    elif model_type == "baseline":
        print(f"📊 Running Baseline Evolutionary Model")
        results = run_baseline_experiment(
            scenarios=scenarios,
            generations=iterations,  # Use iterations as generations
            population_size=20,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    elif model_type == "qlearning":
        print(f"🎯 Running Q-Learning Baseline")
        results = run_qlearning_experiment(
            scenarios=scenarios,
            episodes=iterations * 20,  # More episodes for Q-learning
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    elif model_type == "dqn":
        print(f"🎮 Running DQN Baseline")
        results = run_dqn_experiment(
            scenarios=scenarios,
            episodes=iterations * 20,  # More episodes for DQN
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    elif model_type == "hierarchical":
        print(f"🏗️ Running Hierarchical Bayesian Model")
        results = run_hierarchical_experiment(
            scenarios=scenarios,
            generations=iterations,
            population_size=20,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    elif model_type == "hmm":
        print(f"🎲 Running HMM + Game Theory Model")
        results = run_hmm_experiment(
            scenarios=scenarios,
            iterations=iterations,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run experiments on all datasets")
    
    # Dataset selection
    parser.add_argument("--dataset_type", default="all",
                       choices=["debt", "disaster", "student", "medical", "all"],
                       help="Which dataset to run (or 'all' for all)")
    
    # Experiment parameters
    parser.add_argument("--model_type", default="bayesian",
                       choices=["bayesian", "gpt", "baseline", "qlearning", "dqn", "hierarchical", "hmm"],
                       help="Which model to use")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of iterations per scenario")
    parser.add_argument("--scenarios", type=int, default=5,
                       help="Number of scenarios to test")
    parser.add_argument("--max_dialog_len", type=int, default=30,
                       help="Maximum dialog length per negotiation")
    parser.add_argument("--model_creditor", default="gpt-4o-mini",
                       help="LLM model for creditor agent")
    parser.add_argument("--model_debtor", default="gpt-4o-mini",
                       help="LLM model for debtor agent")
    parser.add_argument("--debtor_emotion", default="neutral",
                       choices=["happy", "surprising", "angry", "sad", "disgust", "fear", "neutral"],
                       help="Fixed debtor emotion for experiments")
    parser.add_argument("--out_dir", default="results/multi_dataset",
                       help="Base output directory for results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    
    args = parser.parse_args()
    
    # Define dataset paths (adjust these to your actual file paths)
    dataset_paths = {
        "debt": "data/debt_collection.csv",
        "disaster": "data/disaster_rescue.csv", 
        "student": "data/student_sleep.csv",
        "medical": "data/medical_surgery.csv"
    }
    
    # Check which datasets to run
    if args.dataset_type == "all":
        datasets_to_run = list(dataset_paths.keys())
    else:
        datasets_to_run = [args.dataset_type]
    
    # Run experiments
    all_results = {}
    
    for dataset_type in datasets_to_run:
        csv_path = dataset_paths.get(dataset_type)
        
        if not csv_path or not os.path.exists(csv_path):
            print(f"⚠️ Dataset file not found: {csv_path}")
            print(f"   Please create the file or update the path in the script")
            continue
        
        print("\n" + "="*80)
        print(f"🚀 STARTING EXPERIMENT: {dataset_type.upper()} DATASET")
        print("="*80)
        
        try:
            results = run_experiment_on_dataset(
                csv_path=csv_path,
                dataset_type=dataset_type,
                model_type=args.model_type,
                iterations=args.iterations,
                n_scenarios=args.scenarios,
                model_creditor=args.model_creditor,
                model_debtor=args.model_debtor,
                debtor_emotion=args.debtor_emotion,
                max_dialog_len=args.max_dialog_len,
                base_out_dir=args.out_dir
            )
            
            all_results[dataset_type] = {
                'summary': results.get('summary_statistics', {}),
                'config': results.get('config', {})
            }
            
            print(f"✅ Completed {dataset_type} dataset")
            
        except Exception as e:
            print(f"❌ Error running {dataset_type} dataset: {e}")
            import traceback
            traceback.print_exc()
    
    # Print cross-dataset comparison
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("📊 CROSS-DATASET COMPARISON")
        print("="*80)
        
        print(f"\n{'Dataset':<15} {'Success Rate':<15} {'Collection Rate':<20} {'Avg Rounds':<15}")
        print("-" * 70)
        
        for dataset_type, result in all_results.items():
            summary = result['summary']
            success_rate = summary.get('success_rate', 0)
            coll_rate = summary.get('collection_rate', {})
            coll_mean = coll_rate.get('mean', 0)
            rounds = summary.get('negotiation_rounds', {})
            rounds_mean = rounds.get('mean', 0)
            
            print(f"{dataset_type:<15} {success_rate:<15.1%} {coll_mean:<20.3f} {rounds_mean:<15.1f}")
    
    # Save cross-dataset summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"{args.out_dir}/cross_dataset_summary_{timestamp}.json"
    
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': vars(args),
            'results': all_results
        }, f, indent=2)
    
    print(f"\n💾 Cross-dataset summary saved to: {summary_file}")
    print("\n🎉 All experiments completed!")

if __name__ == "__main__":
    main()