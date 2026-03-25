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
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.statistical_analysis import enhance_results_with_statistics, format_ci_results

load_dotenv()

# Import from your codebase
from models.bayesian_multiagent import run_bayesian_transition_experiment
from models.llm_multiagent import run_gpt_orchestrator_experiment
from models.vanilla_model import run_vanilla_baseline_experiment
from models.prompt_model import run_prompt_based_experiment
from models.rl_agents import run_qlearning_experiment
from models.gametheory_agents import run_strategic_dominance_experiment
from models.coherence_agents import run_coherence_experiment
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
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    base_out_dir: str = "results"
):
    """Run experiment on a specific dataset"""
    
    # Start timing for this dataset
    dataset_start_time = time.time()
    
    # Create dataset-specific output directory
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = f"{base_out_dir}/{dataset_type}_{dataset_name}"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Preprocess the CSV into scenarios
    preprocessing_start = time.time()
    print(f"\n📊 Processing {dataset_type} dataset: {csv_path}")
    scenarios = preprocess_all_scenarios(
        csv_path=csv_path,
        scenario_type=dataset_type,
        output_path=f"{out_dir}/scenarios.json",  # Save processed scenarios
        n_scenarios=n_scenarios
    )
    
    preprocessing_time = time.time() - preprocessing_start
    print(f"✅ Created {len(scenarios)} scenarios ({preprocessing_time:.2f}s)")
    
    # Choose which experiment to run
    experiment_start = time.time()
    print(f"⏱️ Starting {model_type} experiment at {datetime.now().strftime('%H:%M:%S')}")
    if model_type == "vanilla":
        print(f"🎯 Running Vanilla Baseline (No Emotion Optimization)")
        results = run_vanilla_baseline_experiment(
            scenarios=scenarios,
            iterations=iterations,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,            debtor_model_type=debtor_model_type,            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    elif model_type == "prompt":
        print(f"🎭 Running Prompt-Based Model (Random Emotion Prompts)")
        results = run_prompt_based_experiment(
            scenarios=scenarios,
            iterations=iterations,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,            debtor_model_type=debtor_model_type,            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    elif model_type == "bayesian":
        print(f"🧠 Running Bayesian Transition Optimization")
        results = run_bayesian_transition_experiment(
            scenarios=scenarios,
            iterations=iterations,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            debtor_model_type=debtor_model_type,
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
            debtor_model_type=debtor_model_type,
            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    elif model_type == "rl_agents":
        print(f"🎮 Running RL Agents (Reinforcement Learning)")
        results = run_qlearning_experiment(
            scenarios=scenarios,
            iterations=iterations,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            debtor_model_type=debtor_model_type,
            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    elif model_type == "gametheory":
        print(f"🎯 Running Game Theory Agents")
        results = run_strategic_dominance_experiment(
            scenarios=scenarios,
            iterations=iterations,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            debtor_model_type=debtor_model_type,
            max_dialog_len=max_dialog_len,
            out_dir=out_dir
        )
    elif model_type == "coherence":
        print(f"🎭 Running Coherence Agents (Emotional Consistency)")
        results = run_coherence_experiment(
            scenarios=scenarios,
            iterations=iterations,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            debtor_model_type=debtor_model_type,
            max_dialog_len=max_dialog_len,
            coherence_weight=0.8,  # Use emotional coherence weight
            out_dir=out_dir
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate timing
    experiment_time = time.time() - experiment_start
    total_dataset_time = time.time() - dataset_start_time
    
    # Add timing information to results
    if 'timing' not in results:
        results['timing'] = {}
    
    results['timing'].update({
        'preprocessing_time_seconds': preprocessing_time,
        'experiment_time_seconds': experiment_time,
        'total_dataset_time_seconds': total_dataset_time,
        'preprocessing_time_formatted': f"{preprocessing_time:.2f}s",
        'experiment_time_formatted': f"{experiment_time/60:.2f}m" if experiment_time > 60 else f"{experiment_time:.2f}s",
        'total_dataset_time_formatted': f"{total_dataset_time/60:.2f}m" if total_dataset_time > 60 else f"{total_dataset_time:.2f}s",
        'negotiations_per_minute': (iterations * len(scenarios)) / (experiment_time / 60) if experiment_time > 0 else 0
    })
    
    print(f"⏱️ Dataset timing summary:")
    print(f"   📊 Preprocessing: {results['timing']['preprocessing_time_formatted']}")
    print(f"   🧠 Experiment: {results['timing']['experiment_time_formatted']}")
    print(f"   🏁 Total: {results['timing']['total_dataset_time_formatted']}")
    print(f"   ⚡ Speed: {results['timing']['negotiations_per_minute']:.1f} negotiations/minute")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run experiments on all datasets")
    
    # Dataset selection
    parser.add_argument("--dataset_type", default="all",
                       choices=["debt", "disaster", "student", "medical", "all"],
                       help="Which dataset to run (or 'all' for all)")
    
    # Experiment parameters
    parser.add_argument("--model_type", default="vanilla",
                       choices=["vanilla", "prompt", "bayesian", "gpt", "rl_agents", "gametheory", "coherence"],
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
    parser.add_argument("--debtor_model_type", default="vanilla",
                       choices=["vanilla", "pressure", "victim", "threat"],
                       help="Debtor tactical approach: vanilla (neutral), pressure (aggressive), victim (sympathy), threat (intimidation)")
    parser.add_argument("--out_dir", default="results/multi_dataset",
                       help="Base output directory for results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    
    args = parser.parse_args()
    
    # Start overall timing
    overall_start_time = time.time()
    print(f"\n🚀 EXPERIMENT SUITE STARTED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define dataset paths (adjust these to your actual file paths)
    dataset_paths = {
        "debt": "../data/credit_recovery_scenarios.csv",
        "disaster": "../data/disaster_survivor_scenarios.csv", 
        "student": "../data/education_sleep_scenarios.csv",
        "medical": "../data/hospital_surgery_scenarios.csv"
    }
    
    # Check which datasets to run
    if args.dataset_type == "all":
        datasets_to_run = list(dataset_paths.keys())
    else:
        datasets_to_run = [args.dataset_type]
    
    # Run experiments
    all_results = {}
    dataset_timings = {}
    
    for dataset_type in datasets_to_run:
        csv_path = dataset_paths.get(dataset_type)
        
        if not csv_path or not os.path.exists(csv_path):
            print(f"⚠️ Dataset file not found: {csv_path}")
            print(f"   Please create the file or update the path in the script")
            continue
        
        print("\n" + "="*80)
        print(f"🚀 STARTING EXPERIMENT: {dataset_type.upper()} DATASET")
        print("="*80)
        
        dataset_start = time.time()
        
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
                debtor_model_type=args.debtor_model_type,
                max_dialog_len=args.max_dialog_len,
                base_out_dir=args.out_dir
            )
            
            all_results[dataset_type] = {
                'summary': results.get('summary_statistics', {}),
                'config': results.get('config', {}),
                'detailed_result': results,  # Store full results for statistical analysis
                'timing': results.get('timing', {})
            }
            
            dataset_elapsed = time.time() - dataset_start
            dataset_timings[dataset_type] = dataset_elapsed
            
            print(f"✅ Completed {dataset_type} dataset ({dataset_elapsed/60:.2f} minutes)")
            
        except Exception as e:
            dataset_elapsed = time.time() - dataset_start
            dataset_timings[dataset_type] = dataset_elapsed
            print(f"❌ Error running {dataset_type} dataset: {e} ({dataset_elapsed/60:.2f} minutes)")
            import traceback
            traceback.print_exc()
    
    # Print cross-dataset comparison with confidence intervals
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("📊 CROSS-DATASET COMPARISON WITH CONFIDENCE INTERVALS")
        print("="*80)
        
        print(f"\n{'Dataset':<15} {'Success Rate (95% CI)':<25} {'Collection Rate (95% CI)':<30} {'Avg Rounds (95% CI)':<25}")
        print("-" * 95)
        
        for dataset_type, result in all_results.items():
            summary = result['summary']
            
            # Get confidence intervals if available
            if 'statistical_analysis' in result.get('detailed_result', {}):
                stat_analysis = result['detailed_result']['statistical_analysis']
                sr_ci = stat_analysis['success_rate']['ci_95']
                cr_ci = stat_analysis['collection_rate']['ci_95'] 
                nr_ci = stat_analysis['negotiation_rounds']['ci_95']
                
                sr_text = f"{stat_analysis['success_rate']['mean']:.1%} [{sr_ci[0]:.1%}, {sr_ci[1]:.1%}]"
                cr_text = f"{stat_analysis['collection_rate']['mean']:.3f} [{cr_ci[0]:.3f}, {cr_ci[1]:.3f}]"
                nr_text = f"{stat_analysis['negotiation_rounds']['mean']:.1f} [{nr_ci[0]:.1f}, {nr_ci[1]:.1f}]"
            else:
                # Fallback to old format
                success_rate = summary.get('success_rate', 0)
                coll_rate = summary.get('collection_rate', {})
                coll_mean = coll_rate.get('mean', 0)
                rounds = summary.get('negotiation_rounds', {})
                rounds_mean = rounds.get('mean', 0)
                
                sr_text = f"{success_rate:.1%} [CI N/A]"
                cr_text = f"{coll_mean:.3f} [CI N/A]"
                nr_text = f"{rounds_mean:.1f} [CI N/A]"
            
            print(f"{dataset_type:<15} {sr_text:<25} {cr_text:<30} {nr_text:<25}")
    
    # Calculate overall timing
    overall_elapsed_time = time.time() - overall_start_time
    overall_hours = int(overall_elapsed_time // 3600)
    overall_minutes = int((overall_elapsed_time % 3600) // 60)
    overall_seconds = int(overall_elapsed_time % 60)
    
    # Enhanced all_results with timing analysis
    enhanced_all_results = {}
    total_negotiations = 0
    
    for dataset_type, result in all_results.items():
        if 'detailed_result' in result:
            detailed = result['detailed_result']
            # Count total negotiations for this dataset
            negotiations_count = detailed.get('summary_statistics', {}).get('counts', {}).get('total_negotiations', 0)
            if negotiations_count == 0:  # Fallback calculation
                negotiations_count = args.iterations * args.scenarios
            total_negotiations += negotiations_count
            
            enhanced_all_results[dataset_type] = {
                'summary': result['summary'],
                'config': result['config'],
                'detailed_result': detailed,
                'timing': result.get('timing', {}),
                'negotiations_count': negotiations_count
            }
        else:
            enhanced_all_results[dataset_type] = result
    
    # Save cross-dataset summary with timing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    
    summary_file = f"{args.out_dir}/cross_dataset_summary_{timestamp}.json"
    
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': vars(args),
            'results': enhanced_all_results,
            'timing_summary': {
                'overall_runtime_seconds': overall_elapsed_time,
                'overall_runtime_formatted': f"{overall_hours}h {overall_minutes}m {overall_seconds}s" if overall_hours > 0 else f"{overall_minutes}m {overall_seconds}s",
                'total_negotiations': total_negotiations,
                'negotiations_per_hour': (total_negotiations / (overall_elapsed_time / 3600)) if overall_elapsed_time > 0 else 0,
                'dataset_timings': {k: f"{v/60:.2f}m" for k, v in dataset_timings.items()},
                'start_time': datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }, f, indent=2)
    
    # Print comprehensive timing summary
    print("\n" + "="*80)
    print("⏱️ COMPREHENSIVE TIMING SUMMARY")
    print("="*80)
    print(f"🚀 Started: {datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏁 Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total Runtime: {overall_hours}h {overall_minutes}m {overall_seconds}s" if overall_hours > 0 else f"⏱️ Total Runtime: {overall_minutes}m {overall_seconds}s")
    print(f"🧮 Total Negotiations: {total_negotiations}")
    print(f"⚡ Speed: {(total_negotiations / (overall_elapsed_time / 3600)):.1f} negotiations/hour")
    
    if dataset_timings:
        print(f"\n📊 Per-Dataset Timing:")
        for dataset, duration in dataset_timings.items():
            print(f"   {dataset}: {duration/60:.2f} minutes")
    
    print(f"\n💾 Cross-dataset summary saved to: {summary_file}")
    print("\n🎉 All experiments completed!")

if __name__ == "__main__":
    main()