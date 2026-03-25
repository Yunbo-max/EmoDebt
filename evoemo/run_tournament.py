#!/usr/bin/env python3
"""
Model Tournament System - Run all models against each other
Creates a tournament bracket with statistical analysis
"""

import os
import sys
import time
import json
import itertools
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_training_testing import run_testing_stage
from utils.preprocessing import preprocess_all_scenarios
from utils.statistical_analysis import enhance_results_with_statistics, format_ci_results

def run_model_tournament(
    models: List[str] = None,
    dataset_type: str = "debt",
    test_scenarios: int = 5,
    iterations: int = 3,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini"
):
    """Run a complete tournament between all models"""
    
    if models is None:
        models = ['vanilla', 'evolutionary', 'dqn', 'hierarchical', 'qlearning']
    
    print(f"\n🏆 MODEL TOURNAMENT SYSTEM")
    print("=" * 60)
    print(f"📊 Dataset: {dataset_type}")
    print(f"🤖 Models: {', '.join(models)}")
    print(f"📋 Scenarios per match: {test_scenarios}")
    print(f"🔄 Iterations per scenario: {iterations}")
    print("=" * 60)
    
    # Dataset paths
    dataset_paths = {
        "debt": "../data/credit_recovery_scenarios.csv",
        "disaster": "../data/disaster_survivor_scenarios.csv", 
        "student": "../data/education_sleep_scenarios.csv",
        "medical": "../data/hospital_surgery_scenarios.csv"
    }
    
    csv_path = dataset_paths.get(dataset_type)
    if not csv_path or not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        return
    
    # Preprocess test scenarios
    all_scenarios = preprocess_all_scenarios(
        csv_path=csv_path,
        scenario_type=dataset_type,
        n_scenarios=test_scenarios * 3  # Get extra scenarios for variety
    )
    
    import random
    random.shuffle(all_scenarios)
    test_scenario_pool = all_scenarios[:test_scenarios]
    
    # Create tournament bracket (all vs all)
    matches = list(itertools.combinations(models, 2))
    
    # Also add reverse matches (model A as creditor vs model B as debtor, and vice versa)
    all_matches = []
    for model_a, model_b in matches:
        all_matches.append((model_a, model_b))  # A as creditor, B as debtor
        if model_a != model_b:  # Don't duplicate same vs same
            all_matches.append((model_b, model_a))  # B as creditor, A as debtor
    
    print(f"🎮 Tournament bracket: {len(all_matches)} total matches")
    
    # Create tournament results directory
    tournament_dir = f"results/tournament_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(tournament_dir, exist_ok=True)
    
    tournament_results = {
        'tournament_info': {
            'dataset_type': dataset_type,
            'models': models,
            'test_scenarios': test_scenarios,
            'iterations': iterations,
            'total_matches': len(all_matches),
            'start_time': datetime.now().isoformat()
        },
        'matches': {},
        'leaderboard': {},
        'head_to_head_matrix': {}
    }
    
    # Run all matches
    match_results = []
    
    for i, (creditor_model, debtor_model) in enumerate(all_matches, 1):
        print(f"\n🥊 Match {i}/{len(all_matches)}: {creditor_model.upper()} vs {debtor_model.upper()}")
        
        try:
            # Run the match
            match_result = run_testing_stage(
                creditor_model_name=creditor_model,
                debtor_model_name=debtor_model,
                test_scenarios=test_scenario_pool,
                iterations=iterations,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                max_dialog_len=30
            )
            
            # Store results
            match_key = f"{creditor_model}_vs_{debtor_model}"
            tournament_results['matches'][match_key] = match_result
            match_results.append((creditor_model, debtor_model, match_result))
            
            success_rate = match_result['performance']['success_rate']
            print(f"   Result: {success_rate:.1%} success rate")
            
        except Exception as e:
            print(f"   ❌ Match failed: {e}")
            continue
        
        # Brief pause between matches
        time.sleep(2)
    
    # Calculate leaderboard
    model_stats = {model: {'wins': 0, 'total_matches': 0, 'total_success_rate': 0, 'as_creditor': [], 'as_debtor': []} 
                  for model in models}
    
    # Head-to-head matrix
    h2h_matrix = {model: {opponent: {'wins': 0, 'losses': 0, 'success_rates': []} for opponent in models if opponent != model} 
                 for model in models}
    
    for creditor, debtor, result in match_results:
        success_rate = result['performance']['success_rate']
        
        # Track as creditor
        model_stats[creditor]['as_creditor'].append(success_rate)
        model_stats[creditor]['total_matches'] += 1
        model_stats[creditor]['total_success_rate'] += success_rate
        
        # Head-to-head tracking
        if success_rate > 0.5:  # Arbitrary threshold for "win"
            model_stats[creditor]['wins'] += 1
            h2h_matrix[creditor][debtor]['wins'] += 1
            h2h_matrix[debtor][creditor]['losses'] += 1
        
        h2h_matrix[creditor][debtor]['success_rates'].append(success_rate)
    
    # Calculate final leaderboard
    leaderboard = []
    for model in models:
        stats = model_stats[model]
        avg_success_rate = stats['total_success_rate'] / max(1, stats['total_matches'])
        win_rate = stats['wins'] / max(1, stats['total_matches'])
        avg_as_creditor = sum(stats['as_creditor']) / max(1, len(stats['as_creditor']))
        
        leaderboard.append({
            'model': model,
            'average_success_rate': avg_success_rate,
            'win_rate': win_rate,
            'wins': stats['wins'],
            'total_matches': stats['total_matches'],
            'avg_as_creditor': avg_as_creditor,
            'creditor_matches': len(stats['as_creditor'])
        })
    
    # Sort by average success rate
    leaderboard.sort(key=lambda x: x['average_success_rate'], reverse=True)
    tournament_results['leaderboard'] = leaderboard
    tournament_results['head_to_head_matrix'] = h2h_matrix
    
    # Save tournament results
    tournament_file = f"{tournament_dir}/tournament_results.json"
    with open(tournament_file, 'w', encoding='utf-8') as f:
        json.dump(tournament_results, f, indent=2, default=str)
    
    # Create tournament summary
    summary_file = f"{tournament_dir}/tournament_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("MODEL TOURNAMENT RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {dataset_type}\n")
        f.write(f"Models: {', '.join(models)}\n")
        f.write(f"Test scenarios: {test_scenarios}\n")
        f.write(f"Iterations per scenario: {iterations}\n")
        f.write(f"Total matches: {len(all_matches)}\n\n")
        
        f.write("LEADERBOARD (by average success rate):\n")
        f.write("-" * 50 + "\n")
        for i, entry in enumerate(leaderboard, 1):
            f.write(f"{i}. {entry['model'].upper()}\n")
            f.write(f"   Average Success Rate: {entry['average_success_rate']:.1%}\n")
            f.write(f"   Win Rate: {entry['win_rate']:.1%} ({entry['wins']}/{entry['total_matches']})\n")
            f.write(f"   As Creditor: {entry['avg_as_creditor']:.1%} ({entry['creditor_matches']} matches)\n\n")
        
        f.write("HEAD-TO-HEAD SUMMARY:\n")
        f.write("-" * 50 + "\n")
        for model in models:
            f.write(f"{model.upper()}:\n")
            for opponent in models:
                if opponent != model and opponent in h2h_matrix[model]:
                    h2h_data = h2h_matrix[model][opponent]
                    avg_success = sum(h2h_data['success_rates']) / max(1, len(h2h_data['success_rates']))
                    f.write(f"  vs {opponent}: {avg_success:.1%} success rate\n")
            f.write("\n")
    
    # Print tournament results
    print(f"\n🏆 TOURNAMENT RESULTS")
    print("=" * 60)
    print(f"📊 Leaderboard:")
    for i, entry in enumerate(leaderboard, 1):
        print(f"  {i}. {entry['model'].upper()}: {entry['average_success_rate']:.1%} success rate")
    
    print(f"\n💾 Tournament results saved to: {tournament_file}")
    print(f"💾 Tournament summary saved to: {summary_file}")
    
    return tournament_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model tournament")
    parser.add_argument("--models", nargs="+", 
                       choices=["vanilla", "evolutionary", "dqn", "hierarchical", "qlearning"],
                       help="Models to include in tournament")
    parser.add_argument("--dataset_type", default="debt",
                       choices=["debt", "disaster", "student", "medical"],
                       help="Dataset type")
    parser.add_argument("--test_scenarios", type=int, default=5,
                       help="Number of scenarios per match")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Iterations per scenario")
    parser.add_argument("--model_creditor", default="gpt-4o-mini",
                       help="LLM for creditor")
    parser.add_argument("--model_debtor", default="gpt-4o-mini",
                       help="LLM for debtor")
    
    args = parser.parse_args()
    
    run_model_tournament(
        models=args.models,
        dataset_type=args.dataset_type,
        test_scenarios=args.test_scenarios,
        iterations=args.iterations,
        model_creditor=args.model_creditor,
        model_debtor=args.model_debtor
    )