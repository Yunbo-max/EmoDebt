#!/usr/bin/env python3
"""
Run Coherence Agents for Emotional Debt Negotiation
Usage: python experiments/run_coherence_agents.py --iterations 10 --scenarios 3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from models.coherence_agents import run_coherence_experiment
from utils.helpers import load_scenarios


def main():
    parser = argparse.ArgumentParser(description="Run Coherence Agents for Emotion Optimization")
    
    # Dataset selection
    parser.add_argument("--dataset_type", default="debt",
                       choices=["debt", "disaster", "student", "medical", "all"],
                       help="Which dataset to run")
    
    # Parameters
    parser.add_argument("--scenarios", type=int, default=3,
                       help="Number of scenarios to test")
    parser.add_argument("--iterations", type=int, default=10,
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
    parser.add_argument("--debtor_model_type", default="vanilla",
                       choices=["vanilla", "pressure", "victim", "threat"],
                       help="Debtor tactical approach")
    
    # Coherence specific parameters
    parser.add_argument("--coherence_weight", type=float, default=0.8,
                       help="Weight for coherence in emotion selection")
    
    # Output
    parser.add_argument("--out_dir", default="results/coherence_agents",
                       help="Base output directory for results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    
    args = parser.parse_args()
    
    # Dataset paths
    dataset_paths = {
        "debt": "../data/credit_recovery_scenarios.csv",
        "disaster": "../data/disaster_survivor_scenarios.csv",
        "student": "../data/education_sleep_scenarios.csv",
        "medical": "../data/hospital_surgery_scenarios.csv"
    }
    
    # Load scenarios
    if args.dataset_type == "all":
        all_scenarios = []
        for dataset_type, path in dataset_paths.items():
            if os.path.exists(path):
                from utils.preprocessing import preprocess_all_scenarios
                scenarios = preprocess_all_scenarios(
                    csv_path=path,
                    scenario_type=dataset_type,
                    max_scenarios=args.scenarios
                )
                all_scenarios.extend(scenarios)
        scenarios = all_scenarios[:args.scenarios]
        dataset_name = "multi_dataset"
    else:
        csv_path = dataset_paths.get(args.dataset_type)
        if not csv_path or not os.path.exists(csv_path):
            print(f"❌ Dataset file not found: {csv_path}")
            print("🔍 Available datasets:")
            for name, path in dataset_paths.items():
                status = "✅" if os.path.exists(path) else "❌"
                print(f"  {name}: {path} {status}")
            return
        
        from utils.preprocessing import preprocess_all_scenarios
        scenarios = preprocess_all_scenarios(
            csv_path=csv_path,
            scenario_type=args.dataset_type,
            max_scenarios=args.scenarios
        )
        dataset_name = args.dataset_type
    
    if not scenarios:
        print("❌ No scenarios loaded. Check dataset files.")
        return
    
    print("="*80)
    print("🎭 COHERENCE AGENTS FOR EMOTION OPTIMIZATION")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Iterations per scenario: {args.iterations}")
    print(f"Creditor Model: {args.model_creditor}")
    print(f"Debtor Model: {args.model_debtor}")
    print(f"Debtor Emotion: {args.debtor_emotion}")
    print(f"Debtor Tactics: {args.debtor_model_type}")
    print(f"Coherence Weight: {args.coherence_weight}")
    print("="*80)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{args.out_dir}/coherence_{dataset_name}_{timestamp}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Run coherence experiment
    print(f"\n🎭 Running Coherence Agent...")
    coherence_results = run_coherence_experiment(
        scenarios=scenarios,
        iterations=args.iterations,
        model_creditor=args.model_creditor,
        model_debtor=args.model_debtor,
        debtor_emotion=args.debtor_emotion,
        debtor_model_type=args.debtor_model_type,
        max_dialog_len=args.max_dialog_len,
        coherence_weight=args.coherence_weight,
        out_dir=out_dir
    )
    
    # Print results summary
    print(f"\n" + "="*80)
    print("🎭 COHERENCE AGENT RESULTS")
    print("="*80)
    
    summary = coherence_results.get('summary_statistics', {})
    success_rate = summary.get('success_rate', 0)
    coll_rate = summary.get('collection_rate', {})
    avg_coll_rate = coll_rate.get('mean', 0)
    rounds = summary.get('negotiation_rounds', {})
    avg_rounds = rounds.get('mean', 0)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Success Rate: {success_rate:.1%}")
    print(f"  Collection Rate: {avg_coll_rate:.3f}")
    print(f"  Average Rounds: {avg_rounds:.1f}")
    
    # Coherence-specific insights
    coherence_stats = coherence_results.get('coherence_statistics', {})
    if coherence_stats:
        avg_arc_length = coherence_stats.get('avg_arc_length', 0)
        phase_distribution = coherence_stats.get('phase_distribution', {})
        
        print(f"\n🎭 Coherence Insights:")
        print(f"  Average Arc Length: {avg_arc_length:.1f}")
        print(f"  Phase Distribution:")
        for phase, count in phase_distribution.items():
            print(f"    {phase}: {count} negotiations")
    
    # Statistical analysis
    stats_analysis = coherence_results.get('statistical_analysis')
    if stats_analysis:
        print(f"\n📈 Statistical Analysis:")
        success_stats = stats_analysis.get('success_rate', {})
        print(f"  Success Rate: {success_stats.get('rate', 0):.1%} (95% CI: [{success_stats.get('ci_lower', 0):.1%}, {success_stats.get('ci_upper', 0):.1%}])")
        
        if stats_analysis.get('collection_rate', {}).get('n_samples', 0) > 0:
            cr = stats_analysis['collection_rate']
            print(f"  Collection Rate: {cr.get('mean', 0):.3f} (95% CI: [{cr.get('ci_lower', 0):.3f}, {cr.get('ci_upper', 0):.3f}])")
        
        if stats_analysis.get('negotiation_rounds', {}).get('n_samples', 0) > 0:
            nr = stats_analysis['negotiation_rounds']
            print(f"  Negotiation Rounds: {nr.get('mean', 0):.1f} (95% CI: [{nr.get('ci_lower', 0):.1f}, {nr.get('ci_upper', 0):.1f}])")
    
    # Save comprehensive results
    comprehensive_results = {
        'experiment_info': {
            'agent_type': 'coherence',
            'dataset_type': dataset_name,
            'scenarios_count': len(scenarios),
            'iterations': args.iterations,
            'timestamp': timestamp,
            'config': vars(args)
        },
        'coherence_results': coherence_results
    }
    
    # Save comprehensive results
    results_file = f"{out_dir}/comprehensive_coherence_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"\n💾 Comprehensive results saved to: {results_file}")
    print(f"📁 Individual results in: {out_dir}/")
    
    print(f"\n📈 KEY FINDINGS FOR RESEARCH:")
    print(f"  1. Coherence agent achieved {success_rate:.1%} success rate on {dataset_name} dataset")
    print(f"  2. Average negotiation rounds: {avg_rounds:.1f}")
    print(f"  3. Collection efficiency: {avg_coll_rate:.3f}")
    print(f"  4. Emotional coherence provides psychological consistency in negotiations")
    print(f"  5. Average emotional arc length: {coherence_stats.get('avg_arc_length', 0):.1f}")
    
    print(f"\n✅ Coherence agent experiments completed!")


if __name__ == "__main__":
    main()