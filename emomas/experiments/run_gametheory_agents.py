#!/usr/bin/env python3
"""
Run Game Theory Agents for Emotional Debt Negotiation
Usage: python experiments/run_gametheory_agents.py --agent_type nash --iterations 10 --scenarios 3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from models.gametheory_agents import (
    run_nash_equilibrium_experiment, 
    run_minimax_experiment, 
    run_strategic_dominance_experiment
)
from utils.helpers import load_scenarios


def main():
    parser = argparse.ArgumentParser(description="Run Game Theory Agents for Emotion Optimization")
    
    # Agent selection
    parser.add_argument("--agent_type", default="nash",
                       choices=["nash", "minimax", "dominance", "all"],
                       help="Which game theory agent to run")
    
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
    
    # Game theory specific parameters
    parser.add_argument("--minimax_depth", type=int, default=2,
                       help="Search depth for Minimax algorithm")
    parser.add_argument("--nash_exploration", type=float, default=0.1,
                       help="Exploration rate for Nash equilibrium")
    
    # Output
    parser.add_argument("--out_dir", default="results/gametheory_agents",
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
    print("♟️  GAME THEORY AGENTS FOR EMOTION OPTIMIZATION")
    print("="*80)
    print(f"Agent Type: {args.agent_type}")
    print(f"Dataset: {dataset_name}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Iterations per scenario: {args.iterations}")
    print(f"Creditor Model: {args.model_creditor}")
    print(f"Debtor Model: {args.model_debtor}")
    print(f"Debtor Emotion: {args.debtor_emotion}")
    print(f"Debtor Tactics: {args.debtor_model_type}")
    if args.agent_type in ["minimax", "all"]:
        print(f"Minimax Depth: {args.minimax_depth}")
    if args.agent_type in ["nash", "all"]:
        print(f"Nash Exploration: {args.nash_exploration}")
    print("="*80)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{args.out_dir}/{args.agent_type}_{dataset_name}_{timestamp}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Run experiments based on agent type
    all_results = {}
    
    if args.agent_type == "nash" or args.agent_type == "all":
        print(f"\n⚖️  Running Nash Equilibrium Agent...")
        nash_results = run_nash_equilibrium_experiment(
            scenarios=scenarios,
            iterations=args.iterations,
            model_creditor=args.model_creditor,
            model_debtor=args.model_debtor,
            debtor_emotion=args.debtor_emotion,
            debtor_model_type=args.debtor_model_type,
            max_dialog_len=args.max_dialog_len,
            out_dir=f"{out_dir}/nash"
        )
        all_results['nash'] = nash_results
    
    if args.agent_type == "minimax" or args.agent_type == "all":
        print(f"\n🎯 Running Minimax Agent...")
        minimax_results = run_minimax_experiment(
            scenarios=scenarios,
            iterations=args.iterations,
            model_creditor=args.model_creditor,
            model_debtor=args.model_debtor,
            debtor_emotion=args.debtor_emotion,
            debtor_model_type=args.debtor_model_type,
            max_dialog_len=args.max_dialog_len,
            depth=args.minimax_depth,
            out_dir=f"{out_dir}/minimax"
        )
        all_results['minimax'] = minimax_results
    
    if args.agent_type == "dominance" or args.agent_type == "all":
        print(f"\n💪 Running Strategic Dominance Agent...")
        dominance_results = run_strategic_dominance_experiment(
            scenarios=scenarios,
            iterations=args.iterations,
            model_creditor=args.model_creditor,
            model_debtor=args.model_debtor,
            debtor_emotion=args.debtor_emotion,
            debtor_model_type=args.debtor_model_type,
            max_dialog_len=args.max_dialog_len,
            out_dir=f"{out_dir}/dominance"
        )
        all_results['dominance'] = dominance_results
    
    # Print comprehensive comparison
    if len(all_results) > 1:
        print(f"\n" + "="*80)
        print("♟️  GAME THEORY AGENT COMPARISON")
        print("="*80)
        
        print(f"\n{'Agent':<20} {'Success Rate':<15} {'Collection Rate':<20} {'Avg Rounds':<15}")
        print("-" * 70)
        
        for agent_name, results in all_results.items():
            summary = results.get('summary_statistics', {})
            success_rate = summary.get('success_rate', 0)
            coll_rate = summary.get('collection_rate', {})
            avg_coll_rate = coll_rate.get('mean', 0)
            rounds = summary.get('negotiation_rounds', {})
            avg_rounds = rounds.get('mean', 0)
            
            print(f"{agent_name:<20} {success_rate:<15.1%} {avg_coll_rate:<20.3f} {avg_rounds:<15.1f}")
        
        # Best performing agent
        best_agent = max(all_results.keys(), key=lambda x: all_results[x].get('summary_statistics', {}).get('success_rate', 0))
        best_success_rate = all_results[best_agent].get('summary_statistics', {}).get('success_rate', 0)
        
        print(f"\n🏆 Best performing agent: {best_agent} ({best_success_rate:.1%} success rate)")
        
        # Agent-specific insights
        print(f"\n🔍 AGENT-SPECIFIC INSIGHTS:")
        
        for agent_name, results in all_results.items():
            if agent_name == "nash" and 'nash_statistics' in results:
                stats = results['nash_statistics']
                print(f"  ⚖️  Nash: {stats['equilibrium_updates']} equilibrium updates, {stats['experience_count']} experiences")
                creditor_strategy = stats.get('final_creditor_strategy', [])
                if creditor_strategy:
                    dominant_emotion_idx = creditor_strategy.index(max(creditor_strategy))
                    emotions = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
                    dominant_emotion = emotions[dominant_emotion_idx] if dominant_emotion_idx < len(emotions) else 'unknown'
                    print(f"      Dominant strategy: {dominant_emotion} ({max(creditor_strategy):.3f})")
            
            elif agent_name == "minimax" and 'minimax_statistics' in results:
                stats = results['minimax_statistics']
                print(f"  🎯 Minimax: {stats['evaluations_made']} evaluations, depth-{stats['search_depth']} search")
                print(f"      Decision history: {len(stats['decision_history'])} decisions made")
            
            elif agent_name == "dominance" and 'dominance_statistics' in results:
                stats = results['dominance_statistics']
                strategy_perf = stats.get('strategy_performance', {})
                best_strategy = max(strategy_perf.keys(), key=lambda x: strategy_perf[x]['wins'] / max(1, strategy_perf[x]['total']))
                best_win_rate = strategy_perf[best_strategy]['wins'] / max(1, strategy_perf[best_strategy]['total'])
                print(f"  💪 Dominance: Best strategy {best_strategy} ({best_win_rate:.1%} win rate)")
                print(f"      Domination history: {len(stats['domination_history'])} strategic decisions")
    
    # Save comprehensive results
    comprehensive_results = {
        'experiment_info': {
            'agent_types': list(all_results.keys()),
            'dataset_type': dataset_name,
            'scenarios_count': len(scenarios),
            'iterations': args.iterations,
            'timestamp': timestamp,
            'config': vars(args)
        },
        'agent_results': all_results,
        'comparison': {}
    }
    
    if len(all_results) > 1:
        # Add comparison metrics
        comparison = {}
        for agent_name, results in all_results.items():
            summary = results.get('summary_statistics', {})
            comparison[agent_name] = {
                'success_rate': summary.get('success_rate', 0),
                'collection_rate_mean': summary.get('collection_rate', {}).get('mean', 0),
                'avg_negotiation_rounds': summary.get('negotiation_rounds', {}).get('mean', 0),
                'total_negotiations': summary.get('total_negotiations', 0)
            }
        
        comprehensive_results['comparison'] = comparison
    
    # Save comprehensive results
    results_file = f"{out_dir}/comprehensive_gametheory_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"\n💾 Comprehensive results saved to: {results_file}")
    print(f"📁 Individual results in: {out_dir}/")
    
    print(f"\n📈 KEY FINDINGS FOR RESEARCH:")
    if len(all_results) == 1:
        agent_name = list(all_results.keys())[0]
        results = all_results[agent_name]
        summary = results.get('summary_statistics', {})
        success_rate = summary.get('success_rate', 0)
        print(f"  1. {agent_name.upper()} achieved {success_rate:.1%} success rate on {dataset_name} dataset")
        print(f"  2. Average negotiation rounds: {summary.get('negotiation_rounds', {}).get('mean', 0):.1f}")
        print(f"  3. Collection efficiency: {summary.get('collection_rate', {}).get('mean', 0):.3f}")
    else:
        print(f"  1. Compared {len(all_results)} game theory agents on {dataset_name} dataset")
        success_rates = [results.get('summary_statistics', {}).get('success_rate', 0) for results in all_results.values()]
        print(f"  2. Success rates range: {min(success_rates):.1%} - {max(success_rates):.1%}")
        print(f"  3. Best agent: {best_agent} outperformed others")
        print(f"  4. Game theory provides strategic emotion selection framework")
    
    # Strategic insights
    if "nash" in all_results:
        nash_stats = all_results["nash"].get('nash_statistics', {})
        print(f"  5. Nash equilibrium found optimal mixed strategies after {nash_stats.get('equilibrium_updates', 0)} updates")
    
    if "minimax" in all_results:
        minimax_stats = all_results["minimax"].get('minimax_statistics', {})
        print(f"  6. Minimax evaluated {minimax_stats.get('evaluations_made', 0)} scenarios with depth-{minimax_stats.get('search_depth', 0)} lookahead")
    
    if "dominance" in all_results:
        print(f"  7. Strategic dominance identified emotion hierarchies in negotiation contexts")
    
    print(f"\n✅ Game theory agent experiments completed!")


if __name__ == "__main__":
    main()