#!/usr/bin/env python3
"""
Two-Stage Emotional Negotiation Experiment System:
1. Training Stage: Train models and save emotional policies
2. Testing Stage: Test trained models against each other

Usage:
# Training stage
python experiments/run_training_testing.py --stage train --model_type evolutionary --scenarios 20 --generations 15

# Testing stage  
python experiments/run_training_testing.py --stage test --creditor_model evolutionary --debtor_model dqn --test_scenarios 10
"""

import argparse
import sys
import os
import time
import json
import random
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from utils.statistical_analysis import enhance_results_with_statistics, format_ci_results
from utils.preprocessing import preprocess_all_scenarios

load_dotenv()

# Import all models
from models.vanilla_model import VanillaBaselineModel, run_vanilla_baseline_experiment
from models.baseline_evolutionary import BaselineEvolutionaryOptimizer, run_baseline_experiment
from models.dqn_baseline import DQNBaseline, run_dqn_experiment
from models.hierarchical_evolutionary import HierarchicalBayesianOptimizer, run_hierarchical_experiment
from models.qlearning_baseline import QLearningBaseline, run_qlearning_experiment

class TrainedModelLoader:
    """Loads and manages trained emotional policies for testing"""
    
    def __init__(self, models_dir: str = "trained_models"):
        self.models_dir = models_dir
        self.loaded_models = {}
        
    def save_trained_model(self, model_name: str, model_data: Dict[str, Any], 
                          training_stats: Dict[str, Any]) -> str:
        """Save a trained model's emotional policy"""
        os.makedirs(self.models_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{self.models_dir}/{model_name}_{timestamp}.json"
        
        save_data = {
            'model_name': model_name,
            'model_data': model_data,
            'training_stats': training_stats,
            'timestamp': timestamp,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"💾 Saved trained {model_name} model to: {filepath}")
        return filepath
    
    def load_trained_model(self, model_name: str, model_file: str = None) -> Tuple[Any, Dict]:
        """Load a trained model for testing"""
        if model_file is None:
            # Find the most recent model file - be flexible with filename patterns
            if not os.path.exists(self.models_dir):
                raise FileNotFoundError(f"Models directory not found: {self.models_dir}")
            
            all_files = os.listdir(self.models_dir)
            model_files = [f for f in all_files 
                          if f.startswith(f"{model_name}_") and f.endswith('.json')]
            
            # If strict matching fails, try broader matching
            if not model_files:
                model_files = [f for f in all_files 
                              if model_name in f and f.endswith('.json')]
            
            if not model_files:
                available_models = [f for f in all_files if f.endswith('.json')]
                error_msg = f"No trained {model_name} model found in {self.models_dir}."
                if available_models:
                    error_msg += f"\nAvailable models: {available_models}"
                else:
                    error_msg += f"\nNo models found in {self.models_dir}."
                error_msg += f"\n\n💡 To train a {model_name} model, run:"
                error_msg += f"\n   python experiments/run_training_testing.py --stage train --creditor_method {model_name} --scenarios 10 --generations 15"
                raise FileNotFoundError(error_msg)
            model_file = sorted(model_files)[-1]  # Most recent
            model_file = os.path.join(self.models_dir, model_file)
        
        with open(model_file, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        model_data = save_data['model_data']
        training_stats = save_data['training_stats']
        
        print(f"📂 Found model file: {os.path.basename(model_file)}")
        
        # Create model instance based on type
        if model_name == 'vanilla':
            model = VanillaBaselineModel()
        elif model_name == 'evolutionary':
            model = BaselineEvolutionaryOptimizer()
            # Load transition matrix
            if 'transition_matrix' in model_data:
                model.base_transition_matrix = np.array(model_data['transition_matrix'])
                model.current_policy.transition_matrix = model.base_transition_matrix.copy()
            if 'best_sequence' in model_data:
                model.best_sequence = model_data['best_sequence']
        elif model_name == 'dqn':
            model = DQNBaseline()
            # Load Q-network weights (simplified - in practice you'd load the full network)
            if 'final_q_values' in model_data:
                model.final_q_values = model_data['final_q_values']
            if 'best_sequence' in model_data:
                model.best_sequence = model_data['best_sequence']
        elif model_name == 'hierarchical':
            model = HierarchicalBayesianOptimizer()
            # Load hierarchical matrices
            if 'group_matrix' in model_data:
                model.group_matrix = np.array(model_data['group_matrix'])
            if 'within_group_matrices' in model_data:
                for group, matrix in model_data['within_group_matrices'].items():
                    model.within_group_matrices[group] = np.array(matrix)
                model.base_matrix = model._reconstruct_base_matrix()
            if 'best_sequence' in model_data:
                model.best_sequence = model_data['best_sequence']
        elif model_name == 'qlearning':
            model = QLearningBaseline()
            # Load Q-table
            if 'q_table' in model_data:
                model.q_table = np.array(model_data['q_table'])
            if 'transition_matrix' in model_data:
                model.learned_transition_matrix = np.array(model_data['transition_matrix'])
            if 'best_sequence' in model_data:
                model.best_sequence = model_data['best_sequence']
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        print(f"🔄 Loaded trained {model_name} model from: {model_file}")
        self.loaded_models[model_name] = model
        return model, training_stats

def run_training_stage(
    creditor_method: str,  # The AI method being trained (evolutionary, dqn, etc.)
    debtor_method: str,    # The opponent AI method
    scenarios: List[Dict[str, Any]],
    **training_params
) -> Tuple[str, Dict[str, Any]]:
    """Run training stage for creditor method against debtor method"""
    
    print(f"\n🚀 TRAINING STAGE: {creditor_method.upper()} METHOD vs {debtor_method.upper()} OPPONENT")
    print("=" * 80)
    
    if creditor_method == 'vanilla':
        print("⚠️ Vanilla method doesn't require training (no learning)")
        return None, {}
    
    # Create training output directory with opponent info
    training_dir = f"results/training/{creditor_method}_vs_{debtor_method}"
    os.makedirs(training_dir, exist_ok=True)
    
    # Run training experiment
    training_start = time.time()
    
    if creditor_method == 'evolutionary':
        results = run_baseline_experiment(
            scenarios=scenarios,
            out_dir=training_dir,
            generations=training_params.get('generations', 10),
            population_size=training_params.get('population_size', 20),
            mutation_rate=training_params.get('mutation_rate', 0.1),
            crossover_rate=training_params.get('crossover_rate', 0.7),
            model_creditor=training_params.get('creditor_model', 'gpt-4o-mini'),
            model_debtor=training_params.get('debtor_model', 'gpt-4o-mini'),
            debtor_emotion="neutral",
            max_dialog_len=training_params.get('max_dialog_len', 30)
        )
        # Extract trained model data
        model_data = {
            'transition_matrix': results.get('final_stats', {}).get('base_transition_matrix', []),
            'best_sequence': results.get('final_best_sequence', {}).get('sequence'),
            'best_fitness': results.get('final_best_sequence', {}).get('fitness', 0),
            'opponent_method': debtor_method,
            'trained_as': 'creditor'
        }
        
    elif creditor_method == 'dqn':
        results = run_dqn_experiment(
            scenarios=scenarios,
            out_dir=training_dir,
            episodes=training_params.get('episodes', 100),
            episodes_per_scenario=training_params.get('episodes_per_scenario', 5),
            learning_rate=training_params.get('learning_rate', 1e-4),
            discount_factor=training_params.get('discount_factor', 0.95),
            exploration_rate=training_params.get('exploration_rate', 1.0),
            exploration_decay=training_params.get('exploration_decay', 0.995),
            batch_size=training_params.get('batch_size', 32),
            replay_buffer_size=training_params.get('replay_buffer_size', 10000),
            model_creditor=training_params.get('creditor_model', 'gpt-4o-mini'),
            model_debtor=training_params.get('debtor_model', 'gpt-4o-mini'),
            debtor_emotion="neutral",
            max_dialog_len=training_params.get('max_dialog_len', 30)
        )
        model_data = {
            'final_q_values': results.get('final_stats', {}).get('q_table', []),
            'best_sequence': results.get('best_sequence'),
            'best_reward': results.get('best_reward', 0),
            'opponent_method': debtor_method,
            'trained_as': 'creditor'
        }
        
    elif creditor_method == 'hierarchical':
        results = run_hierarchical_experiment(
            scenarios=scenarios,
            out_dir=training_dir,
            generations=training_params.get('generations', 10),
            negotiations_per_gen=training_params.get('negotiations_per_gen', 20),
            mutation_rate=training_params.get('mutation_rate', 0.1),
            crossover_rate=training_params.get('crossover_rate', 0.7),
            model_creditor=training_params.get('creditor_model', 'gpt-4o-mini'),
            model_debtor=training_params.get('debtor_model', 'gpt-4o-mini'),
            debtor_emotion="neutral",
            max_dialog_len=training_params.get('max_dialog_len', 30)
        )
        model_data = {
            'group_matrix': results.get('final_stats', {}).get('group_matrix', []),
            'within_group_matrices': results.get('final_stats', {}).get('within_group_matrices', {}),
            'base_matrix': results.get('final_stats', {}).get('base_matrix', []),
            'best_sequence': results.get('final_stats', {}).get('best_sequence'),
            'best_fitness': results.get('final_stats', {}).get('best_fitness', 0),
            'opponent_model': debtor_method,
            'trained_as': 'creditor'
        }
        
    elif creditor_method == 'qlearning':
        results = run_qlearning_experiment(
            scenarios=scenarios,
            out_dir=training_dir,
            episodes=training_params.get('episodes', 100),
            episodes_per_scenario=training_params.get('episodes_per_scenario', 5),
            learning_rate=training_params.get('learning_rate', 0.1),
            discount_factor=training_params.get('discount_factor', 0.9),
            exploration_rate=training_params.get('exploration_rate', 1.0),
            exploration_decay=training_params.get('exploration_decay', 0.995),
            temperature=training_params.get('temperature', 1.0),
            model_creditor=training_params.get('creditor_model', 'gpt-4o-mini'),
            model_debtor=training_params.get('debtor_model', 'gpt-4o-mini'),
            debtor_emotion="neutral",
            max_dialog_len=training_params.get('max_dialog_len', 30)
        )
        model_data = {
            'q_table': results.get('final_q_table', []),
            'transition_matrix': results.get('final_transition_matrix', []),
            'best_sequence': results.get('best_sequence'),
            'best_reward': results.get('best_reward', 0),
            'opponent_method': debtor_method,
            'trained_as': 'creditor'
        }
    else:
        raise ValueError(f"Unknown model type: {creditor_method}")
    
    training_time = time.time() - training_start
    
    # Training statistics
    training_stats = {
        'creditor_method': creditor_method,
        'debtor_method': debtor_method,
        'training_scenarios': len(scenarios),
        'training_time_minutes': training_time / 60,
        'success_rate': results.get('overall_success_rate', 0),
        'final_performance': results.get('performance', {}),
        'training_params': training_params
    }
    
    # Save trained model with opponent info
    model_loader = TrainedModelLoader(models_dir="trained_models")
    model_filename = f"{creditor_method}_as_creditor_vs_{debtor_method}"
    model_file = model_loader.save_trained_model(model_filename, model_data, training_stats)
    
    print(f"✅ Training completed in {training_time/60:.2f} minutes")
    print(f"📊 Final success rate: {results.get('overall_success_rate', 0):.1%}")
    
    return model_file, results

def run_testing_stage(
    creditor_model_name: str,
    debtor_model_name: str,
    test_scenarios: List[Dict[str, Any]],
    creditor_model_file: str = None,
    debtor_model_file: str = None,
    **testing_params
) -> Dict[str, Any]:
    """Run testing stage between two trained models"""
    
    print(f"\n🥊 TESTING STAGE: {creditor_model_name.upper()} vs {debtor_model_name.upper()}")
    print("=" * 60)
    
    # Load models
    model_loader = TrainedModelLoader(models_dir="trained_models")
    
    # Show available models for debugging
    if os.path.exists("trained_models"):
        available = [f for f in os.listdir("trained_models") if f.endswith('.json')]
        if available:
            print(f"📁 Available trained models: {len(available)}")
            for model in sorted(available)[:3]:  # Show first 3
                print(f"   - {model}")
            if len(available) > 3:
                print(f"   ... and {len(available)-3} more")
        else:
            print("📁 No trained models found - you may need to run training first")
    else:
        print("📁 trained_models directory not found")
    
    # Load creditor model
    if creditor_model_name == 'vanilla':
        creditor_model = VanillaBaselineModel()
        creditor_stats = {'model_type': 'vanilla', 'training': 'none_required'}
    else:
        creditor_model, creditor_stats = model_loader.load_trained_model(
            creditor_model_name, creditor_model_file
        )
    
    # Load debtor model  
    if debtor_model_name == 'vanilla':
        debtor_model = VanillaBaselineModel()
        debtor_stats = {'model_type': 'vanilla', 'training': 'none_required'}
    else:
        debtor_model, debtor_stats = model_loader.load_trained_model(
            debtor_model_name, debtor_model_file
        )
    
    # Create testing output directory
    test_dir = f"results/testing/{creditor_model_name}_vs_{debtor_model_name}"
    os.makedirs(test_dir, exist_ok=True)
    
    # Run head-to-head negotiations
    from llm.negotiator import DebtNegotiator
    
    testing_start = time.time()
    iterations = testing_params.get('iterations', 5)
    max_dialog_len = testing_params.get('max_dialog_len', 30)
    model_creditor = testing_params.get('model_creditor', 'gpt-4o-mini')
    model_debtor = testing_params.get('model_debtor', 'gpt-4o-mini')
    
    all_results = []
    scenario_results = {}
    
    print(f"🎯 Running {iterations} iterations × {len(test_scenarios)} scenarios")
    
    for iteration in range(iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
        
        iteration_results = []
        
        for i, scenario in enumerate(test_scenarios):
            print(f"  📋 Scenario {i+1}/{len(test_scenarios)}: ", end="")
            
            # Determine which model to use for emotion selection
            # Use trained model for creditor role, vanilla for debtor responses
            emotion_model_to_use = creditor_model
            debtor_emotion_setting = 'neutral'
            
            # If debtor is trained and creditor is vanilla, swap the roles in the negotiator
            if debtor_model_name != 'vanilla' and creditor_model_name == 'vanilla':
                print(f"      🔄 Using {debtor_model_name} model for emotion selection (debtor-focused)")
                emotion_model_to_use = debtor_model
                debtor_emotion_setting = 'trained'
            elif creditor_model_name != 'vanilla':
                print(f"      🧠 Using {creditor_model_name} model for emotion selection (creditor-focused)")
            
            negotiator = DebtNegotiator(
                config=scenario,
                emotion_model=emotion_model_to_use,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                debtor_emotion=debtor_emotion_setting,
                debtor_model_type='trained' if debtor_model_name != 'vanilla' else 'vanilla'
            )
            
            # Run negotiation
            result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
            
            # Add model information to result
            result['creditor_model'] = creditor_model_name
            result['debtor_model'] = debtor_model_name
            result['iteration'] = iteration + 1
            result['scenario_id'] = scenario.get('id', f'scenario_{i}')
            
            # Print result
            if result.get('final_state') == 'accept':
                days = result.get('collection_days', 0)
                amount = result.get('final_amount', 0)
                print(f"✅ Success (${amount:.0f} in {days} days)")
            else:
                print(f"❌ Failed ({result.get('final_state', 'unknown')})")
            
            iteration_results.append(result)
            all_results.append(result)
            
            # Cleanup after each negotiation
            try:
                negotiator.cleanup_models()
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")
        
        scenario_results[f'iteration_{iteration+1}'] = iteration_results
    
    testing_time = time.time() - testing_start
    
    # Calculate statistics
    successful = [r for r in all_results if r.get('final_state') == 'accept']
    success_rate = len(successful) / len(all_results) if all_results else 0
    
    avg_days = np.mean([r.get('collection_days', 0) for r in successful]) if successful else 0
    avg_rounds = np.mean([len(r.get('dialog', [])) for r in all_results]) if all_results else 0
    
    # Compile results
    results = {
        'experiment_type': 'model_vs_model_testing',
        'creditor_model': {
            'name': creditor_model_name,
            'training_stats': creditor_stats
        },
        'debtor_model': {
            'name': debtor_model_name,
            'training_stats': debtor_stats
        },
        'test_scenarios': len(test_scenarios),
        'iterations': iterations,
        'total_negotiations': len(all_results),
        'scenario_results': scenario_results,
        'performance': {
            'success_rate': success_rate,
            'avg_successful_days': float(avg_days),
            'avg_negotiation_rounds': float(avg_rounds),
            'successful_negotiations': len(successful),
            'failed_negotiations': len(all_results) - len(successful)
        },
        'timing': {
            'testing_time_minutes': testing_time / 60,
            'negotiations_per_minute': len(all_results) / (testing_time / 60) if testing_time > 0 else 0
        }
    }
    
    # Add statistical analysis with confidence intervals
    results = enhance_results_with_statistics(
        results, 
        all_results, 
        test_scenarios, 
        method="bootstrap"
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{test_dir}/testing_results_{timestamp}.json"
    
    with open(result_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary
    summary_file = f"{test_dir}/testing_summary_{timestamp}.txt"
    with open(summary_file, "w", encoding='utf-8') as f:
        f.write("MODEL vs MODEL TESTING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Creditor Model: {creditor_model_name}\n")
        f.write(f"Debtor Model: {debtor_model_name}\n")
        f.write(f"Test Scenarios: {len(test_scenarios)}\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Total Negotiations: {len(all_results)}\n\n")
        
        if 'statistical_analysis' in results:
            stat_analysis = results['statistical_analysis']
            sr_ci = stat_analysis['success_rate']['ci_95']
            f.write(f"Success Rate: {stat_analysis['success_rate']['mean']:.1%} ")
            f.write(f"(95% CI: [{sr_ci[0]:.1%}, {sr_ci[1]:.1%}])\n")
        else:
            f.write(f"Success Rate: {success_rate:.1%}\n")
        
        f.write(f"Average Successful Days: {avg_days:.1f}\n")
        f.write(f"Average Rounds: {avg_rounds:.1f}\n")
        f.write(f"Testing Time: {testing_time/60:.2f} minutes\n")
    
    print(f"\n📊 TESTING RESULTS:")
    print(f"   Success Rate: {success_rate:.1%}")
    print(f"   Avg Days (Successful): {avg_days:.1f}")
    print(f"   Avg Rounds: {avg_rounds:.1f}")
    print(f"   Testing Time: {testing_time/60:.2f} minutes")
    
    if 'statistical_analysis' in results:
        print(f"\n📈 STATISTICAL ANALYSIS:")
        print(format_ci_results(results['statistical_analysis']))
    
    print(f"\n💾 Results saved to: {result_file}")
    print(f"💾 Summary saved to: {summary_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Two-stage training and testing system")
    
    # Stage selection
    parser.add_argument("--stage", required=True, choices=["train", "test"],
                       help="Training or testing stage")
    
    # Common parameters
    parser.add_argument("--dataset_type", default="debt",
                       choices=["debt", "disaster", "student", "medical"],
                       help="Dataset type")
    parser.add_argument("--scenarios", type=int, default=10,
                       help="Number of scenarios (for training or testing)")
    parser.add_argument("--iterations", type=int, default=1,
                       help="Number of iterations per scenario (for testing)")
    
    # Training stage parameters
    parser.add_argument("--creditor_method", 
                       choices=["vanilla", "evolutionary", "dqn", "hierarchical", "qlearning"],
                       help="AI method to train as creditor (for training stage)")
    parser.add_argument("--debtor_method", default="vanilla",
                       choices=["vanilla", "evolutionary", "dqn", "hierarchical", "qlearning"],
                       help="Opponent AI method as debtor during training")
    parser.add_argument("--generations", type=int, default=10,
                       help="Evolutionary generations")
    parser.add_argument("--population_size", type=int, default=20,
                       help="Population size")
    parser.add_argument("--mutation_rate", type=float, default=0.1,
                       help="Mutation rate")
    parser.add_argument("--crossover_rate", type=float, default=0.7,
                       help="Crossover rate")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Training episodes (for RL models)")
    
    # Hierarchical-specific parameters
    parser.add_argument("--negotiations_per_gen", type=int, default=20,
                       help="Negotiations per generation (for hierarchical model)")
    parser.add_argument("--learning_rate", type=float, default=0.6,
                       help="Bayesian learning rate for hierarchical model")
    
    # RL-specific parameters (DQN and Q-Learning)
    parser.add_argument("--episodes_per_scenario", type=int, default=5,
                       help="Episodes per scenario before cycling (RL models)")
    parser.add_argument("--discount_factor", type=float, default=0.95,
                       help="Future reward discount factor (gamma) for RL")
    parser.add_argument("--exploration_rate", type=float, default=1.0,
                       help="Initial exploration rate (epsilon) for RL")
    parser.add_argument("--exploration_decay", type=float, default=0.995,
                       help="Exploration rate decay for RL")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size for DQN")
    parser.add_argument("--replay_buffer_size", type=int, default=10000,
                       help="Experience replay buffer size for DQN")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Softmax temperature for Q-Learning")
    
    # Testing stage parameters (keep same naming for consistency)
    parser.add_argument("--creditor_method_test",
                       choices=["vanilla", "evolutionary", "dqn", "hierarchical", "qlearning"],
                       help="Creditor AI method (for testing stage)")
    parser.add_argument("--debtor_method_test",
                       choices=["vanilla", "evolutionary", "dqn", "hierarchical", "qlearning"],
                       help="Debtor AI method (for testing stage)")
    parser.add_argument("--creditor_model_file", help="Specific creditor model file")
    parser.add_argument("--debtor_model_file", help="Specific debtor model file")
    
    # LLM model parameters (the actual language models)
    parser.add_argument("--creditor_model", default="gpt-4o-mini",
                       help="LLM for creditor agent (deepseek-chat, gpt-4o-mini, etc.)")
    parser.add_argument("--debtor_model", default="gpt-4o-mini",
                       help="LLM for debtor agent (deepseek-chat, gpt-4o-mini, etc.)")
    parser.add_argument("--max_dialog_len", type=int, default=30,
                       help="Max dialog length")
    
    # Seed parameters for reproducible train/test splits
    parser.add_argument("--train_seed", type=int, default=42,
                       help="Random seed for training scenario selection")
    parser.add_argument("--test_seed", type=int, default=42,
                       help="Random seed for testing scenario selection")
    
    args = parser.parse_args()
    
    # Define dataset paths
    dataset_paths = {
        "debt": "../data/credit_recovery_scenarios.csv",
        "disaster": "../data/disaster_survivor_scenarios.csv", 
        "student": "../data/education_sleep_scenarios.csv",
        "medical": "../data/hospital_surgery_scenarios.csv"
    }
    
    csv_path = dataset_paths.get(args.dataset_type)
    if not csv_path or not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        return
    
    print(f"\n🚀 TWO-STAGE EXPERIMENT SYSTEM")
    print(f"📊 Dataset: {args.dataset_type}")
    print(f"🔄 Stage: {args.stage}")
    
    if args.stage == "train":
        if not args.creditor_method:
            print("❌ --creditor_method required for training stage")
            return
        
        # Preprocess scenarios for training with seed
        scenarios = preprocess_all_scenarios(
            csv_path=csv_path,
            scenario_type=args.dataset_type,
            n_scenarios=args.scenarios,
            seed=args.train_seed
        )
        
        # Training parameters with all specialized parameters
        training_params = {
            # Common parameters
            'creditor_model': args.creditor_model,
            'debtor_model': args.debtor_model,
            'max_dialog_len': args.max_dialog_len,
            
            # Evolutionary parameters
            'generations': args.generations,
            'population_size': args.population_size,
            'mutation_rate': args.mutation_rate,
            'crossover_rate': args.crossover_rate,
            
            # Hierarchical parameters
            'negotiations_per_gen': args.negotiations_per_gen,
            'learning_rate': args.learning_rate,
            
            # RL parameters
            'episodes': args.episodes,
            'episodes_per_scenario': args.episodes_per_scenario,
            'discount_factor': args.discount_factor,
            'exploration_rate': args.exploration_rate,
            'exploration_decay': args.exploration_decay,
            'batch_size': args.batch_size,
            'replay_buffer_size': args.replay_buffer_size,
            'temperature': args.temperature
        }
        
        # Run training
        model_file, results = run_training_stage(
            args.creditor_method, 
            args.debtor_method,
            scenarios, 
            **training_params
        )
        
    elif args.stage == "test":
        if not args.creditor_method_test or not args.debtor_method_test:
            print("❌ --creditor_method_test and --debtor_method_test required for testing stage")
            return
        
        # Use different scenarios for testing with test_seed
        test_scenarios = preprocess_all_scenarios(
            csv_path=csv_path,
            scenario_type=args.dataset_type,
            n_scenarios=args.scenarios,
            seed=args.test_seed,
            offset=args.scenarios if hasattr(args, 'train_seed') and args.train_seed == args.test_seed else 0
        )
        
        # Testing parameters
        testing_params = {
            'iterations': args.iterations,
            'model_creditor': args.creditor_model,
            'model_debtor': args.debtor_model,
            'max_dialog_len': args.max_dialog_len
        }
        
        # Run testing
        results = run_testing_stage(
            args.creditor_method_test,
            args.debtor_method_test,
            test_scenarios,
            args.creditor_model_file,
            args.debtor_model_file,
            **testing_params
        )

if __name__ == "__main__":
    main()