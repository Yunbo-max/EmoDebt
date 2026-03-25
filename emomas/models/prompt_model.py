"""
Prompt-Based Model - LLM Negotiation with Emotion Selection Prompts
Adds prompts to creditor to select one of seven emotions but no optimization
"""

import numpy as np
import random
from typing import Dict, List, Any
from models.base_model import BaseEmotionModel
import json
from datetime import datetime
import os
from utils.statistical_analysis import enhance_results_with_statistics, analyze_negotiation_results, format_ci_results

# Seven emotions for selection
EMOTIONS = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']

class PromptBasedModel(BaseEmotionModel):
    """
    Prompt-based model that asks LLM to select emotions but doesn't optimize.
    Simply prompts creditor to choose from seven emotions each round.
    """
    
    def __init__(self):
        self.negotiation_count = 0
        self.success_history = []
        self.emotion_history = []
        self.current_emotion = 'neutral'
        
    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prompt-based emotion selection - randomly choose an emotion and prompt creditor
        """
        self.negotiation_count += 1
        
        # Randomly select an emotion (no optimization)
        emotion = random.choice(EMOTIONS)
        self.emotion_history.append(emotion)
        self.current_emotion = emotion
        
        # Create emotion-specific prompts
        emotion_prompts = {
            "happy": "Use a positive, optimistic tone. Show enthusiasm and focus on positive outcomes and solutions.",
            "surprising": "Use an engaging, unexpected approach. Introduce surprising facts or creative solutions to catch attention.", 
            "angry": "Use a firm, assertive tone. Show disappointment about the situation but maintain professionalism.",
            "sad": "Use an empathetic, understanding tone. Show compassion for the debtor's difficult situation.",
            "disgust": "Use a disappointed tone. Express concern about the lack of payment responsibility.",
            "fear": "Use a cautious, concerned tone. Express worry about potential consequences if payment isn't resolved.",
            "neutral": "Use a balanced, professional tone. Keep the conversation factual and businesslike."
        }
        
        return {
            "emotion": emotion,
            "emotion_text": emotion_prompts.get(emotion, "Use a professional tone"),
            "temperature": 0.7,
            "strategy": "prompt_based_random",
            "use_emotion": True,  # Use emotion prompts
            "debtor_emotion": state.get('debtor_emotion', 'neutral')
        }
    
    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        """
        Prompt model doesn't learn - just tracks basic stats and emotion usage
        """
        # Store basic outcome info
        success = negotiation_result.get('final_state') == 'accept'
        self.success_history.append(success)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic model statistics including emotion distribution"""
        success_rate = 0.0
        if self.success_history:
            success_rate = sum(self.success_history) / len(self.success_history)
        
        # Calculate emotion distribution
        emotion_distribution = {}
        for emotion in EMOTIONS:
            emotion_distribution[emotion] = self.emotion_history.count(emotion)
            
        return {
            'model_type': 'prompt_based_random',
            'negotiation_count': self.negotiation_count,
            'success_rate': success_rate,
            'total_negotiations': len(self.success_history),
            'current_emotion': self.current_emotion,
            'emotion_distribution': emotion_distribution,
            'uses_emotion_prompts': True,
            'no_optimization': True
        }
    
    def reset(self) -> None:
        """Reset for new scenario (keep learning history)"""
        self.current_emotion = 'neutral'
        # Keep emotion_history and success_history for analysis

def run_prompt_based_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    out_dir: str = "results/prompt_based"
) -> Dict[str, Any]:
    """
    Run prompt-based experiment - LLM negotiation with emotion prompts
    """
    
    from llm.negotiator import DebtNegotiator
    
    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Create prompt-based model
    model = PromptBasedModel()
    
    results = {
        'experiment_type': 'prompt_based',
        'model_type': 'prompt_based_random',
        'iterations': iterations,
        'scenarios_used': [s['id'] if 'id' in s else f"scenario_{i}" for i, s in enumerate(scenarios)],
        'config': {
            'model_creditor': model_creditor,
            'model_debtor': model_debtor,
            'debtor_emotion': debtor_emotion,
            'max_dialog_len': max_dialog_len,
            'available_emotions': EMOTIONS
        },
        'iteration_results': {}
    }
    
    all_negotiation_results = []
    
    print(f"🎯 Starting Prompt-Based Experiment")
    print(f"📊 Scenarios: {len(scenarios)}, Iterations: {iterations}")
    print(f"🤖 Models: {model_creditor} vs {model_debtor}")
    print(f"😊 Strategy: Random emotion prompts (7 emotions)")
    print(f"🎭 Emotions: {', '.join(EMOTIONS)}")
    print("-" * 50)
    
    for iteration in range(iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
        
        iteration_results = {
            'iteration': iteration + 1,
            'scenario_results': [],
            'model_stats': model.get_stats()
        }
        
        for i, scenario in enumerate(scenarios):
            print(f"  📋 Scenario {i+1}/{len(scenarios)}: ", end="")
            
            # Create negotiator with prompt-based model
            negotiator = DebtNegotiator(
                config=scenario,
                emotion_model=model,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                debtor_emotion=debtor_emotion,
                debtor_model_type=debtor_model_type
            )
            
            # Run negotiation
            result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
            
            # Print quick result with emotion used
            emotion_used = model.current_emotion
            if result.get('final_state') == 'accept':
                days = result.get('collection_days', 0)
                amount = result.get('final_amount', 0)
                print(f"✅ Success ({emotion_used}) - ${amount:.0f} in {days} days")
            else:
                print(f"❌ Failed ({emotion_used}) - {result.get('final_state', 'unknown')}")
            
            iteration_results['scenario_results'].append(result)
            all_negotiation_results.append(result)
            
            # Update model (minimal for prompt-based)
            model.update_model(result)
            
            # Clean up GPU memory after each negotiation (important for offline models)
            try:
                negotiator.cleanup_models()
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")
        
        # Store iteration results
        results['iteration_results'][f'iteration_{iteration+1}'] = iteration_results
    
    # Calculate final performance metrics
    successful = [r for r in all_negotiation_results if r.get('final_state') == 'accept']
    failed = [r for r in all_negotiation_results if r.get('final_state') != 'accept']
    
    if all_negotiation_results:
        success_rate = len(successful) / len(all_negotiation_results)
        
        # Calculate collection rates for successful negotiations
        collection_rates = []
        for r in successful:
            target_days = r.get('creditor_target_days', 30)
            actual_days = r.get('collection_days', target_days)
            if target_days > 0:
                collection_rate = min(1.0, target_days / actual_days)  # Higher is better
                collection_rates.append(collection_rate)
        
        avg_collection_rate = np.mean(collection_rates) if collection_rates else 0.0
        avg_rounds = np.mean([len(r.get('dialog', [])) for r in all_negotiation_results])
    else:
        success_rate = 0
        avg_collection_rate = 0.0
        avg_rounds = 0
    
    # Enhance results with comprehensive statistical analysis
    results = enhance_results_with_statistics(
        results, 
        all_negotiation_results, 
        scenarios, 
        method="bootstrap"
    )
    
    # Final statistics
    results['final_stats'] = model.get_stats()
    results['performance'] = {
        'success_rate': success_rate,
        'avg_collection_rate': float(avg_collection_rate),
        'avg_negotiation_rounds': float(avg_rounds),
        'total_negotiations': len(all_negotiation_results),
        'successful_negotiations': len(successful),
        'failed_negotiations': len(failed)
    }
    
    # Detailed breakdown
    failure_reasons = {}
    for result in failed:
        reason = result.get('final_state', 'unknown')
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    # Emotion effectiveness analysis
    emotion_success = {}
    for emotion in EMOTIONS:
        emotion_results = []
        for i, result in enumerate(all_negotiation_results):
            if i < len(model.emotion_history) and model.emotion_history[i] == emotion:
                emotion_results.append(result.get('final_state') == 'accept')
        
        if emotion_results:
            emotion_success[emotion] = {
                'success_rate': sum(emotion_results) / len(emotion_results),
                'count': len(emotion_results)
            }
    
    results['analysis'] = {
        'failure_breakdown': failure_reasons,
        'emotion_effectiveness': emotion_success,
        'success_patterns': {
            'avg_rounds_successful': float(np.mean([len(r.get('dialog', [])) for r in successful])) if successful else 0,
            'avg_rounds_failed': float(np.mean([len(r.get('dialog', [])) for r in failed])) if failed else 0
        }
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/prompt_based_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save human-readable summary
    summary_file = f"{out_dir}/summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("PROMPT-BASED EXPERIMENT SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment Type: Prompt-Based (Random Emotion Selection)\n")
        f.write(f"Total Negotiations: {len(all_negotiation_results)}\n")
        f.write(f"Success Rate: {success_rate:.1%}\n")
        f.write(f"Average Collection Rate: {avg_collection_rate:.3f}\n")
        f.write(f"Average Rounds: {avg_rounds:.1f}\n\n")
        
        f.write("EMOTION DISTRIBUTION:\n")
        emotion_dist = results['final_stats']['emotion_distribution']
        for emotion, count in emotion_dist.items():
            percentage = (count / sum(emotion_dist.values())) * 100 if sum(emotion_dist.values()) > 0 else 0
            f.write(f"  {emotion.title()}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nEMOTION EFFECTIVENESS:\n")
        for emotion, stats in emotion_success.items():
            f.write(f"  {emotion.title()}: {stats['success_rate']:.1%} ({stats['count']} uses)\n")
        
        f.write("\nFAILURE BREAKDOWN:\n")
        for reason, count in failure_reasons.items():
            f.write(f"  {reason}: {count} ({count/len(failed)*100:.1f}%)\n")
        
        f.write(f"\nModel Configuration:\n")
        f.write(f"  Creditor Model: {model_creditor}\n")
        f.write(f"  Debtor Model: {model_debtor}\n")
        f.write(f"  Emotion Strategy: Random Selection from 7 Emotions\n")
        f.write(f"  Max Dialog Length: {max_dialog_len}\n")
    
    print(f"\n📊 PROMPT-BASED RESULTS")
    print("=" * 50)
    
    # Print statistical analysis with confidence intervals
    if 'statistical_analysis' in results:
        formatted_stats = format_ci_results(results['statistical_analysis'])
        if formatted_stats:
            print(f"📊 {formatted_stats}")
    
    print(f"📈 Avg Collection Rate: {avg_collection_rate:.3f}")
    print(f"💬 Avg Negotiation Rounds: {avg_rounds:.1f}")
    
    # Show emotion effectiveness
    print(f"\n🎭 EMOTION EFFECTIVENESS")
    for emotion, stats in emotion_success.items():
        print(f"  {emotion.title()}: {stats['success_rate']:.1%} ({stats['count']} uses)")
    
    print(f"\n📁 Results saved to: {result_file}")
    print(f"📄 Summary saved to: {summary_file}")
    
    return results