"""
Coherence-Based Emotion Models for Debt Negotiation
Focuses on psychological consistency, emotional transitions, and behavioral memory patterns
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from models.base_model import BaseEmotionModel
from datetime import datetime
from collections import defaultdict, deque
from utils.statistical_analysis import enhance_results_with_statistics, analyze_negotiation_results, format_ci_results

class CoherenceEmotionModel(BaseEmotionModel):
    """
    Simple coherence-based emotion model focusing on psychological plausibility and phase appropriateness
    """
    
    def __init__(self, coherence_weight: float = 0.8):
        super().__init__()
        
        self.emotions = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
        
        # Simple state tracking
        self.current_emotional_arc = []
        self.negotiation_phase = "opening"
        
        # For compatibility with base class
        self.pattern_success_rates = defaultdict(list)
        self.transition_success_rates = defaultdict(list)
        
        # Simple credible transitions
        self.credible_transitions = {
            'neutral': ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear'],  # Neutral can go anywhere
            'happy': ['neutral', 'surprising', 'sad'],  # Happy avoids jarring jumps
            'surprising': ['happy', 'neutral', 'fear', 'sad'],  # Surprise can lead many places
            'angry': ['disgust', 'neutral', 'sad'],  # Anger can cool down or shift
            'sad': ['neutral', 'fear', 'angry', 'happy'],  # Sadness has many paths
            'disgust': ['angry', 'neutral', 'sad'],  # Disgust similar to anger
            'fear': ['sad', 'neutral', 'surprising', 'angry']  # Fear can shift various ways
        }
    
    def select_emotion(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Select emotion using simple coherence rules with exploration incentives"""
        
        # Extract context
        round_num = model_state.get('round', 1)
        debtor_emotion = self._normalize_emotion(model_state.get('debtor_emotion', 'neutral'))
        gap_size = model_state.get('gap_size', 0)
        debt_amount = model_state.get('debt_amount', 10000)
        
        # Determine negotiation phase
        self.negotiation_phase = self._determine_phase(round_num)
        
        # Simple coherence scoring
        emotion_scores = {}
        
        for emotion in self.emotions:
            score = 0.5  # Base score
            
            # Phase appropriateness (simple rules)
            if self.negotiation_phase == "opening" and emotion in ['neutral', 'happy']:
                score += 0.3
            elif self.negotiation_phase == "development" and emotion in ['neutral', 'surprising', 'happy']:
                score += 0.2
            elif self.negotiation_phase == "intensive" and emotion in ['angry', 'disgust', 'neutral']:
                score += 0.3
            elif self.negotiation_phase == "closing" and emotion in ['neutral', 'angry']:
                score += 0.3
            
            # Psychological plausibility (avoid jarring transitions)
            if self.current_emotional_arc:
                last_emotion = self.current_emotional_arc[-1]
                if self._is_credible_transition(last_emotion, emotion):
                    score += 0.2
                else:
                    score -= 0.3
            
            # Context appropriateness
            if gap_size > 25 and emotion in ['angry', 'disgust']:  # Large gap = assertive
                score += 0.2
            elif gap_size < 10 and emotion in ['happy', 'neutral']:  # Small gap = cooperative
                score += 0.2
            
            # 🔥 EXPLORATION INCENTIVES - Prevent getting stuck in one emotion
            repetition_penalty = self._calculate_repetition_penalty(emotion)
            diversity_bonus = self._calculate_diversity_bonus(emotion)
            
            score += diversity_bonus - repetition_penalty
            
            emotion_scores[emotion] = max(0.1, score)
        
        # Select best emotion
        best_emotion = max(emotion_scores.keys(), key=emotion_scores.get)
        confidence = emotion_scores[best_emotion]
        
        # Update state
        self.current_emotional_arc.append(best_emotion)
        if len(self.current_emotional_arc) > 5:  # Keep shorter history
            self.current_emotional_arc.pop(0)
        
        return {
            'emotion': best_emotion,
            'confidence': confidence,
            'emotional_arc': self.current_emotional_arc.copy(),
            'negotiation_phase': self.negotiation_phase,
            'emotion_text': self._get_coherent_prompt(best_emotion),
            'temperature': 0.7,
            'reasoning': f"Simple coherence: {self.negotiation_phase} phase, credible transition, diversity={self._calculate_diversity_bonus(best_emotion):.2f}, repetition_penalty={self._calculate_repetition_penalty(best_emotion):.2f}"
        }
    
    def _calculate_repetition_penalty(self, emotion: str) -> float:
        """Calculate penalty for repeating the same emotion too much"""
        if not self.current_emotional_arc:
            return 0.0
        
        # Count recent occurrences of this emotion
        recent_arc = self.current_emotional_arc[-3:]  # Look at last 3 emotions
        repetition_count = recent_arc.count(emotion)
        
        # Heavy penalty for 2+ repetitions in a row
        if repetition_count >= 2:
            return 0.4  # Strong penalty
        elif repetition_count == 1 and recent_arc and recent_arc[-1] == emotion:
            return 0.2  # Moderate penalty for immediate repetition
        
        return 0.0
    
    def _calculate_diversity_bonus(self, emotion: str) -> float:
        """Calculate bonus for emotional diversity"""
        if not self.current_emotional_arc:
            return 0.1  # Small bonus for starting
        
        # Bonus for emotions we haven't used recently
        recent_arc = self.current_emotional_arc[-4:]  # Look at last 4 emotions
        
        if emotion not in recent_arc:
            return 0.3  # Good diversity bonus
        elif emotion not in self.current_emotional_arc[-2:]:
            return 0.15  # Small diversity bonus
        
        return 0.0
    
    def _normalize_emotion(self, emotion: str) -> str:
        """Normalize emotion encoding from various formats"""
        if isinstance(emotion, str) and len(emotion) <= 2:
            emotion_map = {
                'A': 'angry', 'S': 'sad', 'D': 'disgust', 'F': 'fear',
                'J': 'happy', 'Su': 'surprising', 'N': 'neutral'
            }
            return emotion_map.get(emotion, 'neutral')
        return emotion.lower() if emotion.lower() in self.emotions else 'neutral'
    
    def _determine_phase(self, round_num: int) -> str:
        """Determine current negotiation phase"""
        if round_num <= 3:
            return "opening"
        elif round_num <= 7:
            return "development"
        elif round_num <= 12:
            return "intensive"
        else:
            return "closing"
    
    def _is_credible_transition(self, from_emotion: str, to_emotion: str) -> bool:
        """Simple check for psychologically credible emotional transitions"""
        if from_emotion == to_emotion:
            return True  # Staying the same is always credible
        
        return to_emotion in self.credible_transitions.get(from_emotion, [])
    
    def _get_coherent_prompt(self, emotion: str) -> str:
        """Get simple, coherent prompt for the selected emotion with exploration context"""
        
        base_prompts = {
            'happy': "Be optimistic and cooperative. Show genuine enthusiasm for finding a solution.",
            'surprising': "Express curiosity and ask clarifying questions about the situation.",
            'angry': "Show controlled frustration while remaining professional and solution-focused.",
            'sad': "Express sincere concern about the difficult circumstances.",
            'disgust': "Show dissatisfaction with current terms while seeking better alternatives.",
            'fear': "Express legitimate concerns about consequences and seek reassurance.",
            'neutral': "Maintain a calm, professional, fact-based approach."
        }
        
        prompt = base_prompts.get(emotion, base_prompts['neutral'])
        
        # Add phase-appropriate guidance
        if self.negotiation_phase == "opening":
            prompt += " Focus on building rapport and understanding."
        elif self.negotiation_phase == "development":
            prompt += " Work to explore options and build agreement."
        elif self.negotiation_phase == "intensive":
            prompt += " Apply appropriate pressure while maintaining relationship."
        elif self.negotiation_phase == "closing":
            prompt += " Focus on finalizing terms and ensuring commitment."
        
        # 🎭 Add exploration guidance to encourage emotional variety
        if len(self.current_emotional_arc) >= 2:
            recent_emotions = self.current_emotional_arc[-2:]
            if len(set(recent_emotions)) == 1:  # Been using same emotion
                prompt += " Avoid staying in the same emotional state too long - consider shifting your approach to maintain engagement and authenticity."
            else:
                prompt += " Build on your recent emotional flow to create a coherent but varied interaction."
        
        return prompt
    
    def get_coherence_metrics(self) -> Dict[str, Any]:
        """Get simple coherence metrics for analysis"""
        return {
            'current_emotional_arc': self.current_emotional_arc,
            'negotiation_phase': self.negotiation_phase,
            'arc_length': len(self.current_emotional_arc),
            'last_emotion': self.current_emotional_arc[-1] if self.current_emotional_arc else 'none'
        }
    
    def reset_for_new_negotiation(self):
        """Reset state for a new negotiation while preserving learned knowledge"""
        self.current_emotional_arc = []
        self.negotiation_phase = "opening"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics (required by BaseEmotionModel)"""
        return self.get_coherence_metrics()
    
    def reset(self):
        """Reset model completely (required by BaseEmotionModel)"""
        self.reset_for_new_negotiation()
        self.pattern_success_rates.clear()
        self.transition_success_rates.clear()
    
    def update_model(self, success: bool, final_days: int = None, target_days: int = None):
        """Simple model update - just track success"""
        if self.current_emotional_arc:
            last_emotion = self.current_emotional_arc[-1]
            print(f"  Last emotion: {last_emotion}, Success: {success}")


def run_coherence_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    coherence_weight: float = 0.8,
    out_dir: str = "results/coherence"
) -> Dict[str, Any]:
    """Run Coherence-based emotion optimization experiment"""
    
    from llm.negotiator import DebtNegotiator
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Create Coherence model
    model = CoherenceEmotionModel(coherence_weight=coherence_weight)
    
    results = {
        'experiment_type': 'coherence_emotion_optimization',
        'model_type': 'coherence_based',
        'iterations': iterations,
        'scenarios_used': [s['id'] if 'id' in s else f"scenario_{i}" for i, s in enumerate(scenarios)],
        'config': {
            'model_creditor': model_creditor,
            'model_debtor': model_debtor,
            'debtor_emotion': debtor_emotion,
            'debtor_model_type': debtor_model_type,
            'max_dialog_len': max_dialog_len,
            'coherence_weight': coherence_weight
        },
        'detailed_results': []
    }
    
    all_negotiations = []
    
    for iteration in range(iterations):
        print(f"\n🔄 Coherence Iteration {iteration + 1}/{iterations}")
        
        for i, scenario in enumerate(scenarios):
            print(f"  📋 Scenario {i+1}/{len(scenarios)}: ", end="")
            
            # Create negotiator
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
            
            if result.get('final_state') == 'accept':
                print(f"✅ Success")
            else:
                print(f"❌ Failed - {result.get('final_state', 'unknown')}")
            
            # Add simple coherence metrics to result
            result['coherence_metrics'] = model.get_coherence_metrics()
            
            all_negotiations.append(result)
            
            # Update model
            success = result.get('final_state') == 'accept'
            model.update_model(success)
            
            # Reset for new negotiation
            model.reset_for_new_negotiation()
    
    # Calculate final performance metrics (same as vanilla)
    successful_negotiations = [n for n in all_negotiations if n.get('final_state') == 'accept']
    failed_negotiations = [n for n in all_negotiations if n.get('final_state') != 'accept']
    success_rate = len(successful_negotiations) / len(all_negotiations) if all_negotiations else 0
    
    # Calculate collection rates for successful negotiations (same as vanilla)
    collection_rates = []
    for r in successful_negotiations:
        target_days = r.get('creditor_target_days', 30)
        actual_days = r.get('collection_days', target_days)
        if target_days > 0:
            collection_rate = min(1.0, target_days / actual_days)  # Higher is better
            collection_rates.append(collection_rate)
    
    avg_collection_rate = np.mean(collection_rates) if collection_rates else 0.0
    avg_rounds = np.mean([len(r.get('dialog', [])) for r in all_negotiations])
    
    # Final statistics (same as vanilla)
    results['final_stats'] = model.get_stats()
    results['performance'] = {
        'success_rate': success_rate,
        'avg_collection_rate': float(avg_collection_rate),
        'avg_negotiation_rounds': float(avg_rounds),
        'total_negotiations': len(all_negotiations),
        'successful_negotiations': len(successful_negotiations),
        'failed_negotiations': len(failed_negotiations)
    }
    
    # Detailed breakdown (same as vanilla)
    failure_reasons = {}
    for result in failed_negotiations:
        reason = result.get('final_state', 'unknown')
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    results['analysis'] = {
        'failure_breakdown': failure_reasons,
        'success_patterns': {
            'avg_rounds_successful': float(np.mean([len(r.get('dialog', [])) for r in successful_negotiations])) if successful_negotiations else 0,
            'avg_rounds_failed': float(np.mean([len(r.get('dialog', [])) for r in failed_negotiations])) if failed_negotiations else 0
        }
    }
    
    # ===== ADD STATISTICAL ANALYSIS WITH 95% CONFIDENCE INTERVALS (same as vanilla) =====
    results = enhance_results_with_statistics(
        results, 
        all_negotiations, 
        scenarios, 
        method="bootstrap"
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/coherence_results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # ===== PRINT RESULTS WITH CONFIDENCE INTERVALS (same as vanilla) =====
    print("\n" + "="*80)
    print("📊 COHERENCE RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("="*80)
    
    # Print statistical analysis with CIs (same as vanilla)
    if 'statistical_analysis' in results:
        print(format_ci_results(results['statistical_analysis']))
        print("="*80)
    
    print(f"\n📊 COHERENCE RESULTS")
    print("=" * 50)
    print(f"✅ Success Rate: {success_rate:.1%}")
    print(f"📈 Avg Collection Rate: {avg_collection_rate:.3f}")
    print(f"💬 Avg Negotiation Rounds: {avg_rounds:.1f}")
    print(f"📁 Results saved to: {result_file}")
    
    return results