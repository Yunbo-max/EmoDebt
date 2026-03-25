"""
Model 2: HMM + Game Theory Model
"""

import numpy as np
from typing import Dict, List, Any
from models.base_model import BaseEmotionModel
import json
from datetime import datetime
import os

# Emotion definitions
EMOTIONS = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
N_EMOTIONS = len(EMOTIONS)

# Payoff matrix (creditor payoff, debtor payoff)
PAYOFF_MATRIX = {
    'happy': {'happy': (4,4), 'surprising': (3,3), 'angry': (1,2), 'sad': (2,3), 
              'disgust': (2,2), 'fear': (2,1), 'neutral': (3,3)},
    'surprising': {'happy': (3,3), 'surprising': (4,4), 'angry': (2,1), 'sad': (2,2),
                   'disgust': (1,2), 'fear': (1,1), 'neutral': (3,3)},
    'angry': {'happy': (2,1), 'surprising': (1,2), 'angry': (1,1), 'sad': (1,2),
              'disgust': (0,1), 'fear': (0,0), 'neutral': (1,2)},
    'sad': {'happy': (3,2), 'surprising': (2,2), 'angry': (2,1), 'sad': (3,3),
            'disgust': (1,1), 'fear': (1,2), 'neutral': (2,3)},
    'disgust': {'happy': (2,2), 'surprising': (1,1), 'angry': (1,0), 'sad': (1,1),
                'disgust': (2,2), 'fear': (0,1), 'neutral': (2,2)},
    'fear': {'happy': (1,2), 'surprising': (1,1), 'angry': (0,1), 'sad': (2,1),
             'disgust': (1,0), 'fear': (2,2), 'neutral': (2,3)},
    'neutral': {'happy': (3,3), 'surprising': (3,3), 'angry': (2,1), 'sad': (3,2),
                'disgust': (2,2), 'fear': (3,2), 'neutral': (3,3)}
}

class HMMGameTheoryModel(BaseEmotionModel):
    """
    HMM + Game Theory Model for emotion optimization
    Uses Hidden Markov Model for state transitions and Game Theory for optimal responses
    """
    
    def __init__(self):
        # HMM components
        self.transition_counts = np.ones((N_EMOTIONS, N_EMOTIONS))  # Add-1 smoothing
        self.emission_counts = np.ones((N_EMOTIONS, N_EMOTIONS))  # Debtor emotion → Creditor emotion
        
        # Update matrices
        self.transition_matrix = self.transition_counts / self.transition_counts.sum(axis=1, keepdims=True)
        self.emission_matrix = self.emission_counts / self.emission_counts.sum(axis=1, keepdims=True)
        
        # History
        self.emotion_history = []
        self.debtor_emotion_history = []
        self.current_emotion = 'neutral'
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
    
    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select next emotion using HMM + Game Theory"""
        debtor_emotion = state.get('debtor_emotion', 'neutral')
        round_num = state.get('round', 1)
        
        # Track debtor emotion
        if debtor_emotion in EMOTIONS:
            self.debtor_emotion_history.append(debtor_emotion)
        
        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Explore: random emotion
            next_emotion = np.random.choice(EMOTIONS)
            strategy = "exploration"
        else:
            # Exploit: use HMM + Game Theory
            if len(self.emotion_history) >= 3 and len(self.debtor_emotion_history) >= 3:
                # Use HMM prediction
                next_emotion = self._hmm_prediction(debtor_emotion)
                strategy = "hmm"
            else:
                # Use Game Theory (payoff matrix)
                next_emotion = self._game_theory_response(debtor_emotion)
                strategy = "game_theory"
        
        # Update HMM
        if self.emotion_history:
            prev_emotion = self.emotion_history[-1]
            self._update_hmm(prev_emotion, next_emotion, debtor_emotion)
        
        # Update history
        self.current_emotion = next_emotion
        self.emotion_history.append(next_emotion)
        
        # Get emotion prompt
        emotion_prompts = {
            "happy": "Use an optimistic and positive tone",
            "surprising": "Use an engaging and unexpected approach",
            "angry": "Use a firm and assertive tone",
            "sad": "Use an empathetic and understanding tone",
            "disgust": "Use a disappointed tone",
            "fear": "Use a cautious and concerned tone",
            "neutral": "Use a balanced and professional tone"
        }
        
        # Temperature based on strategy
        if strategy == "exploration":
            temperature = 0.9  # Higher for exploration
        else:
            temperature = max(0.3, 0.7 - (round_num * 0.05))  # Decay over rounds
        
        return {
            "emotion": next_emotion,
            "emotion_text": emotion_prompts.get(next_emotion, "Use a professional tone"),
            "temperature": temperature,
            "strategy": strategy,
            "debtor_emotion": debtor_emotion
        }
    
    def _hmm_prediction(self, debtor_emotion: str) -> str:
        """Predict next emotion using HMM"""
        if not self.emotion_history:
            return self._game_theory_response(debtor_emotion)
        
        current_idx = EMOTIONS.index(self.current_emotion)
        debtor_idx = EMOTIONS.index(debtor_emotion) if debtor_emotion in EMOTIONS else EMOTIONS.index('neutral')
        
        # Calculate state probabilities
        state_probs = (self.transition_matrix[current_idx] * 
                      self.emission_matrix[:, debtor_idx])
        
        # Normalize
        state_probs = state_probs / state_probs.sum()
        
        # Choose emotion with highest probability
        next_idx = np.argmax(state_probs)
        
        return EMOTIONS[next_idx]
    
    def _game_theory_response(self, debtor_emotion: str) -> str:
        """Get optimal response using game theory payoff matrix"""
        if debtor_emotion not in PAYOFF_MATRIX:
            debtor_emotion = 'neutral'
        
        payoffs = PAYOFF_MATRIX[debtor_emotion]
        
        # Find emotion with highest creditor payoff
        best_payoff = -float('inf')
        best_emotions = []
        
        for emotion, payoff in payoffs.items():
            creditor_payoff = payoff[0]
            if creditor_payoff > best_payoff:
                best_payoff = creditor_payoff
                best_emotions = [emotion]
            elif creditor_payoff == best_payoff:
                best_emotions.append(emotion)
        
        # Choose randomly among best emotions
        return np.random.choice(best_emotions)
    
    def _update_hmm(self, prev_emotion: str, current_emotion: str, observation: str):
        """Update HMM with new observation"""
        prev_idx = EMOTIONS.index(prev_emotion)
        current_idx = EMOTIONS.index(current_emotion)
        obs_idx = EMOTIONS.index(observation) if observation in EMOTIONS else EMOTIONS.index('neutral')
        
        # Update counts
        self.transition_counts[prev_idx, current_idx] += 1
        self.emission_counts[current_idx, obs_idx] += 1
        
        # Update matrices
        self.transition_matrix = self.transition_counts / self.transition_counts.sum(axis=1, keepdims=True)
        self.emission_matrix = self.emission_counts / self.emission_counts.sum(axis=1, keepdims=True)
    
    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        """Update model based on negotiation outcome"""
        success = negotiation_result.get('final_state') == 'accept'
        
        # Adjust exploration rate based on success
        if success:
            self.exploration_rate = max(0.05, self.exploration_rate * 0.9)  # Reduce exploration
        else:
            self.exploration_rate = min(0.5, self.exploration_rate * 1.1)  # Increase exploration
        
        # Store emotion sequence
        if 'emotion_sequence' in negotiation_result:
            # Learn from successful sequences
            if success and len(negotiation_result['emotion_sequence']) >= 2:
                sequence = negotiation_result['emotion_sequence']
                for i in range(len(sequence) - 1):
                    if i < len(self.debtor_emotion_history):
                        debtor_emotion = self.debtor_emotion_history[i]
                        self._update_hmm(sequence[i], sequence[i+1], debtor_emotion)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        # Calculate matrix entropy (measure of learning)
        transition_entropy = -np.sum(self.transition_matrix * np.log(self.transition_matrix + 1e-10))
        emission_entropy = -np.sum(self.emission_matrix * np.log(self.emission_matrix + 1e-10))
        
        return {
            'transition_entropy': float(transition_entropy),
            'emission_entropy': float(emission_entropy),
            'exploration_rate': self.exploration_rate,
            'history_length': len(self.emotion_history),
            'current_emotion': self.current_emotion,
            'transition_matrix': self.transition_matrix.tolist(),
            'emission_matrix': self.emission_matrix.tolist()
        }
    
    def reset(self) -> None:
        """Reset model state (keep learned parameters)"""
        self.emotion_history = []
        self.debtor_emotion_history = []
        self.current_emotion = 'neutral'

def run_hmm_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    max_dialog_len: int = 30,
    out_dir: str = "results"
) -> Dict[str, Any]:
    """Run HMM + Game Theory experiment"""
    
    from llm.negotiator import DebtNegotiator
    
    # Create model
    model = HMMGameTheoryModel()
    
    results = {
        'experiment_type': 'hmm_game_theory',
        'iterations': iterations,
        'scenarios_used': [s['id'] for s in scenarios],
        'iteration_results': {}
    }
    
    all_negotiation_results = []
    
    for iteration in range(iterations):
        print(f"\n🎯 Iteration {iteration + 1}/{iterations}")
        
        iteration_results = {
            'iteration': iteration + 1,
            'scenario_results': [],
            'model_stats': model.get_stats()
        }
        
        for scenario in scenarios:
            # Create negotiator with HMM model
            negotiator = DebtNegotiator(
                config=scenario,
                emotion_model=model,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                debtor_emotion=debtor_emotion
            )
            
            # Run negotiation
            result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
            iteration_results['scenario_results'].append(result)
            all_negotiation_results.append(result)
            
            # Update model with result
            model.update_model(result)
        
        # Store iteration results
        results['iteration_results'][f'iteration_{iteration+1}'] = iteration_results
        
        print(f"  Model stats: exploration_rate={model.exploration_rate:.3f}")
    
    # Final results
    results['final_stats'] = model.get_stats()
    
    # Calculate success metrics
    successful = [r for r in all_negotiation_results if r.get('final_state') == 'accept']
    if successful:
        avg_days = np.mean([r.get('collection_days', 0) for r in successful])
        success_rate = len(successful) / len(all_negotiation_results)
    else:
        avg_days = 0
        success_rate = 0
    
    results['performance'] = {
        'success_rate': success_rate,
        'avg_collection_days': float(avg_days),
        'total_negotiations': len(all_negotiation_results),
        'successful_negotiations': len(successful)
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/hmm_results_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    print(f"💾 Results saved to: {result_file}")
    print(f"📊 Final Performance: Success Rate = {success_rate:.1%}, Avg Days = {avg_days:.1f}")
    
    return results