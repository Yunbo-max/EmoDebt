"""
Bayesian Optimization for Emotional Transition Matrices (EmoDebt)
Implementation based on the mathematical formulation in the paper
"""

import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from models.base_model import BaseEmotionModel
import json
from datetime import datetime
import os
from scipy.stats import norm, dirichlet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

# ============================================================================
# EMOTION DEFINITIONS
# ============================================================================

EMOTIONS = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
N_EMOTIONS = len(EMOTIONS)

# Psychological priors for emotion transitions (from Thornton et al., 2017)
PSYCHOLOGICAL_PRIORS = np.array([
    # To: happy, surprising, angry, sad, disgust, fear, neutral
    [0.40, 0.15, 0.05, 0.10, 0.05, 0.05, 0.20],  # From happy
    [0.20, 0.30, 0.10, 0.10, 0.10, 0.10, 0.10],  # From surprising
    [0.10, 0.10, 0.35, 0.15, 0.10, 0.10, 0.10],  # From angry
    [0.15, 0.10, 0.10, 0.35, 0.10, 0.10, 0.10],  # From sad
    [0.10, 0.10, 0.15, 0.15, 0.30, 0.10, 0.10],  # From disgust
    [0.10, 0.10, 0.10, 0.15, 0.10, 0.30, 0.15],  # From fear
    [0.15, 0.15, 0.10, 0.15, 0.10, 0.10, 0.25],  # From neutral
])

class EmoDebtBO(BaseEmotionModel):
    """
    Bayesian Optimization for Emotional Transition Matrices (EmoDebt)
    
    Based on the mathematical formulation:
    1. Emotional states: 7 emotions
    2. Transition matrix P (7×7) with P_ij = P(e_{t+1}=j | e_t=i)
    3. Bayesian optimization over flattened vector p = vec(P) ∈ ℝ^49
    4. Gaussian Process with Matérn kernel for unknown reward function
    5. Expected Improvement (EI) acquisition function
    """
    
    def __init__(
        self,
        exploration_param: float = 0.01,  # ξ in EI equation
        dirichlet_alpha: float = 10.0,    # α in Dirichlet perturbation
        smoothing_epsilon: float = 0.1,   # ε for numerical stability
        n_candidates: int = 20,           # Number of candidates per iteration
        initial_samples: int = 5,         # Initial random samples
    ):
        # Parameters from equations
        self.xi = exploration_param
        self.alpha = dirichlet_alpha
        self.epsilon = smoothing_epsilon
        self.n_candidates = n_candidates
        
        # Initialize with psychological priors (Equation 3)
        self.current_matrix = PSYCHOLOGICAL_PRIORS.copy()
        self.normalize_matrix()
        
        # Gaussian Process model (Equations 5-6)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=1.0, 
            length_scale_bounds=(1e-2, 1e2),
            nu=1.5  # Matérn 3/2 kernel
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )
        
        # History of observations D_k = {(p_i, r_i)} (Equation 9)
        self.observations_X = []  # Flattened matrices
        self.observations_y = []  # Rewards
        
        # Best observed strategy
        self.best_matrix = self.current_matrix.copy()
        self.best_reward = -np.inf
        
        # Current state
        self.current_emotion_idx = EMOTIONS.index('neutral')
        self.current_emotion = 'neutral'
        self.emotion_history = []
        
        # Learning statistics
        self.iteration = 0
        self.entropy_history = []
        
        # Generate initial random samples for GP training
        self._initialize_with_random_samples(initial_samples)
    
    def _initialize_with_random_samples(self, n_samples: int):
        """Generate initial random transition matrices for GP warm-up"""
        print(f"🧪 Generating {n_samples} initial random samples for GP...")
        
        for _ in range(n_samples):
            # Create random valid transition matrix
            random_matrix = np.random.dirichlet([1.0] * N_EMOTIONS, size=N_EMOTIONS)
            
            # Store as observation with dummy reward (will be replaced)
            flattened = random_matrix.flatten()
            self.observations_X.append(flattened)
            self.observations_y.append(0.0)  # Will be updated after evaluation
        
        print(f"✅ Initialized with {len(self.observations_X)} samples")
    
    def normalize_matrix(self):
        """Ensure matrix rows sum to 1 (Equation 2)"""
        row_sums = self.current_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Prevent division by zero
        self.current_matrix = self.current_matrix / row_sums
    
    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select next emotion based on current transition matrix
        Using: e_{t+1} ∼ P(current_emotion, :)
        """
        round_num = state.get('round', 1)
        
        # Sample next emotion from transition probabilities (Equation 1)
        transition_probs = self.current_matrix[self.current_emotion_idx]
        next_idx = np.random.choice(N_EMOTIONS, p=transition_probs)
        next_emotion = EMOTIONS[next_idx]
        
        # Update state
        self.current_emotion_idx = next_idx
        self.current_emotion = next_emotion
        self.emotion_history.append(next_emotion)
        
        # Temperature schedule (not in equations but useful for LLM)
        temperature = max(0.1, 0.7 * (0.95 ** round_num))
        
        # Emotion prompts
        emotion_prompts = {
            "happy": "Use an optimistic and positive tone, expressing confidence",
            "surprising": "Use an engaging and unexpected approach, introducing creative solutions",
            "angry": "Use a firm and assertive tone, emphasizing urgency and importance",
            "sad": "Use an empathetic and understanding tone, acknowledging difficulty",
            "disgust": "Use a disappointed tone while remaining professional",
            "fear": "Use a cautious and concerned tone, highlighting potential consequences",
            "neutral": "Use a balanced and professional tone, focusing on practical solutions"
        }
        
        return {
            "emotion": next_emotion,
            "emotion_text": emotion_prompts.get(next_emotion, "Professional tone"),
            "temperature": temperature,
            "iteration": self.iteration,
            "matrix_entropy": self._calculate_entropy(),
            "transition_probs": transition_probs.tolist()
        }
    
    def calculate_reward(self, result: Dict[str, Any]) -> float:
        """
        Calculate reward based on Equation 4:
        r(P) = -α * log(n_rounds) / d_extended if successful
        r(P) = -d_max if failed
        """
        success = result.get('final_state') == 'accept'
        collection_days = result.get('collection_days', 0)
        target_days = result.get('creditor_target_days', 30)
        negotiation_rounds = result.get('negotiation_rounds', 1)
        
        # Default maximum days penalty (Equation 4)
        d_max = 180  # 6 months as maximum penalty
        
        if not success:
            return -d_max
        
        # For successful negotiations (Equation 4)
        if collection_days is None or collection_days <= 0:
            # Use target days as fallback
            d_extended = target_days
        else:
            d_extended = collection_days
        
        # Ensure positive values
        n_rounds = max(1, negotiation_rounds)
        d_extended = max(1, d_extended)
        
        # α parameter (tune based on importance of speed vs timeline)
        alpha = 100.0
        
        # Calculate reward: -α * log(n_rounds) / d_extended
        # Negative because we want to minimize (shorter is better)
        reward = -alpha * np.log(n_rounds) / d_extended
        
        return reward
    
    def _generate_candidate_matrices(self) -> List[np.ndarray]:
        """
        Generate candidate matrices via Dirichlet perturbations (Equation 10)
        P_candidate^(i) ∼ Dirichlet(α * P_current^(row) + ε)
        """
        candidates = []
        
        for _ in range(self.n_candidates):
            candidate = np.zeros_like(self.current_matrix)
            
            for i in range(N_EMOTIONS):
                # Current row probabilities
                current_row = self.current_matrix[i]
                
                # Dirichlet concentration parameters
                alpha_params = self.alpha * current_row + self.epsilon
                
                # Sample new row from Dirichlet distribution
                new_row = dirichlet.rvs(alpha_params)[0]
                candidate[i] = new_row
            
            candidates.append(candidate)
        
        return candidates
    
    def expected_improvement(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        Calculate Expected Improvement (EI) acquisition function (Equation 9)
        EI(p) = E[max(0, g(p) - g(p^+) - ξ)]
        """
        if len(self.observations_y) < 2:
            # Not enough data, return uniform exploration
            return np.ones(len(X_candidates))
        
        # Current best reward g(p^+)
        g_best = max(self.observations_y)
        
        # GP predictions for candidates
        try:
            mu, sigma = self.gp.predict(X_candidates, return_std=True)
        except:
            # GP not ready, return random exploration
            return np.random.random(len(X_candidates))
        
        # Calculate Expected Improvement
        with np.errstate(divide='warn'):
            improvement = mu - g_best - self.xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        # Handle numerical issues
        ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
        
        return np.maximum(ei, 0.0)
    
    def update_with_negotiation_result(self, result: Dict[str, Any]):
        """
        Bayesian optimization update after negotiation (Equation 11)
        P_{k+1} = argmax_{P ∈ C_k} EI(vec(P))
        """
        self.iteration += 1
        
        # Calculate reward for this negotiation
        reward = self.calculate_reward(result)
        print(f"  🔄 Iteration {self.iteration}: Reward = {reward:.3f}")
        
        # Add to observations (flatten current matrix)
        flattened = self.current_matrix.flatten()
        self.observations_X.append(flattened)
        self.observations_y.append(reward)
        
        # Update best if improved
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_matrix = self.current_matrix.copy()
            print(f"  🏆 New best reward: {reward:.3f}")
        
        # Train GP on all observations
        if len(self.observations_y) >= 3:
            X = np.array(self.observations_X)
            y = np.array(self.observations_y)
            
            try:
                self.gp.fit(X, y)
                print(f"  🧠 GP trained on {len(y)} observations")
            except Exception as e:
                print(f"  ⚠️ GP training failed: {e}")
        
        # Generate candidate matrices (Equation 10)
        candidates = self._generate_candidate_matrices()
        
        if candidates:
            # Flatten candidates for EI calculation
            X_candidates = np.array([c.flatten() for c in candidates])
            
            # Calculate EI for each candidate
            ei_values = self.expected_improvement(X_candidates)
            
            # Select best candidate (Equation 11)
            best_idx = np.argmax(ei_values)
            self.current_matrix = candidates[best_idx]
            self.normalize_matrix()
            
            print(f"  🎯 Selected candidate with EI = {ei_values[best_idx]:.4f}")
        
        # Record entropy for monitoring (Equation 13)
        entropy = self._calculate_entropy()
        self.entropy_history.append(entropy)
        print(f"  📊 Matrix entropy: {entropy:.3f}")
    
    def _calculate_entropy(self) -> float:
        """
        Calculate normalized entropy of transition matrix (Equation 13)
        H(P) = -1/7 ∑_{i=1}^7 ∑_{j=1}^7 P_ij log(P_ij)
        """
        # Add small epsilon to avoid log(0)
        P = self.current_matrix + 1e-10
        
        # Calculate entropy
        entropy = -np.sum(P * np.log(P))
        
        # Normalize by number of rows
        normalized_entropy = entropy / N_EMOTIONS
        
        return normalized_entropy
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        return {
            'iteration': self.iteration,
            'best_reward': float(self.best_reward),
            'current_entropy': float(self._calculate_entropy()),
            'n_observations': len(self.observations_y),
            'current_emotion': self.current_emotion,
            'emotion_history_length': len(self.emotion_history),
            'gp_trained': len(self.observations_y) >= 3
        }
    
    def get_learned_transitions(self) -> Dict[str, Any]:
        """Get detailed learned transition probabilities"""
        transitions = {}
        
        for i, from_emotion in enumerate(EMOTIONS):
            for j, to_emotion in enumerate(EMOTIONS):
                prob = self.current_matrix[i, j]
                if prob > 0.1:  # Only significant transitions
                    key = f"{from_emotion}→{to_emotion}"
                    transitions[key] = float(prob)
        
        return transitions
    
    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        """Interface method - calls the Bayesian update"""
        self.update_with_negotiation_result(negotiation_result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Interface method for statistics"""
        return self.get_learning_stats()
    
    def reset(self) -> None:
        """Reset for new negotiation (keep learned matrix)"""
        self.current_emotion_idx = EMOTIONS.index('neutral')
        self.current_emotion = 'neutral'
        self.emotion_history = []

# ============================================================================
# EXPERIMENT RUNNER FOR EMODEBT
# ============================================================================

def run_emodebt_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 20,
    negotiations_per_iteration: int = 3,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    max_dialog_len: int = 30,
    out_dir: str = "results/emodebt"
) -> Dict[str, Any]:
    """Run EmoDebt Bayesian Optimization experiment"""
    
    from llm.negotiator import DebtNegotiator
    
    # Create EmoDebt optimizer
    optimizer = EmoDebtBO(
        exploration_param=0.01,
        dirichlet_alpha=10.0,
        smoothing_epsilon=0.1,
        n_candidates=20,
        initial_samples=5
    )
    
    results = {
        'experiment_type': 'emodebt_bayesian_optimization',
        'iterations': iterations,
        'negotiations_per_iteration': negotiations_per_iteration,
        'parameters': {
            'exploration_param': 0.01,
            'dirichlet_alpha': 10.0,
            'smoothing_epsilon': 0.1,
            'n_candidates': 20
        },
        'iteration_results': {},
        'scenarios_used': [s['id'] for s in scenarios]
    }
    
    all_negotiation_results = []
    
    for iteration in range(iterations):
        print(f"\n{'='*80}")
        print(f"🎯 EMODEBT ITERATION {iteration + 1}/{iterations}")
        print(f"{'='*80}")
        
        iteration_negotiations = []
        
        # Run multiple negotiations per iteration
        for neg_idx in range(negotiations_per_iteration):
            scenario = scenarios[neg_idx % len(scenarios)]
            
            print(f"  🧪 Negotiation {neg_idx + 1}/{negotiations_per_iteration} - {scenario['id']}")
            
            # Create negotiator with EmoDebt optimizer
            negotiator = DebtNegotiator(
                config=scenario,
                emotion_model=optimizer,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                debtor_emotion=debtor_emotion
            )
            
            # Run negotiation
            result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
            iteration_negotiations.append(result)
            all_negotiation_results.append(result)
            
            # Show quick result
            outcome = "✅" if result.get('final_state') == 'accept' else "❌"
            days = result.get('collection_days', 'N/A')
            seq_len = len(result.get('emotion_sequence', []))
            print(f"     {outcome} Days: {days} | Seq length: {seq_len}")
        
        # ================= BAYESIAN OPTIMIZATION UPDATE =================
        print(f"\n  🔄 BAYESIAN OPTIMIZATION UPDATE...")
        
        # Update with each negotiation result
        for result in iteration_negotiations:
            optimizer.update_with_negotiation_result(result)
        
        # Calculate iteration statistics
        successful = [r for r in iteration_negotiations if r.get('final_state') == 'accept']
        success_rate = len(successful) / len(iteration_negotiations)
        
        if successful:
            avg_days = np.mean([r.get('collection_days', 0) for r in successful])
            avg_rounds = np.mean([len(r.get('dialog', [])) for r in successful])
        else:
            avg_days = avg_rounds = 0
        
        # Get optimizer stats
        stats = optimizer.get_learning_stats()
        
        print(f"  📊 Iteration Summary:")
        print(f"     Success rate: {success_rate:.1%}")
        print(f"     Avg days: {avg_days:.1f}")
        print(f"     Best reward: {optimizer.best_reward:.3f}")
        print(f"     Matrix entropy: {stats['current_entropy']:.3f}")
        print(f"     Observations: {stats['n_observations']}")
        
        # Store iteration results
        results['iteration_results'][f'iteration_{iteration+1}'] = {
            'stats': stats,
            'success_rate': success_rate,
            'avg_days': float(avg_days),
            'avg_rounds': float(avg_rounds),
            'learned_transitions': optimizer.get_learned_transitions()
        }
    
    # Final results
    results['final_stats'] = optimizer.get_learning_stats()
    results['final_matrix'] = optimizer.current_matrix.tolist()
    results['best_matrix'] = optimizer.best_matrix.tolist()
    results['best_reward'] = float(optimizer.best_reward)
    
    # Calculate overall performance
    successful_all = [r for r in all_negotiation_results if r.get('final_state') == 'accept']
    overall_success_rate = len(successful_all) / len(all_negotiation_results)
    
    if successful_all:
        overall_avg_days = np.mean([r.get('collection_days', 0) for r in successful_all])
    else:
        overall_avg_days = 0
    
    results['performance'] = {
        'overall_success_rate': overall_success_rate,
        'overall_avg_days': float(overall_avg_days),
        'total_negotiations': len(all_negotiation_results),
        'successful_negotiations': len(successful_all)
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    result_file = f"{out_dir}/emodebt_results_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    print(f"\n💾 Results saved to: {result_file}")
    
    # Print final learned structure
    print(f"\n🏆 FINAL LEARNED EMOTIONAL TRANSITIONS (EmoDebt):")
    print(f"   Best reward achieved: {optimizer.best_reward:.3f}")
    print(f"   Final matrix entropy: {optimizer._calculate_entropy():.3f}")
    
    # Show significant learned transitions
    transitions = optimizer.get_learned_transitions()
    print(f"\n   Significant learned transitions (>0.1):")
    for transition, prob in sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"     {transition}: {prob:.3f}")
    
    # Compare with initial psychological priors
    print(f"\n   Comparison with psychological priors:")
    for i, from_emotion in enumerate(EMOTIONS):
        for j, to_emotion in enumerate(EMOTIONS):
            learned = optimizer.current_matrix[i, j]
            prior = PSYCHOLOGICAL_PRIORS[i, j]
            diff = learned - prior
            if abs(diff) > 0.15:  # Significant change
                print(f"     {from_emotion}→{to_emotion}: {prior:.3f} → {learned:.3f} (Δ={diff:+.3f})")
    
    return results