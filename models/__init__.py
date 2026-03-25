# models/__init__.py

from .base_model import BaseEmotionModel
from .baseline_evolutionary import BaselineEvolutionaryOptimizer, run_baseline_experiment
from .qlearning_baseline import QLearningBaseline, run_qlearning_experiment
from .dqn_baseline import DQNBaseline, run_dqn_experiment
from .hierarchical_evolutionary import HierarchicalBayesianOptimizer, run_hierarchical_experiment
from .hmm_game_theory import HMMGameTheoryModel, run_hmm_experiment
# Replace the old import with the new one:
from .bayesian_multiagent import BayesianTransitionModel, run_bayesian_transition_experiment
# Remove or comment out the old one:
# from .bayesian_multiagent import BayesianMultiAgentModel, run_bayesian_multiagent_experiment

__all__ = [
    'BaseEmotionModel',
    'BaselineEvolutionaryOptimizer',
    'run_baseline_experiment',
    'QLearningBaseline',
    'run_qlearning_experiment',
    'DQNBaseline',
    'run_dqn_experiment',
    'HierarchicalBayesianOptimizer', 
    'run_hierarchical_experiment',
    'HMMGameTheoryModel',
    'run_hmm_experiment',
    # Updated to the new model:
    'BayesianTransitionModel',
    'run_bayesian_transition_experiment',
    # Remove the old one:
    # 'BayesianMultiAgentModel',
    # 'run_bayesian_multiagent_experiment'
]