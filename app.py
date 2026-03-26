"""
EmoDebt: Bayesian-Optimized Emotional Intelligence for Debt Recovery
Hugging Face Space Demo — Interactive visualization of the EmoDebt framework

Paper: https://arxiv.org/abs/2503.21080
Accepted at AAMAS 2026
"""

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import dirichlet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.stats import norm
import json
import io

# ============================================================================
# EMOTION DEFINITIONS (from paper Section 3.2)
# ============================================================================

EMOTIONS = ['Happy', 'Surprise', 'Angry', 'Sad', 'Disgust', 'Fear', 'Neutral']
N_EMOTIONS = len(EMOTIONS)

# Paper Table 2: Psychological priors from Thornton & Tamir (2017)
PSYCHOLOGICAL_PRIORS = np.array([
    [0.30, 0.15, 0.05, 0.10, 0.05, 0.05, 0.30],  # Happy
    [0.20, 0.20, 0.15, 0.10, 0.10, 0.10, 0.15],  # Surprise
    [0.10, 0.10, 0.25, 0.15, 0.15, 0.10, 0.15],  # Angry
    [0.15, 0.10, 0.10, 0.20, 0.10, 0.15, 0.20],  # Sad
    [0.10, 0.15, 0.20, 0.15, 0.15, 0.10, 0.15],  # Disgust
    [0.15, 0.10, 0.10, 0.20, 0.10, 0.15, 0.20],  # Fear
    [0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.20],  # Neutral
])

EMOTION_COLORS = {
    'Happy': '#FFD700', 'Surprise': '#FF69B4', 'Angry': '#FF4444',
    'Sad': '#4169E1', 'Disgust': '#9370DB', 'Fear': '#20B2AA',
    'Neutral': '#808080'
}


def calculate_entropy(matrix):
    """Eq. 10: H(P) = -1/7 * sum P_ij log P_ij"""
    P = matrix + 1e-10
    entropy = -np.sum(P * np.log(P)) / N_EMOTIONS
    return entropy


def generate_candidates(current_matrix, n_candidates=20, alpha=10.0, epsilon=0.1):
    """Eq. 7: Dirichlet perturbation"""
    candidates = []
    for _ in range(n_candidates):
        candidate = np.zeros_like(current_matrix)
        for i in range(N_EMOTIONS):
            alpha_params = alpha * current_matrix[i] + epsilon
            candidate[i] = dirichlet.rvs(alpha_params)[0]
        candidates.append(candidate)
    return candidates


def calculate_reward(success, n_rounds, collection_days, alpha_reward=100.0, d_max=180):
    """Eq. 3: Reward function"""
    if not success:
        return -d_max
    d_extended = max(1, collection_days)
    n = max(1, n_rounds)
    return -alpha_reward * np.log(n) / d_extended


def expected_improvement(gp, X_candidates, observations_y, xi=0.01):
    """Eq. 6: EI acquisition function"""
    if len(observations_y) < 2:
        return np.ones(len(X_candidates))
    g_best = max(observations_y)
    mu, sigma = gp.predict(X_candidates, return_std=True)
    improvement = mu - g_best - xi
    Z = np.divide(improvement, sigma, where=sigma > 0, out=np.zeros_like(improvement))
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return np.maximum(np.nan_to_num(ei, 0.0), 0.0)


def simulate_negotiation(matrix, scenario_difficulty=0.5):
    """Simulate a debt negotiation outcome based on emotional strategy"""
    emotion_idx = EMOTIONS.index('Neutral')
    n_rounds = 0
    max_rounds = 30
    creditor_pos = 30.0
    debtor_pos = 120.0

    for _ in range(max_rounds):
        n_rounds += 1
        probs = matrix[emotion_idx]
        emotion_idx = np.random.choice(N_EMOTIONS, p=probs)
        emotion = EMOTIONS[emotion_idx]

        # Emotion-dependent concession
        if emotion == 'Happy':
            creditor_move = np.random.uniform(2, 6)
            debtor_move = np.random.uniform(3, 8)
        elif emotion == 'Sad':
            creditor_move = np.random.uniform(3, 7)
            debtor_move = np.random.uniform(4, 10)
        elif emotion == 'Angry':
            creditor_move = np.random.uniform(0, 2)
            debtor_move = np.random.uniform(0, 3)
        elif emotion == 'Fear':
            creditor_move = np.random.uniform(1, 4)
            debtor_move = np.random.uniform(5, 12)
        elif emotion == 'Surprise':
            creditor_move = np.random.uniform(2, 5)
            debtor_move = np.random.uniform(2, 7)
        elif emotion == 'Disgust':
            creditor_move = np.random.uniform(0, 2)
            debtor_move = np.random.uniform(0, 2)
        else:
            creditor_move = np.random.uniform(1, 4)
            debtor_move = np.random.uniform(2, 6)

        difficulty_factor = 1.0 + scenario_difficulty * 0.5
        creditor_pos += creditor_move * difficulty_factor
        debtor_pos -= debtor_move

        if abs(creditor_pos - debtor_pos) <= 5:
            final_days = (creditor_pos + debtor_pos) / 2
            return True, n_rounds, final_days

    return False, n_rounds, 180


def plot_transition_matrix(matrix, title="Emotional Transition Matrix"):
    """Plot heatmap of transition matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                ax=ax, vmin=0, vmax=0.5, cbar_kws={'label': 'Probability'})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('To Emotion', fontsize=11)
    ax.set_ylabel('From Emotion', fontsize=11)
    plt.tight_layout()
    return fig


def plot_optimization_progress(rewards, entropies, success_rates):
    """Plot learning curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(rewards, 'b-o', markersize=4, linewidth=1.5)
    axes[0].set_title('Reward per Iteration', fontweight='bold')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(entropies, 'r-s', markersize=4, linewidth=1.5)
    axes[1].set_title('Matrix Entropy', fontweight='bold')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Entropy')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot([sr * 100 for sr in success_rates], 'g-^', markersize=4, linewidth=1.5)
    axes[2].set_title('Success Rate (%)', fontweight='bold')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Success Rate')
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_emotion_sequence(sequence):
    """Plot emotion trajectory"""
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = [list(EMOTION_COLORS.values())[EMOTIONS.index(e)] for e in sequence]
    y_vals = [EMOTIONS.index(e) for e in sequence]
    ax.scatter(range(len(sequence)), y_vals, c=colors, s=80, zorder=5)
    ax.plot(range(len(sequence)), y_vals, 'k-', alpha=0.3, linewidth=1)
    ax.set_yticks(range(N_EMOTIONS))
    ax.set_yticklabels(EMOTIONS)
    ax.set_xlabel('Negotiation Round')
    ax.set_title('Creditor Emotion Trajectory', fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def run_emodebt_demo(n_iterations, n_negotiations, scenario_difficulty, dirichlet_alpha, exploration_xi):
    """Main demo: run Bayesian optimization for emotional transitions"""
    np.random.seed(42)

    current_matrix = PSYCHOLOGICAL_PRIORS.copy()

    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5)

    observations_X = []
    observations_y = []
    best_matrix = current_matrix.copy()
    best_reward = -np.inf

    all_rewards = []
    all_entropies = []
    all_success_rates = []
    last_emotion_sequence = []

    log_text = "🧠 EmoDebt Bayesian Optimization\n" + "=" * 50 + "\n\n"

    for iteration in range(int(n_iterations)):
        iteration_rewards = []
        iteration_successes = 0

        for neg in range(int(n_negotiations)):
            success, n_rounds, collection_days = simulate_negotiation(
                current_matrix, scenario_difficulty
            )
            reward = calculate_reward(success, n_rounds, collection_days)
            iteration_rewards.append(reward)
            if success:
                iteration_successes += 1

            # Track emotion sequence for last negotiation
            if iteration == int(n_iterations) - 1 and neg == 0:
                emotion_idx = EMOTIONS.index('Neutral')
                last_emotion_sequence = ['Neutral']
                for _ in range(n_rounds - 1):
                    probs = current_matrix[emotion_idx]
                    emotion_idx = np.random.choice(N_EMOTIONS, p=probs)
                    last_emotion_sequence.append(EMOTIONS[emotion_idx])

        avg_reward = np.mean(iteration_rewards)
        success_rate = iteration_successes / int(n_negotiations)

        # Update GP
        flattened = current_matrix.flatten()
        observations_X.append(flattened)
        observations_y.append(avg_reward)

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_matrix = current_matrix.copy()

        # Train GP and optimize
        if len(observations_y) >= 3:
            X = np.array(observations_X)
            y = np.array(observations_y)
            try:
                gp.fit(X, y)
            except Exception:
                pass

        candidates = generate_candidates(current_matrix, n_candidates=20,
                                         alpha=dirichlet_alpha, epsilon=0.1)
        X_candidates = np.array([c.flatten() for c in candidates])
        ei_values = expected_improvement(gp, X_candidates, observations_y, xi=exploration_xi)
        best_idx = np.argmax(ei_values)
        current_matrix = candidates[best_idx]

        entropy = calculate_entropy(current_matrix)
        all_rewards.append(avg_reward)
        all_entropies.append(entropy)
        all_success_rates.append(success_rate)

        log_text += f"Iteration {iteration + 1}: Reward={avg_reward:.3f} | "
        log_text += f"SR={success_rate:.0%} | Entropy={entropy:.3f}"
        if avg_reward == best_reward:
            log_text += " ★ Best"
        log_text += "\n"

    log_text += f"\n{'=' * 50}\n"
    log_text += f"Best reward: {best_reward:.3f}\n"
    log_text += f"Final entropy: {all_entropies[-1]:.3f}\n"
    log_text += f"Final success rate: {all_success_rates[-1]:.0%}\n"

    # Generate plots
    prior_fig = plot_transition_matrix(PSYCHOLOGICAL_PRIORS, "Initial Psychological Priors (P⁰)")
    learned_fig = plot_transition_matrix(best_matrix, "Learned Optimal Matrix (P*)")
    progress_fig = plot_optimization_progress(all_rewards, all_entropies, all_success_rates)

    if last_emotion_sequence:
        emotion_fig = plot_emotion_sequence(last_emotion_sequence)
    else:
        emotion_fig = plt.figure(figsize=(10, 3))

    return prior_fig, learned_fig, progress_fig, emotion_fig, log_text


def show_paper_info():
    """Return paper information"""
    return """
## EmoDebt: Bayesian-Optimized Emotional Intelligence for Strategic Agent-to-Agent Debt Recovery

**Accepted at AAMAS 2026** — 25th International Conference on Autonomous Agents and Multiagent Systems

**Authors:** Yunbo Long, Yuhan Liu, Liming Xu, Alexandra Brintrup

**Affiliations:** University of Cambridge, University of Toronto, The Alan Turing Institute

### Abstract
EmoDebt introduces a Bayesian-optimized emotional intelligence engine that reframes emotional
intelligence as a sequential decision-making problem. Through a 7×7 transition probability matrix
across seven emotional states, initialized with psychologically informed priors, and optimized via
Gaussian Process-based Bayesian optimization, EmoDebt discovers optimal emotional counter-strategies
for agent-to-agent debt recovery negotiations.

### Links
- 📄 [arXiv Paper](https://arxiv.org/abs/2503.21080)
- 💻 [GitHub Code](https://github.com/Yunbo-max/EmoDebt)
- 🏆 [AAMAS 2026 Accepted Papers](https://cyprusconferences.org/aamas2026/accepted-research-track/)
"""


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(
    title="EmoDebt: Bayesian Emotional Intelligence for Debt Recovery",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🧠 EmoDebt: Bayesian-Optimized Emotional Intelligence
    ### Strategic Agent-to-Agent Debt Recovery | [Paper](https://arxiv.org/abs/2503.21080) | [GitHub](https://github.com/Yunbo-max/EmoDebt) | Accepted at AAMAS 2026
    """)

    with gr.Tabs():
        with gr.Tab("🔬 Interactive Demo"):
            gr.Markdown("### Run Bayesian Optimization for Emotional Transition Strategies")
            gr.Markdown("Simulate how EmoDebt learns optimal creditor emotional transitions against debtor strategies.")

            with gr.Row():
                with gr.Column(scale=1):
                    n_iterations = gr.Slider(3, 30, value=10, step=1, label="Optimization Iterations")
                    n_negotiations = gr.Slider(1, 10, value=5, step=1, label="Negotiations per Iteration")
                    scenario_difficulty = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Scenario Difficulty")
                    dirichlet_alpha = gr.Slider(1.0, 30.0, value=10.0, step=1.0, label="Dirichlet α (concentration)")
                    exploration_xi = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Exploration ξ (EI parameter)")
                    run_btn = gr.Button("🚀 Run EmoDebt Optimization", variant="primary", size="lg")

                with gr.Column(scale=2):
                    log_output = gr.Textbox(label="Optimization Log", lines=15, max_lines=25)

            with gr.Row():
                prior_plot = gr.Plot(label="Initial Priors (P⁰)")
                learned_plot = gr.Plot(label="Learned Matrix (P*)")

            with gr.Row():
                progress_plot = gr.Plot(label="Learning Progress")

            with gr.Row():
                emotion_plot = gr.Plot(label="Sample Emotion Trajectory")

            run_btn.click(
                fn=run_emodebt_demo,
                inputs=[n_iterations, n_negotiations, scenario_difficulty, dirichlet_alpha, exploration_xi],
                outputs=[prior_plot, learned_plot, progress_plot, emotion_plot, log_output]
            )

        with gr.Tab("📄 About the Paper"):
            gr.Markdown(show_paper_info())

            gr.Markdown("### Psychological Priors (Table 2 from Paper)")
            prior_data = []
            for i, row_emotion in enumerate(EMOTIONS):
                row = [row_emotion] + [f"{PSYCHOLOGICAL_PRIORS[i, j]:.2f}" for j in range(N_EMOTIONS)]
                prior_data.append(row)
            gr.Dataframe(
                headers=["From \\ To"] + EMOTIONS,
                value=prior_data,
                interactive=False
            )

        with gr.Tab("📖 How It Works"):
            gr.Markdown("""
            ### EmoDebt Algorithm Overview

            **1. Emotional State Modeling (Eq. 1)**
            - 7 emotional states: Happy, Surprise, Angry, Sad, Disgust, Fear, Neutral
            - Transitions governed by a 7×7 stochastic matrix P where P_ij = P(e_{t+1}=j | e_t=i)

            **2. Bayesian Optimization (Eq. 2-5)**
            - Treats negotiation outcome as black-box function g: ℝ⁴⁹ → ℝ
            - Gaussian Process with Matérn 3/2 kernel models the reward surface
            - Expected Improvement (EI) acquisition function balances exploration/exploitation

            **3. Reward Function (Eq. 3)**
            - Success: r(P) = -α · log(n_rounds) / d_extended
            - Failure: r(P) = -d_max

            **4. Online Learning (Eq. 7-8)**
            - Dirichlet perturbations generate candidate matrices
            - GP-based EI selects the most promising candidate
            - Early stopping after K=5 iterations without improvement

            **5. Key Innovation**
            - Reframes emotional intelligence as sequential decision-making
            - Psychologically-grounded initialization ensures plausible starting strategies
            - Online learning discovers counter-strategies against specific debtor tactics
            """)

    gr.Markdown("---")
    gr.Markdown("*EmoDebt — Yunbo Long, Yuhan Liu, Liming Xu, Alexandra Brintrup | University of Cambridge & University of Toronto*")

if __name__ == "__main__":
    demo.launch()
