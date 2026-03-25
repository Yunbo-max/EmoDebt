#!/usr/bin/env python3
"""
Quick test to see if the coherence model exploration is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.bayesian_multiagent import EmotionalCoherenceAgent, TransitionContext

def test_coherence_exploration():
    """Test that the coherence agent explores different emotions"""
    
    agent = EmotionalCoherenceAgent()
    
    print("🎭 Testing Coherence Agent Exploration")
    print("="*50)
    
    # Simulate a sequence of emotional decisions
    emotions_selected = []
    
    for round_num in range(1, 11):  # 10 rounds
        
        # Create test context
        context = TransitionContext(
            current_emotion='N' if not emotions_selected else emotions_selected[-1],
            debtor_emotion='N',  # Neutral debtor
            negotiation_phase='middle',
            round_number=round_num,
            emotional_history=emotions_selected.copy(),
            debt_amount=15000,
            recent_success_rate=0.5,
            gap_size=20.0
        )
        
        # Get prediction
        prediction = agent.predict(context)
        selected_emotion = prediction.target_emotion
        emotions_selected.append(selected_emotion)
        
        print(f"Round {round_num:2d}: {selected_emotion} (conf: {prediction.confidence:.2f}) - {prediction.reasoning}")
    
    print("\n📊 Exploration Analysis:")
    print(f"Emotions used: {emotions_selected}")
    print(f"Unique emotions: {len(set(emotions_selected))}/7 possible")
    print(f"Repetition patterns:")
    
    # Check for repetitive patterns
    for emotion in ['N', 'J', 'A', 'S', 'Su', 'D', 'F']:
        count = emotions_selected.count(emotion)
        if count > 0:
            print(f"  {emotion}: {count} times ({count/len(emotions_selected)*100:.1f}%)")
    
    # Check for consecutive repetitions
    consecutive_count = 0
    for i in range(1, len(emotions_selected)):
        if emotions_selected[i] == emotions_selected[i-1]:
            consecutive_count += 1
    
    print(f"Consecutive repetitions: {consecutive_count}/{len(emotions_selected)-1} ({consecutive_count/(len(emotions_selected)-1)*100:.1f}%)")
    
    if len(set(emotions_selected)) >= 5:
        print("✅ GOOD: Agent explores multiple emotions")
    elif consecutive_count < 3:
        print("✅ GOOD: Agent avoids too much repetition")
    else:
        print("❌ ISSUE: Agent may be getting stuck in patterns")

if __name__ == "__main__":
    test_coherence_exploration()