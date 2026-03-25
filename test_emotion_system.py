#!/usr/bin/env python3
"""
Quick test for the enhanced emotion system with DeepSeek offline model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph_bargain_debt import AdaptiveDebtBargain

def test_emotion_system():
    """Quick test of the emotion detection and interaction system"""
    
    # Sample debt collection scenario
    config = {
        "id": "test_emotion_001", 
        "product": {"type": "debt_collection", "amount": 15000},
        "seller": {"target_price": 30},
        "buyer": {"target_price": 90},
        "metadata": {
            "outstanding_balance": 15000,
            "creditor_name": "ABC Collections",
            "debtor_name": "John Doe", 
            "cash_flow_situation": "Irregular income",
            "business_impact": "High impact on credit score",
            "recovery_stage": "Early collection"
        }
    }
    
    print("🧪 Testing Advanced Emotion System with DeepSeek 7B")
    print("=" * 60)
    
    try:
        # Create negotiation system with DeepSeek 7B for creditor
        negotiation = AdaptiveDebtBargain(
            id=config['id'],
            config=config,
            model_creditor="deepseek-7b",   # Use DeepSeek for emotion detection
            model_debtor="gpt-4o-mini",     # Use GPT for debtor responses
            use_advanced_emotions=True      # Enable HMM/Payoff system
        )
        
        print(f"✅ System initialized successfully")
        print(f"🎭 Emotion Detector: {type(negotiation.emotion_detector).__name__}")
        print(f"🧠 Emotion System: {type(negotiation.emotion_system).__name__}")
        
        # Test emotion detection on sample debtor messages
        test_messages = [
            "I can't pay right now, this is very stressful!",  # Fear/Sadness
            "This is ridiculous! I don't owe that much!",      # Anger
            "Thank you for being understanding about this.",    # Joy
            "I'm disgusted by how you're treating me.",        # Disgust
            "Wow, I didn't expect this offer.",                # Surprise
            "Let me think about your payment plan."            # Neutral
        ]
        
        print("\n🔍 Testing Emotion Detection:")
        print("-" * 40)
        
        for i, message in enumerate(test_messages, 1):
            detected_emotion = negotiation.emotion_detector.detect_emotion(message)
            print(f"{i}. '{message[:40]}...'")
            print(f"   Detected: {detected_emotion}")
            
        print("\n🎭 Testing Emotion Response System:")
        print("-" * 40)
        
        # Test emotion interaction system
        test_sequence = ["Anger", "Fear", "Sadness", "Anger", "Surprise"]
        current_creditor_emotion = None
        
        for debtor_emotion in test_sequence:
            creditor_emotion, strategy = negotiation.emotion_system.get_next_emotion(
                debtor_emotion, current_creditor_emotion
            )
            current_creditor_emotion = creditor_emotion
            
            print(f"Debtor: {debtor_emotion:8} -> Creditor: {creditor_emotion:8} | Strategy: {strategy}")
        
        print(f"\n📊 Emotion System Stats:")
        print(f"   Customer History: {list(negotiation.emotion_system.customer_history)}")
        print(f"   Robot History: {list(negotiation.emotion_system.robot_history)}")
        print(f"   Should Use HMM: {negotiation.emotion_system._should_use_hmm()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing emotion system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_emotion_system()
    if success:
        print("\n✅ Emotion system test completed successfully!")
        print("\n📖 Ready to run full negotiations with:")
        print("   python langgraph_bargain_debt.py --mode test --scenarios 1 --model_creditor deepseek-7b")
    else:
        print("\n❌ Emotion system test failed!")