#!/usr/bin/env python3
"""
Test disaster CSV processing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import preprocess_disaster_rescue_scenarios

def test_disaster_processing():
    """Test disaster CSV processing"""
    
    csv_path = "data/disaster_survivor_scenarios.csv"
    
    print("🚨 Testing Disaster CSV Processing")
    print("=" * 50)
    
    try:
        scenarios = preprocess_disaster_rescue_scenarios(
            csv_path=csv_path,
            output_path="test_disaster_scenarios.json", 
            n_scenarios=3
        )
        
        print(f"\n✅ Successfully processed {len(scenarios)} scenarios!")
        
        if scenarios:
            print("\n🔍 First scenario sample:")
            first_scenario = scenarios[0]
            print(f"  ID: {first_scenario['id']}")
            print(f"  Type: {first_scenario['metadata']['disaster_type']}")
            print(f"  Condition: {first_scenario['metadata']['survivor_condition']}")
            print(f"  Endurance: {first_scenario['metadata']['estimated_endurance_minutes']} min")
            print(f"  ETA: {first_scenario['metadata']['rescue_team_eta_minutes']} min")
            print(f"  Critical Needs: {first_scenario['metadata']['critical_needs']}")
            print(f"  Negotiation: {first_scenario['seller']['target_price']} vs {first_scenario['buyer']['target_price']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_disaster_processing()
    if success:
        print("\n🎉 Disaster processing test PASSED!")
    else:
        print("\n💥 Disaster processing test FAILED!")