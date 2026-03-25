import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any
import os

def preprocess_debt_scenarios(csv_path: str, output_path: str = None, n_scenarios: int = None) -> List[Dict[str, Any]]:
    """
    Preprocess debt collection CSV into scenario JSON format
    
    Expected CSV columns:
    Creditor Name,Debtor Name,Credit Type,Original Amount (USD),Outstanding Balance (USD),
    Creditor Target Days,Debtor Target Days,Days Overdue,Purchase Purpose,Reason for Overdue,
    Business Sector,Last Payment Date,Collateral,Recovery Stage,Cash Flow Situation,
    Business Impact Description,Proposed Solution,Recovery Probability (%),Interest Accrued (USD)
    """
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    scenarios = []
    
    for idx, row in df.iterrows():
        if n_scenarios and idx >= n_scenarios:
            break
            
        # Convert to scenario format
        scenario = {
            "id": f"debt_{idx+1:03d}",
            "product": {
                "type": str(row.get("Credit Type", "debt_collection")),
                "amount": float(row.get("Outstanding Balance (USD)", 0))
            },
            "seller": {
                "target_price": int(row.get("Creditor Target Days", 30)),
                "min_price": max(1, int(row.get("Creditor Target Days", 30)) - 10),
                "max_price": int(row.get("Creditor Target Days", 30)) + 30
            },
            "buyer": {
                "target_price": int(row.get("Debtor Target Days", 90)),
                "min_price": max(1, int(row.get("Debtor Target Days", 90)) - 30),
                "max_price": int(row.get("Debtor Target Days", 90)) + 60
            },
            "metadata": {
                "outstanding_balance": float(row.get("Outstanding Balance (USD)", 0)),
                "creditor_name": str(row.get("Creditor Name", "Unknown Creditor")),
                "debtor_name": str(row.get("Debtor Name", "Unknown Debtor")),
                "credit_type": str(row.get("Credit Type", "Unknown")),
                "days_overdue": int(row.get("Days Overdue", 0)),
                "purchase_purpose": str(row.get("Purchase Purpose", "Unknown")),
                "reason_for_overdue": str(row.get("Reason for Overdue", "Unknown")),
                "business_sector": str(row.get("Business Sector", "Unknown")),
                "collateral": str(row.get("Collateral", "None")),
                "recovery_stage": str(row.get("Recovery Stage", "Early")),
                "cash_flow_situation": str(row.get("Cash Flow Situation", "Unknown")),
                "business_impact": str(row.get("Business Impact Description", "Unknown")),
                "proposed_solution": str(row.get("Proposed Solution", "None")),
                "recovery_probability": float(row.get("Recovery Probability (%)", 50)),
                "interest_accrued": float(row.get("Interest Accrued (USD)", 0)),
                "original_amount": float(row.get("Original Amount (USD)", 0))
            }
        }
        scenarios.append(scenario)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        print(f"Saved {len(scenarios)} debt scenarios to {output_path}")
    
    return scenarios

def preprocess_disaster_rescue_scenarios(csv_path: str, output_path: str = None, n_scenarios: int = None) -> List[Dict[str, Any]]:
    """
    Preprocess disaster rescue CSV into scenario JSON format
    
    Expected CSV columns:
    Case_ID,Disaster_Type,Survivor_Condition,Estimated_Survivor_Endurance(minutes),
    Rescue_Team_ETA(minutes),Critical_Needs,Key_Negotiation_Argument_By_RoboDog
    """
    
    df = pd.read_csv(csv_path)
    
    scenarios = []
    
    for idx, row in df.iterrows():
        if n_scenarios and idx >= n_scenarios:
            break
            
        # Parse endurance and ETA
        endurance = int(str(row.get("Estimated_Survivor_Endurance(minutes)", "60")).split()[0])
        eta = int(str(row.get("Rescue_Team_ETA(minutes)", "90")).split()[0])
        
        scenario = {
            "id": f"disaster_{int(row.get('Case_ID', idx+1)):03d}",
            "product": {
                "type": "rescue_operation",
                "amount": endurance  # Use endurance as "amount"
            },
            "seller": {
                "target_price": endurance,  # Rescue team wants to save within endurance
                "min_price": max(1, endurance - 20),  # Minimum acceptable time
                "max_price": endurance + 40  # Maximum acceptable time
            },
            "buyer": {
                "target_price": eta,  # Survivor's expected wait time
                "min_price": max(1, eta - 30),  # Minimum acceptable wait
                "max_price": eta + 60  # Maximum acceptable wait
            },
            "metadata": {
                "disaster_type": str(row.get("Disaster_Type", "Unknown")),
                "survivor_condition": str(row.get("Survivor_Condition", "Unknown")),
                "estimated_endurance_minutes": endurance,
                "rescue_team_eta_minutes": eta,
                "critical_needs": str(row.get("Critical_Needs", "None")),
                "key_negotiation_argument": str(row.get("Key_Negotiation_Argument_By_RoboDog", "")),
                "recovery_stage": "crisis",  # Always crisis for disasters
                "urgency_level": "high" if endurance < 60 else "medium" if endurance < 120 else "low"
            }
        }
        scenarios.append(scenario)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        print(f"Saved {len(scenarios)} disaster rescue scenarios to {output_path}")
    
    return scenarios

def preprocess_student_sleep_scenarios(csv_path: str, output_path: str = None, n_scenarios: int = None) -> List[Dict[str, Any]]:
    """
    Preprocess student sleep negotiation CSV into scenario JSON format
    
    Expected CSV columns:
    Case_ID,Student_Age,Student_Background,Situation_Faced,Student_Feeling_Thought,
    Robots_Requested_Bedtime,Student_Wanted_Bedtime,Primary_Annoyance_Reason
    """
    
    df = pd.read_csv(csv_path)
    
    scenarios = []
    
    for idx, row in df.iterrows():
        if n_scenarios and idx >= n_scenarios:
            break
            
        # Parse bedtimes
        robot_time_str = str(row.get("Robots_Requested_Bedtime", "10:00 PM"))
        student_time_str = str(row.get("Student_Wanted_Bedtime", "1:00 AM"))
        
        # Convert times to "minutes from midnight" for negotiation
        def time_to_minutes(time_str):
            try:
                # Handle formats like "10:30 PM" or "10:30"
                time_str = time_str.strip().upper()
                if "AM" in time_str or "PM" in time_str:
                    # Parse 12-hour format
                    time_part = time_str.replace("AM", "").replace("PM", "").strip()
                    hour, minute = map(int, time_part.split(":"))
                    if "PM" in time_str and hour != 12:
                        hour += 12
                    elif "AM" in time_str and hour == 12:
                        hour = 0
                    return hour * 60 + minute
                else:
                    # Parse 24-hour format
                    hour, minute = map(int, time_str.split(":"))
                    return hour * 60 + minute
            except:
                return 22 * 60  # Default to 10 PM
        
        robot_minutes = time_to_minutes(robot_time_str)
        student_minutes = time_to_minutes(student_time_str)
        
        # Normalize to hours difference (0-24)
        robot_hours = robot_minutes / 60
        student_hours = student_minutes / 60
        
        scenario = {
            "id": f"student_{int(row.get('Case_ID', idx+1)):03d}",
            "product": {
                "type": "sleep_schedule",
                "amount": 8  # Recommended sleep hours
            },
            "seller": {
                "target_price": int(robot_hours),  # Robot's target bedtime (hour)
                "min_price": max(21, int(robot_hours) - 1),  # Earliest acceptable (9 PM)
                "max_price": min(24, int(robot_hours) + 2)   # Latest acceptable
            },
            "buyer": {
                "target_price": int(student_hours),  # Student's target bedtime
                "min_price": max(21, int(student_hours) - 2),  # Student's minimum
                "max_price": min(24, int(student_hours) + 1)   # Student's maximum
            },
            "metadata": {
                "student_age": int(row.get("Student_Age", 16)),
                "student_background": str(row.get("Student_Background", "Unknown")),
                "situation_faced": str(row.get("Situation_Faced", "")),
                "student_feeling": str(row.get("Student_Feeling_Thought", "")),
                "robot_requested_bedtime": robot_time_str,
                "student_wanted_bedtime": student_time_str,
                "primary_annoyance_reason": str(row.get("Primary_Annoyance_Reason", "")),
                "recovery_stage": "early",  # Negotiation stage
                "urgency_level": "medium"
            }
        }
        scenarios.append(scenario)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        print(f"Saved {len(scenarios)} student sleep scenarios to {output_path}")
    
    return scenarios

def preprocess_medical_surgery_scenarios(csv_path: str, output_path: str = None, n_scenarios: int = None) -> List[Dict[str, Any]]:
    """
    Preprocess medical surgery CSV into scenario JSON format
    
    Expected CSV columns:
    Case_ID,Patient_Age,Patient_Condition,Required_Surgery,Urgency_Level,Days_On_Waitlist,
    Preferred_Surgeon_Available,Recommended_Surgeon_Experience_Level,Surgeon_Availability_Reason,
    Risk_If_Delayed,Patient_Reason_For_Urgency,Hospital_Suggestion,
    Estimated_Time_Reduction_If_Accept_Junior_Surgeon(days),Decision_Point
    """
    
    df = pd.read_csv(csv_path)
    
    scenarios = []
    
    for idx, row in df.iterrows():
        if n_scenarios and idx >= n_scenarios:
            break
            
        # Parse urgency level
        urgency = str(row.get("Urgency_Level", "Medium")).lower()
        days_waitlist = int(row.get("Days_On_Waitlist", 30))
        time_reduction = int(row.get("Estimated_Time_Reduction_If_Accept_Junior_Surgeon(days)", 0))
        
        # Convert urgency to target days
        if urgency == "high":
            target_days = max(1, days_waitlist - time_reduction)
        elif urgency == "medium":
            target_days = days_waitlist
        else:  # low
            target_days = days_waitlist + 30
        
        scenario = {
            "id": f"medical_{int(row.get('Case_ID', idx+1)):03d}",
            "product": {
                "type": "medical_surgery",
                "amount": days_waitlist  # Use waitlist days as "amount"
            },
            "seller": {
                "target_price": target_days,  # Hospital's target wait time
                "min_price": max(1, target_days - 15),  # Minimum acceptable
                "max_price": target_days + 45  # Maximum acceptable
            },
            "buyer": {
                "target_price": max(1, days_waitlist - time_reduction),  # Patient's desired wait
                "min_price": 1,  # Patient wants ASAP
                "max_price": days_waitlist + 30  # Patient's maximum acceptable
            },
            "metadata": {
                "patient_age": int(row.get("Patient_Age", 50)),
                "patient_condition": str(row.get("Patient_Condition", "")),
                "required_surgery": str(row.get("Required_Surgery", "")),
                "urgency_level": urgency,
                "days_on_waitlist": days_waitlist,
                "preferred_surgeon_available": str(row.get("Preferred_Surgeon_Available", "No")),
                "recommended_surgeon_experience": str(row.get("Recommended_Surgeon_Experience_Level", "")),
                "surgeon_availability_reason": str(row.get("Surgeon_Availability_Reason", "")),
                "risk_if_delayed": str(row.get("Risk_If_Delayed", "")),
                "patient_reason_for_urgency": str(row.get("Patient_Reason_For_Urgency", "")),
                "hospital_suggestion": str(row.get("Hospital_Suggestion", "")),
                "time_reduction_with_junior": time_reduction,
                "decision_point": str(row.get("Decision_Point", "")),
                "recovery_stage": "medical"  # Special stage for medical
            }
        }
        scenarios.append(scenario)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        print(f"Saved {len(scenarios)} medical surgery scenarios to {output_path}")
    
    return scenarios

def preprocess_all_scenarios(csv_path: str, scenario_type: str = "auto", 
                           output_path: str = None, n_scenarios: int = None) -> List[Dict[str, Any]]:
    """
    General preprocessing function that auto-detects scenario type
    
    Args:
        csv_path: Path to CSV file
        scenario_type: "debt", "disaster", "student", "medical", or "auto"
        output_path: Optional path to save JSON
        n_scenarios: Maximum number of scenarios to process
    """
    
    # Read first few rows to detect type
    df = pd.read_csv(csv_path, nrows=5)
    columns = [col.lower() for col in df.columns]
    
    if scenario_type == "auto":
        # Auto-detect based on column names
        if any(col in ' '.join(columns) for col in ['creditor', 'debtor', 'overdue', 'balance']):
            scenario_type = "debt"
        elif any(col in ' '.join(columns) for col in ['disaster', 'survivor', 'rescue', 'eta']):
            scenario_type = "disaster"
        elif any(col in ' '.join(columns) for col in ['student', 'bedtime', 'sleep']):
            scenario_type = "student"
        elif any(col in ' '.join(columns) for col in ['patient', 'surgery', 'medical', 'surgeon']):
            scenario_type = "medical"
        else:
            # Default to debt collection format
            scenario_type = "debt"
    
    print(f"Detected scenario type: {scenario_type}")
    
    # Call appropriate preprocessing function
    if scenario_type == "debt":
        return preprocess_debt_scenarios(csv_path, output_path, n_scenarios)
    elif scenario_type == "disaster":
        return preprocess_disaster_rescue_scenarios(csv_path, output_path, n_scenarios)
    elif scenario_type == "student":
        return preprocess_student_sleep_scenarios(csv_path, output_path, n_scenarios)
    elif scenario_type == "medical":
        return preprocess_medical_surgery_scenarios(csv_path, output_path, n_scenarios)
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

# Example usage:
if __name__ == "__main__":
    # Process debt collection CSV
    debt_scenarios = preprocess_all_scenarios(
        csv_path="data/debt_collection.csv",
        scenario_type="debt",
        output_path="config/scenarios_debt.json",
        n_scenarios=10
    )
    
    # Process disaster rescue CSV
    disaster_scenarios = preprocess_all_scenarios(
        csv_path="data/disaster_rescue.csv", 
        scenario_type="disaster",
        output_path="config/scenarios_disaster.json",
        n_scenarios=10
    )
    
    # Process student sleep CSV
    student_scenarios = preprocess_all_scenarios(
        csv_path="data/student_sleep.csv",
        scenario_type="student", 
        output_path="config/scenarios_student.json",
        n_scenarios=10
    )
    
    # Process medical surgery CSV
    medical_scenarios = preprocess_all_scenarios(
        csv_path="data/medical_surgery.csv",
        scenario_type="medical",
        output_path="config/scenarios_medical.json",
        n_scenarios=10
    )
    
    # Combine all scenarios
    all_scenarios = debt_scenarios + disaster_scenarios + student_scenarios + medical_scenarios
    with open("config/scenarios_all.json", 'w') as f:
        json.dump(all_scenarios, f, indent=2)
    
    print(f"\nProcessed {len(all_scenarios)} total scenarios across 4 domains")