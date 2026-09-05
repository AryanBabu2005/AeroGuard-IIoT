def maintenance_agent_reasoning(sensor_data, predicted_rul):
    """
    Simulates an AI Agent's thought process.
    Uses 'Chain of Thought' to justify maintenance.
    """
    findings = []
    if sensor_data['s_11_mean'] > sensor_data['s_11'].mean():
        findings.append("Static Pressure (s_11) is trending above historical norms.")
    
    if predicted_rul < 50:
        decision = "CRITICAL: Initiate 'Phase 2' Shutdown Procedure."
    else:
        decision = "CONTINUE: System stable within stochastic bounds."
        
    return {
        "agent_thought": " | ".join(findings),
        "final_decision": decision
    }