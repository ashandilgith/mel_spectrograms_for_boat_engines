import os
import json
import argparse
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Loads the .env file if it exists (ignored if using GitHub Codespaces Secrets)
load_dotenv()

def generate_owner_report(inference_output, telemetry_data, repair_history):
    """
    Synthesizes ML model acoustic detections with operating conditions
    and maintenance logs using the Google GenAI SDK.
    """
    # Automatically uses os.environ.get("GEMINI_API_KEY")
    client = genai.Client()
    
    prompt = f"""
    You are Piranaware's Senior Marine Diagnostic AI.
    Explain the following engine inspection results to a boat owner in plain English.
    
    1. ACOUSTIC ML CLASSIFICATION RESULTS:
    {json.dumps(inference_output, indent=2)}
    
    2. CURRENT TELEMETRY & AMBIENT DATA:
    {json.dumps(telemetry_data, indent=2)}
    
    3. KNOWN RECENT REPAIRS / NOTES:
    {json.dumps(repair_history, indent=2)}
    
    REQUIREMENTS:
    - Summarize what physical fault occurred based on the acoustic detections.
    - Correlate the acoustic fault with the telemetry.
    - Provide a concrete step-by-step checklist of what to inspect or repair.
    - Warn them if operating the boat in this condition risks catastrophic damage.
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    )
    return response.text

if __name__ == "__main__":
    # Production CLI entry point (No mock data)
    parser = argparse.ArgumentParser(description="Generate Piranaware Owner Report")
    parser.add_argument("--inference", type=str, required=True, help="Path to JSON file containing inference results")
    parser.add_argument("--telemetry", type=str, required=True, help="Path to JSON file containing telemetry data")
    parser.add_argument("--repairs", type=str, required=True, help="Text string of known recent repairs/notes")
    
    args = parser.parse_args()
    
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable is missing. Set it in .env or Codespaces Secrets.")
        
    with open(args.inference, 'r') as f:
        inference_data = json.load(f)
        
    with open(args.telemetry, 'r') as f:
        telemetry_data = json.load(f)
        
    report = generate_owner_report(inference_data, telemetry_data, args.repairs)
    print("\n=== PIRANAWARE OWNER DIAGNOSTIC REPORT ===\n")
    print(report)