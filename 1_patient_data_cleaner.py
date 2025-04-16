#!/usr/bin/env python3
"""
Patient Data Cleaner

This script standardizes and filters patient records according to specific rules:

Data Cleaning Rules:
1. Names: Capitalize each word (e.g., "john smith" -> "John Smith")
2. Ages: Convert to integers, set invalid ages to 0
3. Filter: Remove patients under 18 years old
4. Remove any duplicate records

Input JSON format:
    [
        {
            "name": "john smith",
            "age": "32",
            "gender": "male",
            "diagnosis": "hypertension"
        },
        ...
    ]

Output:
- Cleaned list of patient dictionaries
- Each patient should have:
  * Properly capitalized name
  * Integer age (≥ 18)
  * Original gender and diagnosis preserved
- No duplicate records
- Prints cleaned records to console

Example:
    Input: {"name": "john smith", "age": "32", "gender": "male", "diagnosis": "flu"}
    Output: {"name": "John Smith", "age": 32, "gender": "male", "diagnosis": "flu"}

Usage:
    python patient_data_cleaner.py
"""

import json
import os

def load_patient_data(filepath):
    """
    Load patient data from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        list: List of patient dictionaries
    """
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading file: {e}")


def clean_patient_data(patients):
    """
    Clean patient data by:
    - Capitalizing names
    - Converting ages to integers
    - Filtering out patients under 18
    - Removing duplicates
    
    Args:
        patients (list): List of patient dictionaries
        
    Returns:
        list: Cleaned list of patient dictionaries
    """
    cleaned_patients = []
    seen = set()
    
    for patient in patients:
        patient['name'] = patient['name'].title()
        
        try:
            patient['age'] = int(patient['age'])
        except (ValueError, TypeError):
            patient['age'] = 0

        if patient['age'] < 18:
            continue

        identifier = json.dumps(patient, sort_keys=True)
        if identifier in seen:
            continue
        seen.add(identifier)

        cleaned_patients.append(patient)
    
    if not cleaned_patients:
        return []

    return cleaned_patients

def main():
    """Main function to run the script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'raw', 'patients.json')
    
    patients = load_patient_data(data_path)
    
    if not patients:
        print("No valid data to clean.")
        return
    
    cleaned_patients = clean_patient_data(patients)
    
    if not cleaned_patients:
        print("No patients meet the criteria.")
        return

    print("Cleaned Patient Data:")
    for patient in cleaned_patients:
        print(f"Name: {patient['name']}, Age: {patient['age']}, Gender: {patient['gender']}, Diagnosis: {patient['diagnosis']}")
    
    return cleaned_patients

if __name__ == "__main__":
    main()
