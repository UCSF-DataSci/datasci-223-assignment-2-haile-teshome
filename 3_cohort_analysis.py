import polars as pl

def analyze_patient_cohorts(input_file: str) -> pl.DataFrame:
    """
    Analyze patient cohorts based on BMI ranges.
    
    Args:
        input_file: Path to the input CSV file
        
    Returns:
        DataFrame containing cohort analysis results with columns:
        - bmi_range: The BMI range (e.g., "Underweight", "Normal", "Overweight", "Obese")
        - avg_glucose: Mean glucose level by BMI range
        - patient_count: Number of patients by BMI range
        - avg_age: Mean age by BMI range
    """
    
    # Create a lazy query to analyze cohorts
    cohort_results = pl.scan_csv(input_file).pipe(
        lambda df: df.filter((pl.col("BMI") >= 10) & (pl.col("BMI") <= 60))
    ).pipe(
        lambda df: df.with_columns([
            pl.col("BMI").cut(
                breaks=[10, 18.5, 25, 30, 60],
                labels=["Underweight", "Normal", "Overweight", "Obese"],
                left_closed=True
            ).alias("bmi_range"),
            pl.when(pl.col("Outcome") == 1)
              .then("Diabetic")
              .otherwise("Non-Diabetic")
              .alias("diagnosis")
    ]).pipe(
        lambda df: df.group_by(["bmi_range", "diagnosis"]).agg([
            pl.col("Glucose").mean().alias("avg_glucose"),
            pl.count().alias("patient_count"),
            pl.col("Age").mean().alias("avg_age")
        ])
    ).collect()
    
    return cohort_results

def main():
    # Input file
    input_file = "patients_large.csv"
    
    # Run analysis
    results = analyze_patient_cohorts(input_file)
    
    # Print summary statistics
    print("\nCohort Analysis Summary:")
    print(results)
    results.write_csv("cohort_analysis_summary.csv")

if __name__ == "__main__":
    main() 
