import polars as pl
import os

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
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found. Please run generate_large_health_data.py first.")

    # Convert CSV to Parquet for efficient processing
    pl.read_csv(input_file).write_parquet("patients_large.parquet", compression="zstd")

    # Manually define BMI ranges using when().then().otherwise()
    bmi_range_expr = (
        pl.when(pl.col("BMI") < 18.5).then(pl.lit("Underweight"))
        .when(pl.col("BMI") < 25).then(pl.lit("Normal"))
        .when(pl.col("BMI") < 30).then(pl.lit("Overweight"))
        .when(pl.col("BMI") <= 60).then(pl.lit("Obese"))
        .otherwise(None)
    )

    # Analyze cohorts using Polars' lazy API
    cohort_results = (
        pl.scan_parquet("patients_large.parquet")
        .filter((pl.col("BMI") >= 10) & (pl.col("BMI") <= 60))
        .with_columns([
            bmi_range_expr.alias("bmi_range")
        ])
        .group_by("bmi_range")
        .agg([
            pl.col("Glucose").mean().alias("avg_glucose"),
            pl.len().alias("patient_count"),  # Replaces deprecated pl.count()
            pl.col("Age").mean().alias("avg_age")
        ])
        .sort("bmi_range")
        .collect()
    )

    return cohort_results

def main():
    input_file = "patients_large.csv"
    results = analyze_patient_cohorts(input_file)

    print("\nCohort Analysis Results:")
    print(results)

if __name__ == "__main__":
    main()

