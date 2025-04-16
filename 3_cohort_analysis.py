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

    # Step 1: Convert CSV to Parquet
    pl.read_csv(input_file).write_parquet("patients_large.parquet")

    # Step 2: Lazy pipeline from Parquet
    breaks = [10, 18.5, 25, 30, 60]
    bin_labels = ["Underweight", "Normal", "Overweight", "Obese"]

    lazy_result = (
        pl.scan_parquet("patients_large.parquet")
        .filter((pl.col("BMI") >= 10) & (pl.col("BMI") <= 60))
        .with_columns([
            pl.col("BMI").cut(breaks=breaks, left_closed=True).alias("bmi_bin"),
            pl.when(pl.col("Outcome") == 1)
              .then("Diabetic")
              .otherwise("Non-Diabetic")
              .alias("diagnosis")
        ])
        .group_by(["bmi_bin", "diagnosis"])
        .agg([
            pl.col("Glucose").mean().alias("avg_glucose"),
            pl.len().alias("patient_count"),
            pl.col("Age").mean().alias("avg_age")
        ])
        .collect(streaming=True)  # enable streaming for big data
    )

    # Step 3: Replace numeric bin with labels
    result = lazy_result.with_columns([
        pl.when(pl.col("bmi_bin") == 0).then(bin_labels[0])
        .when(pl.col("bmi_bin") == 1).then(bin_labels[1])
        .when(pl.col("bmi_bin") == 2).then(bin_labels[2])
        .when(pl.col("bmi_bin") == 3).then(bin_labels[3])
        .otherwise("Unknown")
        .alias("bmi_range")
    ]).drop("bmi_bin")

    return result

def main():
    input_file = "patients_large.csv"
    results = analyze_patient_cohorts(input_file)

    print("\nCohort Analysis Summary:")
    print(results)

    # Save results
    results.write_csv("cohort_analysis_summary.csv")

if __name__ == "__main__":
    main()
