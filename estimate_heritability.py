import pandas as pd
import numpy as np
import vcfpy
import os

def load_and_filter_data():
    # Load VCF file
    reader = vcfpy.Reader.from_path('Training_data/5_Genotype_Data_All_2014_2025_Hybrids.vcf')
    vcf_samples = set(reader.header.samples.names)  # Extract sample names from the VCF

    # Load CSV file
    traits = pd.read_csv('Training_data/1_Training_Trait_Data_2014_2023.csv')
    nan_filtered_traits = traits.dropna()

    # Extract hybrid names from the CSV
    csv_hybrids = set(nan_filtered_traits['Hybrid'])

    # Find matching hybrids
    matching_hybrids = vcf_samples.intersection(csv_hybrids)

    # Filter CSV to keep only matching hybrids
    filtered_traits = nan_filtered_traits[nan_filtered_traits['Hybrid'].isin(matching_hybrids)]

    # Filter VCF samples and records
    filtered_vcf_header = reader.header.copy()
    filtered_vcf_header.samples.names = list(matching_hybrids)  # Update header to include only matching hybrids

    filtered_vcf_records = []
    for record in reader:
        filtered_calls = [call for call in record.calls if call.sample in matching_hybrids]
        if filtered_calls:
            record.calls = filtered_calls
            filtered_vcf_records.append(record)

    # Write the filtered VCF file
    with vcfpy.Writer.from_path('Filtered_Genotype_Data.vcf', filtered_vcf_header) as writer:
        for record in filtered_vcf_records:
            writer.write_record(record)

    # Save filtered CSV
    filtered_traits.to_csv('Filtered_Traits.csv', index=False)

def average_traits_by_hybrid():
    """
    Load the filtered trait CSV and compute the average for each hybrid across years,
    retaining only specific columns of interest.
    """
    # Load the filtered traits CSV
    traits = pd.read_csv('Filtered_Traits.csv')

    # Columns to average
    columns_to_average = [
        'Pollen_DAP_days', 'Silk_DAP_days', 'Plant_Height_cm', 'Ear_Height_cm',
        'Root_Lodging_plants', 'Stalk_Lodging_plants', 'Yield_Mg_ha',
        'Grain_Moisture', 'Twt_kg_m3'
    ]

    # Ensure the required columns exist
    missing_columns = [col for col in columns_to_average if col not in traits.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in the traits file: {missing_columns}")

    # Group by 'Hybrid' and compute the mean for the selected columns
    averaged_traits = (
        traits.groupby('Hybrid', as_index=False)[columns_to_average].mean()
    )

    # Save the averaged traits to a new CSV
    averaged_traits.to_csv('Averaged_Traits.csv', index=False)
    print(f"Averaged traits saved to 'Averaged_Traits.csv'")


def main():
    if not os.path.exists('Filtered_Traits.csv') and not os.path.exists('Filtered_Genotype_Data.vcf'):
        print('Filtering data')
        load_and_filter_data()

    # Compute averaged traits
    average_traits_by_hybrid()

    # Load and verify filtered data
    traits = pd.read_csv('Averaged_Traits.csv')
    genotypes = vcfpy.Reader.from_path('Filtered_Genotype_Data.vcf')

    print(f"Number of hybrids in traits: {len(set(traits['Hybrid'].tolist()))}")
    print(f"Number of hybrids in genotype data: {len(set(genotypes.header.samples.names))}")

if __name__ == "__main__":
    main()
