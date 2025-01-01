import pandas as pd
import numpy as np
import vcfpy
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.stats import chi2
from tqdm import tqdm
from joblib import Parallel, delayed
from cyvcf2 import VCF
import cuml
from cuml.ensemble import RandomForestRegressor as cuRF
from cuml.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def calculate_allele_frequency(record):
    """
    Calculate allele frequency for a VCF record based on genotype data.
    Returns None if allele frequency cannot be computed.
    """
    total_alleles = 0
    alt_alleles = 0

    for call in record.calls:
        if call.data.get("GT"):  # Genotype field
            # Genotype is usually in the format "0/1", "1/1", "0|1", etc.
            alleles = call.data["GT"].replace('|', '/').split('/')
            for allele in alleles:
                if allele.isdigit():
                    total_alleles += 1
                    if int(allele) > 0:  # Non-reference alleles
                        alt_alleles += 1

    if total_alleles > 0:
        return alt_alleles / total_alleles
    return None  # Return None if total_alleles is 0 or missing data

def load_and_filter_data():
    if not os.path.exists('Filtered_Traits.csv') and not os.path.exists('Filtered_Genotype_Data.vcf'):
        # Load VCF file
        reader = vcfpy.Reader.from_path('Training_data/5_Genotype_Data_All_2014_2025_Hybrids.vcf')
        vcf_samples = set(reader.header.samples.names)
        
        # Load and clean trait data
        traits = pd.read_csv('Training_data/1_Training_Trait_Data_2014_2023.csv')
        traits = traits.dropna()

        # Extract hybrid names from the traits data
        csv_hybrids = set(traits['Hybrid'])
        matching_hybrids = vcf_samples.intersection(csv_hybrids)

        # Filter VCF records by MAF threshold
        maf_threshold = 0.05
        filtered_vcf_data = []
        for record in reader:
            af = calculate_allele_frequency(record)
            if af is not None and af >= maf_threshold:
                record_data = {"Hybrid": record.calls[0].sample}  # First sample assumed as an example
                filtered_vcf_data.append(record_data)

        # Create a DataFrame from filtered VCF data
        vcf_df = pd.DataFrame(filtered_vcf_data)
        vcf_df = vcf_df[vcf_df['Hybrid'].isin(matching_hybrids)]

        # Save filtered data
        traits.to_csv('Filtered_Traits.csv', index=False)
        with vcfpy.Writer.from_path('Filtered_Genotype_Data.vcf', reader.header) as writer:
            for record in filtered_vcf_data:
                writer.write_record(record)

    if not os.path.exists('processed_soil_data.csv'):
        soil_data = pd.read_csv('Training_data/3_Training_Soil_Data_2015_2023.csv')
        drop_columns = ["LabID", "Date Received", "Date Reported", "Texture", "Comments"]
        soil_data = soil_data.drop(columns=drop_columns)
        soil_data = soil_data.iloc[:, :-5]
        soil_data = soil_data[soil_data['Year'] != 2015]
        print(soil_data)

        # Ensure correct formatting
        split_columns = soil_data["Env"].str.rsplit("_", n=1)
        soil_data["Environment"] = split_columns.str[0]  # Assign the first part to 'Environment'
        soil_data["Year"] = split_columns.str[1].astype(int)
        # soil_data["Year"] = soil_data["Year"].astype(int)

        # Sort and backfill missing values
        soil_data = soil_data.sort_values(by=["Environment", "Year"])
        soil_data = soil_data.groupby("Environment").apply(lambda group: group.ffill().bfill()).reset_index(drop=True)

        # Interpolate remaining NaN values
        soil_data = soil_data.interpolate(method='linear', limit_direction='both', axis=0)

        # Save processed soil data
        soil_data.to_csv('processed_soil_data.csv', index=False)

    if not os.path.exists('processed_weather_data.csv'):
        weather_data = pd.read_csv('Training_data/4_Training_Weather_Data_2014_2023_seasons_only.csv')
        weather_data = weather_data.drop(columns=['Date'])
        weather_data = weather_data.groupby('Env', as_index=False).sum()
        weather_data.to_csv('processed_weather_data.csv', index=False)


    # Load processed files
    traits = pd.read_csv('Filtered_Traits.csv')
    soil_data = pd.read_csv('processed_soil_data.csv')
    genotype_data = pd.read_csv('genotype_matrix.csv')
    weather_data = pd.read_csv('processed_weather_data.csv')

    # Merge datasets
    merged_data = pd.merge(traits, genotype_data, on='Hybrid', how='inner')
    merged_data = pd.merge(merged_data, soil_data, on='Env', how='inner')
    merged_data = pd.merge(merged_data, weather_data, on='Env', how='inner')

    # Keep only rows present in all datasets
    merged_data = merged_data.dropna()

    # Save the final dataset
    merged_data.to_csv('Final_Merged_Data.csv', index=False)


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def perform_random_forest_regression(vcf_path, traits_path):
    """
    Perform random forest regression to predict yield from genotype data.
    
    Parameters:
    - vcf_path: Path to the filtered VCF file.
    - traits_path: Path to the filtered traits CSV file.
    """
    # Load filtered traits
    traits = pd.read_csv(traits_path)

    target_column = "Yield_Mg_ha"  
    if target_column not in traits.columns:
        raise ValueError(f"Target column '{target_column}' not found in traits data.")

    traits = traits.dropna(subset=[target_column])
    y = traits[target_column].values

    # Check if processed genotype data exists
    genotype_file = "genotype_matrix.csv"
    if not os.path.exists(genotype_file):
        # Initialize VCF reader
        vcf = VCF(vcf_path)
        
        # Extract sample indices once
        sample_indices = [vcf.samples.index(sample) for sample in vcf.samples]
        # Preallocate genotype matrix
        num_samples = len(sample_indices)
        num_records = sum(1 for _ in vcf)  # Get number of SNP records
        genotype_matrix = np.zeros((num_samples, num_records), dtype=np.int8)

        marker_names = []
        # Reset VCF and fill genotype matrix
        vcf = VCF(vcf_path)  # Re-initialize VCF iterator
        for idx, record in tqdm(enumerate(vcf), total=num_records, desc="Processing VCF records"):
            # Extract genotypes for the relevant samples
            genotypes = record.genotypes  # List of tuples (allele1, allele2, phase)
            genotype_matrix[:, idx] = [genotypes[i][0] + genotypes[i][1] for i in sample_indices]
            marker_name = record.ID if record.ID else f"{record.CHROM}:{record.POS}"
            marker_names.append(marker_name)

        # Convert to DataFrame
        genotype_df = pd.DataFrame(genotype_matrix, index=vcf.samples)
        genotype_df.columns = marker_names

        # Save genotype DataFrame to file
        genotype_df.to_csv(genotype_file)
    else:
        # Load the existing genotype DataFrame
        genotype_df = pd.read_csv(genotype_file)    

    # Merge traits with genotype data based on Hybrid
    vcf = VCF(vcf_path)

    soil_data = pd.read_csv('processed_soil_data.csv')
    weather_data = pd.read_csv('processed_weather_data.csv')
    EC_data = pd.read_csv('Training_data/6_Training_EC_Data_2014_2023.csv')

    genotype_df["Hybrid"] = vcf.samples
    genotype_merged_data = pd.merge(traits, genotype_df, on="Hybrid")
    soil_merged_data = pd.merge(genotype_merged_data, soil_data, on="Env")
    merged_data=pd.merge(soil_merged_data, weather_data, on='Env')
    # merged_data=pd.merge(EC_data, merged_data, on='Env')

    # merged_data.to_csv('genotype_soil_data.csv')

    # Prepare data for regression
    columns_to_drop = ["Hybrid", target_column, 'Year_x', 'Plot', 'Replicate', 'Block', 'Range', 'Pass', 'Plot_Area_ha', 'Stand_Count_plants', 'Stalk_Lodging_plants','Pollen_DAP_days', 'Silk_DAP_days', 'Plant_Height_cm', 'Ear_Height_cm', 'Root_Lodging_plants', 'Grain_Moisture', 'Twt_kg_m3']

    # Identify columns that cannot be converted to floats
    non_float_columns = merged_data.select_dtypes(include=['object']).columns
    non_convertible_columns = [col for col in non_float_columns if pd.to_numeric(merged_data[col], errors='coerce').isnull().any()]

    # Combine the lists of columns to drop
    all_columns_to_drop = set(columns_to_drop).union(non_convertible_columns)

    # Drop the columns from the DataFrame
    X = merged_data.drop(columns=all_columns_to_drop)
    # print(X)

    scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # print(X)

    y = merged_data[target_column]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train Random Forest Regressor using cuML
    rf = cuRF(n_estimators=1024, max_depth=2560, n_bins=528, min_samples_split=16, min_samples_leaf=4, split_criterion=2)
    print('Fitting')
    rf.fit(X_train, y_train)

    # Predict and evaluate
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Random Forest Regression Results:")
    print(f"Training Set: RMSE: {np.sqrt(train_mse):.4f}, R^2: {train_r2:.4f}")
    print(f"Test Set: RMSE: {np.sqrt(test_mse):.4f}, R^2: {test_r2:.4f}")

    # Plot feature importance
    # feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=feature_importances[:20], y=feature_importances.index[:20])
    # plt.title("Top 20 Feature Importances")
    # plt.xlabel("Importance")
    # plt.ylabel("Feature")
    # plt.tight_layout()
    # plt.savefig('importance.png')

    # Plot observed vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_train_pred, color='blue', label=f'Training Set (R^2: {train_r2:.4f}, RMSE: {np.sqrt(train_mse):.4f})', alpha=0.6, s=5)
    plt.scatter(y_test, y_test_pred, color='red', label=f'Test Set (R^2: {test_r2:.4f}, RMSE: {np.sqrt(test_mse):.4f})', alpha=0.6, s=5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--', label='Ideal Fit')
    plt.title("Observed vs Predicted")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig('observed_vs_predicted.png')

    testing_data = pd.read_csv('Testing_data/1_Submission_Template_2024.csv')

    testing_soil_data = pd.read_csv('Testing_data/3_Testing_Soil_Data_2024_imputed.csv')
    drop_columns = ["LabID", "Date Received", "Date Reported", "Texture", "Comments"]
    testing_soil_data = testing_soil_data.drop(columns=drop_columns)
    testing_soil_data = testing_soil_data.iloc[:, :-5]
    testing_soil_data = testing_soil_data.interpolate(method='linear', limit_direction='both', axis=0)
    testing_soil_data = testing_soil_data.groupby('Env').mean()

    testing_weather_data = pd.read_csv('Testing_data/4_Testing_Weather_Data_2024_seasons_only.csv')
    testing_weather_data = testing_weather_data.drop(columns=['Date'])
    testing_weather_data = testing_weather_data.groupby('Env', as_index=False).sum()

    testing_EC_data = pd.read_csv('Testing_data/6_Testing_EC_Data_2024.csv')

    testing_merged = pd.merge(testing_data, genotype_df, on='Hybrid')
    testing_merged = pd.merge(testing_merged, testing_soil_data, on='Env')
    testing_merged = pd.merge(testing_merged, testing_weather_data, on='Env')
    # testing_merged = pd.merge(testing_merged, testing_EC_data, on='Env')

    feature_columns = testing_merged.columns[testing_merged.columns.get_loc("Yield_Mg_ha")+1:]  # Get columns after 'Yield_Mg_ha'
    features = testing_merged[feature_columns]

    testing_merged['Yield_Mg_ha'] = rf.predict(features)

    # Select the required columns for the final DataFrame
    final_testing_df = testing_merged[['Env', 'Hybrid', 'Yield_Mg_ha']]

    # Save the resulting DataFrame to a CSV file
    output_path = 'final_yield_predictions.csv'
    final_testing_df.to_csv(output_path, index=False)


def main():

    print('Filtering data')
    load_and_filter_data()

    # Perform Random Forest Regression
    perform_random_forest_regression('5_Genotype_Data_All_2014_2025_Hybrids_filtered.vcf', 'Filtered_Traits.csv')

if __name__ == "__main__":
    main()