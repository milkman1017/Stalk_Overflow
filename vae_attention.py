import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from scipy.spatial import distance
from sklearn.impute import KNNImputer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import shap
from sklearn.decomposition import PCA


def load_genomic_data(genome_file_path):
    """Load and preprocess SNP data."""
    data = pd.read_csv(genome_file_path, sep='\t', index_col=0)
    # Convert SNPs to numerical values using one-hot encoding
    unique_snps = sorted(set(data.values.flatten()) - {np.nan})
    snp_mapping = {snp: i for i, snp in enumerate(unique_snps)}

    def one_hot_encode(row):
        encoded = np.zeros((len(row), len(snp_mapping)))
        for idx, snp in enumerate(row):
            if snp in snp_mapping:
                encoded[idx, snp_mapping[snp]] = 1
        return encoded.flatten()

    # Apply one-hot encoding to each row
    encoded_data = np.array([one_hot_encode(row) for row in data.values])
    return encoded_data, data.index

def process_soil_data(soil_filepath, metadata_filepath):
    # Load data
    soil_data = pd.read_csv(soil_filepath)
    metadata = pd.read_csv(metadata_filepath)

    # Drop unnecessary columns from soil data
    drop_columns = ["Texture", "Comments", "LabID", "Date Received", "Date Reported", "Boron ppm B"]
    for col in drop_columns:
        try:
            soil_data = soil_data.drop(columns=[col])
        except:
            pass
    
    soil_data = soil_data.set_index(['Env', 'Year'])

    # Define helper function to compute average latitude and longitude of a field
    def compute_average_coordinates(row):
        lat_columns = [
            "Latitude_of_Field_Corner_#1 (lower left)",
            "Latitude_of_Field_Corner_#2 (lower right)",
            "Latitude_of_Field_Corner_#3 (upper right)",
            "Latitude_of_Field_Corner_#4 (upper left)"
        ]
        lon_columns = [
            "Longitude_of_Field_Corner_#1 (lower left)",
            "Longitude_of_Field_Corner_#2 (lower right)",
            "Longitude_of_Field_Corner_#3 (upper right)",
            "Longitude_of_Field_Corner_#4 (upper left)"
        ]
        avg_lat = row[lat_columns].dropna().mean()
        avg_lon = row[lon_columns].dropna().mean()
        return avg_lat, avg_lon

    # Add average coordinates to metadata
    metadata[['Avg_Lat', 'Avg_Lon']] = metadata.apply(lambda row: pd.Series(compute_average_coordinates(row)), axis=1)

    # Fill missing average latitude and longitude for environments
    for idx, row in metadata.iterrows():
        if pd.isnull(row['Avg_Lat']) or pd.isnull(row['Avg_Lon']):
            env_name, env_year = row['Env'].rsplit("_", 1)
            other_envs = metadata[(metadata['Env'].str.startswith(env_name + "_") & metadata[['Avg_Lat', 'Avg_Lon']].notnull().all(axis=1))]
            if not other_envs.empty:
                metadata.at[idx, 'Avg_Lat'] = other_envs.iloc[0]['Avg_Lat']
                metadata.at[idx, 'Avg_Lon'] = other_envs.iloc[0]['Avg_Lon']

    # Map average coordinates to soil data based on matching 'Env'
    soil_data = soil_data.merge(metadata[['Env', 'Avg_Lat', 'Avg_Lon']], on='Env', how='left')

    # Sort soil data by closeness to the average coordinates within each year
    soil_data['Year'] = soil_data['Env'].str.split("_").str[-1]  # Extract year from Env
    for year in soil_data['Year'].unique():
        soil_data_year_indices = soil_data[soil_data['Year'] == year].index
        soil_data_year = soil_data.loc[soil_data_year_indices]
        
        # Use Avg_Lat and Avg_Lon for distance calculation
        avg_lat_lon = soil_data_year[['Avg_Lat', 'Avg_Lon']].mean().values
        soil_data.loc[soil_data_year_indices, 'Distance'] = soil_data_year.apply(
            lambda row: distance.euclidean((row['Avg_Lat'], row['Avg_Lon']), avg_lat_lon),
            axis=1
        )
        # Sort and interpolate within the year
        soil_data.loc[soil_data_year_indices] = soil_data.loc[soil_data_year_indices].sort_values('Distance').interpolate(method='linear')

    # Fill remaining missing values using KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    numeric_cols = soil_data.select_dtypes(include=['float64', 'int64']).columns
    soil_data[numeric_cols] = imputer.fit_transform(soil_data[numeric_cols])

    # Drop the Avg_Lat, Avg_Lon, and Distance columns before returning
    soil_data = soil_data.drop(columns=['Avg_Lat', 'Avg_Lon', 'Year', 'Distance'], errors='ignore')
    soil_data = soil_data.set_index('Env')

    return soil_data

def process_weather_data(weather_filepath, metadata_filepath):
    # Load the data
    weather_data = pd.read_csv(weather_filepath)
    metadata = pd.read_csv(metadata_filepath)
    daymet_data = pd.read_csv('daymet_data_filtered.csv')

    # Merge weather data and Daymet data on Env and Date
    merged_data = pd.merge(weather_data, daymet_data, on=["Env", "Date"], how="inner")

    # Calculate the difference between Daymet "tmin (deg c)" and weather "T2M_MIN"
    merged_data["diff_tmin"] = np.abs(merged_data["tmin (deg c)"] - merged_data["T2M_MIN"])

    # Filter to central 60% of the distribution for each environment (Env)
    processed_data = []

    for env in merged_data["Env"].unique():
        env_data = merged_data[merged_data["Env"] == env].copy()

        # Calculate percentiles for central 60%
        lower_bound = env_data["diff_tmin"].quantile(0.2)
        upper_bound = env_data["diff_tmin"].quantile(0.8)

        # Filter data within the bounds
        env_data = env_data[(env_data["diff_tmin"] >= lower_bound) & (env_data["diff_tmin"] <= upper_bound)]

        # Append processed environment data
        processed_data.append(env_data)

    # Concatenate all processed environments
    filtered_data = pd.concat(processed_data)

    filtered_data = filtered_data.interpolate(method='linear', limit_direction='both', axis=0)
    filtered_data = filtered_data.drop(columns=['year','yday','dayl (s)','prcp (mm/day)','srad (W/m^2)','swe (kg/m^2)','tmax (deg c)','tmin (deg c)','vp (Pa)','diff_tmin', 'Date'])

    filtered_data = filtered_data.groupby('Env', as_index=True).sum()
    
    T_base = 10 

    # Calculate GDD for each day
    filtered_data["GDD"] = (
        (filtered_data["T2M_MAX"] + filtered_data["T2M_MIN"]) / 2 - T_base
    ).clip(lower=0)  # Ensures no negative GDD values

    # Aggregate GDD by environment
    aggregated_data = filtered_data.groupby("Env").agg({
        "RH2M": "mean",  # Relative Humidity
        "T2M_MAX": "mean",  # Mean max temperature
        "ALLSKY_SFC_SW_DWN": "sum",  # Total solar radiation
        "T2MWET": "mean",  # Wet-bulb temperature
        "GWETTOP": "mean",  # Top-layer soil moisture
        "QV2M": "mean",  # Specific humidity
        "GWETPROF": "mean",  # Soil profile moisture
        "T2M_MIN": "mean",  # Mean min temperature
        "T2MDEW": "mean",  # Dew point temperature
        "PS": "mean",  # Surface pressure
        "T2M": "mean",  # Mean daily temperature
        "GWETROOT": "mean",  # Root-zone soil moisture
        "ALLSKY_SFC_PAR_TOT": "sum",  # Total photosynthetically active radiation
        "WS2M": "mean",  # Mean wind speed
        "ALLSKY_SFC_SW_DNI": "sum",  # Total direct normal irradiance
        "PRECTOTCORR": "sum",  # Total precipitation
        "GDD": "sum",  # Total growing degree days
    })

    return aggregated_data

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def process_EC_data(EC_data_filepath, metadata_filepath):
    # Load data
    EC_data = pd.read_csv(EC_data_filepath)
    metadata = pd.read_csv(metadata_filepath)

    # Set Env and Year as index to prevent calculations on them
    EC_data.set_index(['Env'], inplace=True)

    # Define helper function to compute average latitude and longitude of a field
    def compute_average_coordinates(row):
        lat_columns = [
            "Latitude_of_Field_Corner_#1 (lower left)",
            "Latitude_of_Field_Corner_#2 (lower right)",
            "Latitude_of_Field_Corner_#3 (upper right)",
            "Latitude_of_Field_Corner_#4 (upper left)"
        ]
        lon_columns = [
            "Longitude_of_Field_Corner_#1 (lower left)",
            "Longitude_of_Field_Corner_#2 (lower right)",
            "Longitude_of_Field_Corner_#3 (upper right)",
            "Longitude_of_Field_Corner_#4 (upper left)"
        ]
        avg_lat = row[lat_columns].dropna().mean()
        avg_lon = row[lon_columns].dropna().mean()
        return avg_lat, avg_lon

    # Add average coordinates to metadata
    metadata[['Avg_Lat', 'Avg_Lon']] = metadata.apply(lambda row: pd.Series(compute_average_coordinates(row)), axis=1)

    # Fill missing average latitude and longitude for environments
    for idx, row in metadata.iterrows():
        if pd.isnull(row['Avg_Lat']) or pd.isnull(row['Avg_Lon']):
            env_name, env_year = row['Env'].rsplit("_", 1)
            previous_years = metadata[(metadata['Env'].str.startswith(env_name + "_") & metadata['Year'] < int(env_year))]
            previous_years = previous_years.dropna(subset=['Avg_Lat', 'Avg_Lon']).sort_values('Year', ascending=False)
            if not previous_years.empty:
                metadata.at[idx, 'Avg_Lat'] = previous_years.iloc[0]['Avg_Lat']
                metadata.at[idx, 'Avg_Lon'] = previous_years.iloc[0]['Avg_Lon']

    # Handle missing entries in soil data
    for idx, row in EC_data.iterrows():
        if row.isnull().any():
            # Find corresponding Env and Year in metadata
            env = idx[0]
            year = idx[1]
            field_metadata = metadata[(metadata['Env'] == env) & (metadata['Year'] == year)]

            if not field_metadata.empty:
                avg_lat, avg_lon = field_metadata.iloc[0][['Avg_Lat', 'Avg_Lon']]

                # Compute distances to other fields
                metadata['Distance'] = metadata.apply(
                    lambda r: distance.euclidean((avg_lat, avg_lon), (r['Avg_Lat'], r['Avg_Lon'])), axis=1
                )

                # Find the 5 closest fields
                closest_fields = metadata.nsmallest(5, 'Distance')

                # Get soil data for the closest fields
                closest_soil_data = EC_data[EC_data.index.get_level_values('Env').isin(closest_fields['Env'])]

                # Interpolate missing values across years
                for col in EC_data.columns:
                    if pd.isnull(row[col]):
                        closest_col_values = closest_soil_data.groupby(level='Year')[col].mean()
                        if not closest_col_values.empty:
                            EC_data.at[idx, col] = np.interp(
                                year,
                                closest_col_values.index,
                                closest_col_values.values
                            )

    # Standardize the data for PCA
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(EC_data)

    n_components=2
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(standardized_data)

    # Print variance explained by each principal component
    variance_explained = pca.explained_variance_ratio_
    print("Variance explained by each principal component:")
    for i, variance in enumerate(variance_explained, start=1):
        print(f"PC{i}: {variance:.4f}")

    # Convert PCA result back to a DataFrame
    pca_df = pd.DataFrame(pca_result, index=EC_data.index, columns=[f"PC{i}" for i in range(1, n_components+1)])

    return pca_df

def build_encoder(input_dim, latent_dim):
    """Build the encoder part of the VAE."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(1536, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)  # Add Batch Normalization after the first dense layer
    x = layers.Dense(1536, activation="relu")(x)
    x = layers.BatchNormalization()(x)  # Add Batch Normalization after the second dense layer
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

def build_decoder(latent_dim, output_dim):
    """Build the decoder part of the VAE."""
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(1028, activation="relu")(latent_inputs)
    x = layers.Dense(2048, activation="relu")(x)
    outputs = layers.Dense(output_dim, activation="sigmoid")(x)

    decoder = models.Model(latent_inputs, outputs, name="decoder")
    return decoder

def build_vae(encoder, decoder):
    """Combine encoder and decoder to build the VAE."""
    inputs = encoder.input
    z_mean, z_log_var, z = encoder(inputs)
    reconstructed = decoder(z)

    def vae_loss(y_true, y_pred):
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss

    vae = models.Model(inputs, reconstructed, name="vae")
    vae.add_loss(vae_loss(inputs, reconstructed))
    return vae

def train_vae(vae, data, batch_size=1024, epochs=100):
    """Train the VAE on the provided data."""
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    history = vae.fit(data, data, batch_size=batch_size, epochs=epochs)
    return history

def plot_training_loss(history, output_path="training_loss.png"):
    """Plot the training loss over epochs."""
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def analyze_latent_space(encoder, data, index, output_path="latent_space.csv"):
    """Extract and save the latent space representation to a CSV file."""
    z_mean, _, _ = encoder.predict(data)
    latent_df = pd.DataFrame(z_mean, index=index, columns=[f"z{i+1}" for i in range(z_mean.shape[1])])
    latent_df.index.name = "Hybrid"
    latent_df.to_csv(output_path)
    print(f"Latent space representation saved to {output_path}")

def reconstruction_error(vae, data, output_path="reconstruction_error.png"):
    """Compute and plot reconstruction errors."""
    reconstructed = vae.predict(data)
    errors = np.mean(np.square(data - reconstructed), axis=1)
    plt.figure()
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.savefig(output_path)
    plt.close()

def split_inputs(data, genomic_columns, weather_columns, soil_columns, EC_columns):
    genomic_data = data[genomic_columns].values
    weather_data = data[weather_columns].values
    soil_data = data[soil_columns].values
    EC_data = data[EC_columns].values
    yield_data = data["Yield_Mg_ha"].values
    return genomic_data, weather_data, soil_data, EC_data, yield_data

def build_attention_model(genomic_dim, weather_dim, soil_dim, EC_dim):
    genomic_input = layers.Input(shape=(genomic_dim,), name="Genomic")
    weather_input = layers.Input(shape=(weather_dim,), name="Weather")
    soil_input = layers.Input(shape=(soil_dim,), name="Soil")
    EC_input = layers.Input(shape=(EC_dim,), name="EC")

    # Regularization parameters
    l1_reg = 1e-5
    l2_reg = 1e-4

    # Individual pathways with LeakyReLU activation and regularization
    genomic_dense = layers.Dense(64, kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(genomic_input)
    genomic_dense = layers.LeakyReLU()(genomic_dense)

    weather_dense = layers.Dense(64, kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(weather_input)
    weather_dense = layers.LeakyReLU()(weather_dense)

    soil_dense = layers.Dense(64, kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(soil_input)
    soil_dense = layers.LeakyReLU()(soil_dense)

    EC_dense = layers.Dense(64, kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(EC_input)
    EC_dense = layers.LeakyReLU()(EC_dense)

    # Cross-stitching: genomic with weather, soil, and EC using Multi-Head Attention
    # Genomic + Weather
    stitch_gw = layers.Concatenate()([genomic_dense, weather_dense])
    stitch_gw = layers.Reshape((1, -1))(stitch_gw)  # Reshape for MultiHeadAttention
    stitch_gw = layers.MultiHeadAttention(num_heads=4, key_dim=4)(stitch_gw, stitch_gw)
    stitch_gw = layers.Flatten()(stitch_gw)
    stitch_gw = layers.Dense(128, kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(stitch_gw)
    stitch_gw = layers.LeakyReLU()(stitch_gw)
    stitch_gw = layers.BatchNormalization()(stitch_gw)

    # Genomic + Soil
    stitch_gs = layers.Concatenate()([genomic_dense, soil_dense])
    stitch_gs = layers.Reshape((1, -1))(stitch_gs)  # Reshape for MultiHeadAttention
    stitch_gs = layers.MultiHeadAttention(num_heads=4, key_dim=4)(stitch_gs, stitch_gs)
    stitch_gs = layers.Flatten()(stitch_gs)
    stitch_gs = layers.Dense(128, kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(stitch_gs)
    stitch_gs = layers.LeakyReLU()(stitch_gs)
    stitch_gs = layers.BatchNormalization()(stitch_gs)

    # Genomic + EC
    stitch_ge = layers.Concatenate()([genomic_dense, EC_dense])
    stitch_ge = layers.Reshape((1, -1))(stitch_ge)  # Reshape for MultiHeadAttention
    stitch_ge = layers.MultiHeadAttention(num_heads=4, key_dim=4)(stitch_ge, stitch_ge)
    stitch_ge = layers.Flatten()(stitch_ge)
    stitch_ge = layers.Dense(128, kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(stitch_ge)
    stitch_ge = layers.LeakyReLU()(stitch_ge)
    stitch_ge = layers.BatchNormalization()(stitch_ge)

    # Combine all stitches and pass to final multi-head attention
    combined_stitch = layers.Concatenate()([stitch_gw, stitch_gs, stitch_ge])
    combined_stitch = layers.Reshape((3, 128))(combined_stitch)  # Reshape for MultiHeadAttention

    # Final Self-Attention Layer
    self_attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=8)(combined_stitch, combined_stitch)
    self_attention_output = layers.LayerNormalization()(self_attention_output + combined_stitch)  # Residual connection

    # Fully connected layers with regularization
    flattened = layers.Flatten()(self_attention_output)
    x = layers.BatchNormalization()(flattened)
    x = layers.Dense(512, kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    # Output layer
    outputs = layers.Dense(1, kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg))(x)
    norm = layers.BatchNormalization()(outputs)
    outputs = layers.Add()([outputs, norm])

    # Model compilation
    model = models.Model(inputs=[genomic_input, weather_input, soil_input, EC_input], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse', metrics=["mae"])

    model.summary()
    return model

def train_attention_model(data, genomic_columns, weather_columns, soil_columns, EC_columns, target_column, 
                          batch_size=2048, epochs=500, k_folds=2, output_plot="observed_vs_expected.png"):
    # Split inputs
    genomic_data, weather_data, soil_data, EC_data, y = split_inputs(data, genomic_columns, weather_columns, soil_columns, EC_columns)

    data = data.set_index(['Hybrid', 'Env', 'Year'])
    env_names = data.index.get_level_values('Env').unique()
    print(data)

    # Standardize inputs
    scaler_genomic = StandardScaler().fit(genomic_data)
    scaler_weather = StandardScaler().fit(weather_data)
    scaler_soil = StandardScaler().fit(soil_data)
    scaler_EC = StandardScaler().fit(EC_data)

    genomic_data = scaler_genomic.transform(genomic_data)
    weather_data = scaler_weather.transform(weather_data)
    soil_data = scaler_soil.transform(soil_data)
    EC_data = scaler_EC.transform(EC_data)

    # Split data into training (non-2023) and test (2023) sets
    train_idx = data.index.get_level_values('Year') != 2023
    test_idx = data.index.get_level_values('Year') == 2023

    X_train = [genomic_data[train_idx], weather_data[train_idx], soil_data[train_idx], EC_data[train_idx]]
    X_test = [genomic_data[test_idx], weather_data[test_idx], soil_data[test_idx], EC_data[test_idx]]
    y_train, y_test = y[train_idx], y[test_idx]
    test_envs = data.index[test_idx].get_level_values('Env')

    # K-Folds on training data
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    models = []
    training_predictions = []
    training_envs = data.index[train_idx].get_level_values('Env')
    results = []

    for fold, (k_train_idx, k_val_idx) in enumerate(kfold.split(X_train[0])):
        print(f"Training fold {fold + 1}")

        X_k_train = [x[k_train_idx] for x in X_train]
        X_k_val = [x[k_val_idx] for x in X_train]
        y_k_train, y_k_val = y_train[k_train_idx], y_train[k_val_idx]

        model = build_attention_model(
            genomic_dim=genomic_data.shape[1], 
            weather_dim=weather_data.shape[1], 
            soil_dim=soil_data.shape[1], 
            EC_dim=EC_data.shape[1]
        )

        early_stopping = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=False)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=1e-7)

        history = model.fit(
            X_k_train, y_k_train, validation_data=(X_k_val, y_k_val),
            batch_size=batch_size, epochs=epochs, callbacks=[early_stopping, reduce_lr], verbose=1
        )

        models.append(model)
        model.save(f"attention_model_fold_{fold}.h5")

        y_k_val_pred = model.predict(X_k_val).flatten()
        r2 = r2_score(y_k_val, y_k_val_pred)
        rmse = mean_squared_error(y_k_val, y_k_val_pred, squared=False)
        print(f"Fold {fold + 1} R^2: {r2:.2f}, RMSE: {rmse:.2f}")
        results.append((fold, r2, rmse))

        # Plot observed vs expected for the current fold
        plt.figure(figsize=(8, 6))
        plt.scatter(y_k_train, model.predict(X_k_train).flatten(), color="blue", alpha=0.6, label=f"Training (R^2={r2:.2f}, RMSE={rmse:.2f})", s=3)
        plt.scatter(y_k_val, y_k_val_pred, color="red", alpha=0.6, label=f"Validation (R^2={r2:.2f}, RMSE={rmse:.2f})", s=5)
        plt.plot([min(y), max(y)], [min(y), max(y)], color="black", linestyle="--", label="Ideal")
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Observed vs Predicted")
        plt.savefig(f"observed_vs_predicted_fold_{fold + 1}.png")
        plt.close()

        # Collect training predictions for observed vs expected plot
        training_predictions.extend((y_k_val, y_k_val_pred))

    # Predict on test data using all models and average predictions
    test_predictions = []
    for model in models:
        test_predictions.append(model.predict(X_test).flatten())
    y_test_pred = np.mean(test_predictions, axis=0)

    # Calculate RMSE and R^2 for test data
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

    # Calculate environment-specific R^2
    env_r2s = {}
    for env in test_envs.unique():
        env_mask = test_envs == env
        env_r2 = r2_score(y_test[env_mask], y_test_pred[env_mask])
        env_r2s[env] = env_r2
        print(f"R^2 for 2023 {env}: {env_r2:.2f}")
    average_r2 = np.mean(list(env_r2s.values()))
    print(f"Average total R^2: {average_r2:.2f}")

    # Plot observed vs. expected
    plt.figure(figsize=(10, 6))

    # Training data (blue)
    plt.scatter(training_predictions[0], training_predictions[1], color="blue", alpha=0.6, label="Training", s=3)

    # Test data (color-coded by environment)
    env_colors = plt.cm.tab10(np.linspace(0, 1, len(test_envs.unique())))
    for i, env in enumerate(test_envs.unique()):
        env_mask = test_envs == env
        plt.scatter(y_test[env_mask], y_test_pred[env_mask], color=env_colors[i], alpha=0.6, label=f"2023 {env}", s=5)

    # Plot formatting
    plt.plot([min(y), max(y)], [min(y), max(y)], color="black", linestyle="--", label="Ideal")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.legend()
    plt.title("Observed vs Predicted")
    plt.savefig(output_plot)
    plt.close()

    print(f"Test R^2: {test_r2:.2f}, RMSE: {test_rmse:.2f}")
    print("Training and evaluation completed.")


def predict_yield_attention(test_data, model_paths, genomic_columns, weather_columns, soil_columns, EC_columns):
    genomic_data, weather_data, soil_data, EC_data, _ = split_inputs(test_data, genomic_columns, weather_columns, soil_columns, EC_columns)

    scaler_genomic = StandardScaler().fit(genomic_data)
    scaler_weather = StandardScaler().fit(weather_data)
    scaler_soil = StandardScaler().fit(soil_data)
    scaler_EC = StandardScaler().fit(EC_data)

    genomic_data = scaler_genomic.transform(genomic_data)
    weather_data = scaler_weather.transform(weather_data)
    soil_data = scaler_soil.transform(soil_data)
    EC_data = scaler_EC.transform(EC_data)

    predictions = np.zeros(len(test_data))
    for model_path in model_paths:
        model = load_model(model_path)
        predictions += model.predict([genomic_data, weather_data, soil_data, EC_data]).flatten()

    predictions /= len(model_paths)
    test_data["Yield_Mg_ha"] = predictions
    test_data[["Env", "Hybrid", "Yield_Mg_ha"]].to_csv("final_predictions_attention.csv", index=False)

if __name__ == "__main__":
    # File path to SNP data
    genome_file_path = "5_Genotype_Data_All_2014_2025_Hybrids_filtered_letters.csv"

    mode = 'Training'
    vae = False

    if mode == 'Training':
        soil_data = process_soil_data(soil_filepath='Training_data/3_Training_Soil_Data_2015_2023.csv', metadata_filepath='Training_data/2_Training_Meta_Data_2014_2023.csv')
        weather_data = process_weather_data(weather_filepath='Training_data/4_Training_Weather_Data_2014_2023_seasons_only.csv', metadata_filepath='Training_data/2_Training_Meta_Data_2014_2023.csv')
        EC_data = process_EC_data(EC_data_filepath='Training_data/6_Training_EC_Data_2014_2023.csv', metadata_filepath='Training_data/2_Training_Meta_Data_2014_2023.csv')

        if vae:
            # Load and preprocess the data
            snp_data, index = load_genomic_data(genome_file_path)

            # Define dimensions
            input_dim = snp_data.shape[1]
            latent_dim = 200 # Adjust based on desired dimensionality

            # Build the encoder, decoder, and VAE
            encoder = build_encoder(input_dim, latent_dim)
            decoder = build_decoder(latent_dim, input_dim)
            vae = build_vae(encoder, decoder)

            # Train the VAE
            history = train_vae(vae, snp_data)

            # # Save the encoder for latent space analysis
            encoder.save("vae_encoder.h5")

            # # Plot training loss
            plot_training_loss(history)

            # # Extract and save latent space representation
            analyze_latent_space(encoder, snp_data, index)

            # # Compute and plot reconstruction error
            reconstruction_error(vae, snp_data)

        latent_genomic_data = pd.read_csv("latent_space.csv")

        trait_data = pd.read_csv("Training_data/1_Training_Trait_Data_2014_2023.csv")
        print(np.shape(trait_data))
        trait_data = trait_data.dropna(subset=['Yield_Mg_ha'])
        print(np.shape(trait_data))
        trait_data = trait_data.groupby(['Env', 'Hybrid', 'Year'], as_index=False)['Yield_Mg_ha'].mean()
        print(np.shape(trait_data))
        print(trait_data)

        genomic_columns = latent_genomic_data.columns.drop('Hybrid')
        weather_columns = weather_data.columns
        soil_columns = soil_data.columns
        EC_columns = EC_data.columns

        merged_df = pd.merge(latent_genomic_data, trait_data, on="Hybrid")
        merged_df = pd.merge(merged_df, weather_data, on="Env")
        merged_df = pd.merge(merged_df, soil_data, on="Env")
        merged_df = pd.merge(merged_df, EC_data, on="Env")
        print(merged_df)

        train_attention_model(merged_df, genomic_columns, weather_columns, soil_columns, EC_columns, "Yield_Mg_ha")

    if mode == 'Predict':
        latent_genomic_data = pd.read_csv("latent_space.csv")
        trait_data = pd.read_csv('Testing_data/1_Submission_Template_2024.csv')

        soil_data = process_soil_data(soil_filepath='Testing_data/3_Testing_Soil_Data_2024_imputed.csv', metadata_filepath='Testing_data/2_Testing_Meta_Data_2024.csv').groupby('Env').mean()
        weather_data = pd.read_csv('Testing_data/4_Testing_Weather_Data_2024_seasons_only.csv')
        weather_data = weather_data.drop(columns=['Date'])
        weather_data = weather_data.groupby('Env', as_index=False).sum()

        EC_data = process_EC_data(EC_data_filepath='Testing_data/6_Testing_EC_Data_2024.csv', metadata_filepath='Testing_data/2_Testing_Meta_Data_2024.csv').interpolate()

        merged_df = pd.merge(latent_genomic_data, trait_data, on='Hybrid')
        merged_df = pd.merge(merged_df, weather_data, on='Env')
        merged_df = pd.merge(merged_df, soil_data, on='Env')
        merged_df = pd.merge(merged_df, EC_data, on='Env')

        genomic_columns = latent_genomic_data.columns.drop('Hybrid')
        weather_columns = weather_data.columns.drop('Env')
        soil_columns = soil_data.columns
        EC_columns = EC_data.columns

        predict_yield_attention(merged_df, [f"attention_model_fold_{i}.h5" for i in range(5)], genomic_columns, weather_columns, soil_columns, EC_columns)
