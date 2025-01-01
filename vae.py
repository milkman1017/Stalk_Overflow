import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.spatial import distance
from sklearn.impute import KNNImputer

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
    drop_columns = ["Texture", "Comments", "LabID","Date Received","Date Reported"]
    soil_data = soil_data.drop(columns=drop_columns, errors='ignore')

    # Set Env and Year as index to prevent calculations on them
    soil_data.set_index(['Env', 'Year'], inplace=True)

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
    for idx, row in soil_data.iterrows():
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
                closest_soil_data = soil_data[soil_data.index.get_level_values('Env').isin(closest_fields['Env'])]

                # Interpolate missing values across years
                for col in soil_data.columns:
                    if pd.isnull(row[col]):
                        closest_col_values = closest_soil_data.groupby(level='Year')[col].mean()
                        if not closest_col_values.empty:
                            soil_data.at[idx, col] = np.interp(
                                year,
                                closest_col_values.index,
                                closest_col_values.values
                            )

    # Use KNN to fill in remaining missing values
    knn_imputer = KNNImputer(n_neighbors=5)
    soil_data.iloc[:, :] = knn_imputer.fit_transform(soil_data)
    return soil_data

def process_weather_data(weather_filepath, metadata_filepath):
    weather_data = pd.read_csv(weather_filepath)
    metadata = pd.read_csv(metadata_filepath)






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

def train_vae(vae, data, batch_size=32, epochs=100):
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

def merge_training_data():
    latent_genomic_data = pd.read_csv("latent_space.csv")

    trait_data = pd.read_csv("Training_data/1_Training_Trait_Data_2014_2023.csv")
    trait_data = trait_data.dropna()
    trait_data = trait_data.drop(columns=[
    'Year',
    'Field_Location',
    'Experiment',
    'Replicate',
    'Block',
    'Plot',
    'Range',
    'Pass',
    'Hybrid_orig_name',
    'Hybrid_Parent1',
    'Hybrid_Parent2',
    'Plot_Area_ha',
    'Date_Planted',
    'Date_Harvested',
    'Stand_Count_plants',
    'Pollen_DAP_days',
    'Silk_DAP_days',
    'Plant_Height_cm',
    'Ear_Height_cm',
    'Root_Lodging_plants',
    'Stalk_Lodging_plants',
    'Grain_Moisture',
    'Twt_kg_m3'
])
    print(trait_data)

    training_soil_data = pd.read_csv('Training_data/3_Training_Soil_Data_2015_2023.csv')
    drop_columns = ["LabID", "Date Received", "Date Reported", "Texture", "Comments"]
    training_soil_data = training_soil_data.drop(columns=drop_columns)
    training_soil_data = training_soil_data.iloc[:, :-5]
    training_soil_data = training_soil_data.interpolate(method='linear', limit_direction='both', axis=0)
    training_soil_data = training_soil_data.groupby('Env').mean()

    training_weather_data = pd.read_csv('/home/wimahler/Stalk_Overflow/Training_data/4_Training_Weather_Data_2014_2023_full_year.csv')
    training_weather_data = training_weather_data.drop(columns=['Date'])
    training_weather_data = training_weather_data.groupby('Env', as_index=False).sum()

    EC_data = pd.read_csv('Training_data/6_Training_EC_Data_2014_2023.csv')

    trainning_merged = pd.merge(trait_data, latent_genomic_data, on='Hybrid')
    trainning_merged = pd.merge(trainning_merged, training_soil_data, on='Env')
    trainning_merged = pd.merge(trainning_merged, training_weather_data, on='Env')
    trainning_merged = pd.merge(trainning_merged, EC_data, on='Env')

    return trainning_merged

def merge_test_data():
    latent_genomic_data = pd.read_csv("latent_space.csv")

    trait_data = pd.read_csv("Testing_data/1_Submission_Template_2024.csv")

    training_soil_data = pd.read_csv('Testing_data/3_Testing_Soil_Data_2024_imputed.csv')
    drop_columns = ["LabID", "Date Received", "Date Reported", "Texture", "Comments"]
    training_soil_data = training_soil_data.drop(columns=drop_columns)
    training_soil_data = training_soil_data.iloc[:, :-4]
    training_soil_data = training_soil_data.interpolate(method='linear', limit_direction='both', axis=0)
    training_soil_data = training_soil_data.groupby('Env').mean()

    training_weather_data = pd.read_csv('Testing_data/4_Testing_Weather_Data_2024_full_year.csv')
    training_weather_data = training_weather_data.drop(columns=['Date'])
    training_weather_data = training_weather_data.groupby('Env', as_index=False).sum()

    EC_data = pd.read_csv('Testing_data/6_Testing_EC_Data_2024.csv')

    trainning_merged = pd.merge(trait_data, latent_genomic_data, on='Hybrid')
    trainning_merged = pd.merge(trainning_merged, training_soil_data, on='Env')
    trainning_merged = pd.merge(trainning_merged, training_weather_data, on='Env')
    trainning_merged = pd.merge(trainning_merged, EC_data, on='Env')

    return trainning_merged

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def build_feedforward_nn(input_dim):
    """Build a feedforward neural network."""
    inputs = layers.Input(shape=(input_dim,))  # Define input tensor
    reshaped_inputs = layers.Reshape((1, input_dim))(inputs)  # Reshape to (batch_size, 1, input_dim)

    # Multi-head attention layer
    attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=256)(reshaped_inputs, reshaped_inputs)
    attention_output = layers.Flatten()(attention_output)  # Flatten the output

    # Fully connected layers
    x = layers.BatchNormalization()(attention_output)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1500, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(3800, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(1750, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(850, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(1)(x)

    # Build the model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model

def train_fold(fold, train_idx, val_idx, X_train, y_train, input_dim, batch_size, epochs):
    """Train a single fold of the neural network."""
    print(f"Training fold {fold + 1}")

    # Reinitialize TensorFlow session
    tf.keras.backend.clear_session()

    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    model = build_feedforward_nn(input_dim)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

    history = model.fit(
        X_fold_train, y_fold_train,
        validation_data=(X_fold_val, y_fold_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=False,
        callbacks=[early_stopping, reduce_lr]
    )

    val_r2 = r2_score(y_fold_val, model.predict(X_fold_val).flatten())
    print(f"Fold {fold + 1} R^2: {val_r2:.2f}")
    return fold, model, val_r2


def train_nn(data, target_column, test_size=0.3, batch_size=64, epochs=500, k_folds=5, output_plot="predicted.png"):
    """Train and evaluate the feedforward neural network using parallel k-fold validation."""
    data.reset_index(drop=True, inplace=True)
    data.set_index(['Hybrid', 'Env'], inplace=True)

    def safe_convert_to_float(col):
        """Convert column to float, replacing non-convertible values with NaN."""
        return pd.to_numeric(col, errors='coerce')

    data = data.apply(safe_convert_to_float, axis=0).dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(columns=[target_column]).values)
    y = data[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    kfold = KFold(n_splits=k_folds, shuffle=True)

    results = []
    models = []

    with ThreadPoolExecutor(max_workers=k_folds) as executor:
        futures = [
            executor.submit(
                train_fold,
                fold,
                train_idx,
                val_idx,
                X_train,
                y_train,
                X.shape[1],
                batch_size,
                epochs
            )
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train))
        ]

        for future, (train_idx, val_idx) in zip(futures, kfold.split(X_train)):
            fold, model, val_r2 = future.result()
            results.append((fold, model, val_r2))
            models.append(model)

            # Save each model
            model.save(f"model_fold_{fold}.h5")

            # Generate and save observed vs predicted plot for each model
            y_val_pred = model.predict(X_train[val_idx]).flatten()
            r2 = r2_score(y_train[val_idx], y_val_pred)
            rmse = mean_squared_error(y_train[val_idx], y_val_pred, squared=False)

            plt.figure()
            plt.scatter(y_train[val_idx], y_val_pred, label=f"r^2: {r2:.2f}, RMSE: {rmse:.2f}", color='blue', s=5, alpha=0.6)
            plt.plot(
                [min(y_train[val_idx]), max(y_train[val_idx])],
                [min(y_train[val_idx]), max(y_train[val_idx])],
                color="black", linestyle="--", label="Ideal"
            )
            plt.title(f"Observed vs Predicted - Fold {fold}")
            plt.xlabel("Observed")
            plt.ylabel("Predicted")
            plt.legend()
            plt.savefig(f"predicted_model_{fold}.png")
            plt.close()

    # Predict and average outputs for all models
    y_train_preds = np.zeros_like(y_train, dtype=float)
    y_test_preds = np.zeros_like(y_test, dtype=float)

    for model in models:
        y_train_preds += model.predict(X_train).flatten()
        y_test_preds += model.predict(X_test).flatten()

    y_train_preds /= k_folds
    y_test_preds /= k_folds

    # Generate final observed vs predicted plot
    train_r2 = r2_score(y_train, y_train_preds)
    train_rmse = mean_squared_error(y_train, y_train_preds, squared=False)
    test_r2 = r2_score(y_test, y_test_preds)
    test_rmse = mean_squared_error(y_test, y_test_preds, squared=False)

    plt.figure()
    plt.scatter(y_train, y_train_preds, label=f"Train: r^2: {train_r2:.2f}, RMSE: {train_rmse:.2f}", color='blue', s=5, alpha=0.6)
    plt.scatter(y_test, y_test_preds, label=f"Test: r^2: {test_r2:.2f}, RMSE: {test_rmse:.2f}", color='red', s=5, alpha=0.6)
    plt.plot(
        [min(min(y_train), min(y_test)), max(max(y_train), max(y_test))],
        [min(min(y_train), min(y_test)), max(max(y_train), max(y_test))],
        color="black", linestyle="--", label="Ideal"
    )
    plt.title("Observed vs Predicted - All Models")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.legend()
    plt.savefig(output_plot)
    plt.close()

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def predict_yield(testing_merged, model_paths):
    """
    Predicts the yield for each entry in the testing dataset using an ensemble of models.
    
    Args:
        testing_merged (pd.DataFrame): The merged DataFrame containing the test data.
        model_paths (list of str): List of file paths to the .h5 models.
    """
    print(testing_merged)

    # Identify feature columns (columns after 'Yield_Mg_ha')
    feature_columns = testing_merged.columns[testing_merged.columns.get_loc("Yield_Mg_ha")+1:]
    features = testing_merged[feature_columns]
    print(features)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)


    # Initialize an array to store predictions from all models
    predictions = np.zeros((len(testing_merged), len(model_paths)))

    # Predict using each model and store results
    for i, model_path in enumerate(model_paths):
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        predictions[:, i] = model.predict(scaled_features).flatten()  # Flatten to handle shape

    # Average predictions across all models
    averaged_predictions = predictions.mean(axis=1)

    # Update the testing DataFrame with the averaged predictions
    testing_merged['Yield_Mg_ha'] = averaged_predictions

    # Select the required columns for the final DataFrame
    final_testing_df = testing_merged[['Env', 'Hybrid', 'Yield_Mg_ha']]

    # Save the resulting DataFrame to a CSV file
    output_path = 'final_yield_predictions.csv'
    final_testing_df.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # File path to SNP data
    genome_file_path = "/home/wimahler/Stalk_Overflow/5_Genotype_Data_All_2014_2025_Hybrids_filtered_letters.csv"


    mode = 'Training'

    if mode == 'Training':
        soil_data = process_soil_data(soil_filepath='Training_data/3_Training_Soil_Data_2015_2023.csv', metadata_filepath='Training_data/2_Training_Meta_Data_2014_2023.csv')
        weather_data = process_weather_data(weather_filepath='Training_data/4_Training_Weather_Data_2014_2023_seasons_only.csv', metadata_filepath='Training_data/2_Training_Meta_Data_2014_2023.csv')


        # Load and preprocess the data
        snp_data, index = load_genomic_data(genome_file_path)

        # Define dimensions
        input_dim = snp_data.shape[1]
        latent_dim = 1000  # Adjust based on desired dimensionality

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

        # Train and evaluate the feedforward neural network

        merged_training_df = merge_training_data()

        train_nn(merged_training_df, target_column="Yield_Mg_ha", output_plot="predicted.png")

    if mode == 'Predict':
        data = merge_test_data()

        predict_yield(data, ['model_fold_0.h5', 'model_fold_1.h5', 'model_fold_2.h5', 'model_fold_3.h5', 'model_fold_4.h5'])