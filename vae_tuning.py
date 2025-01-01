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
import optuna

def load_data(file_path):
    """Load and preprocess SNP data."""
    data = pd.read_csv(file_path, sep='\t', index_col=0)
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

def build_encoder(input_dim, latent_dim):
    """Build the encoder part of the VAE."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(1028, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)  # Add Batch Normalization after the first dense layer
    x = layers.Dense(2048, activation="relu")(x)
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

def train_vae(vae, data, batch_size=32, epochs=50):
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

    trainning_merged = pd.merge(trait_data, latent_genomic_data, on='Hybrid')
    trainning_merged = pd.merge(trainning_merged, training_soil_data, on='Env')
    trainning_merged = pd.merge(trainning_merged, training_weather_data, on='Env')

    return trainning_merged

def build_feedforward_nn(input_dim):
    """Build a feedforward neural network."""
    inputs = layers.Input(shape=(input_dim,))  # Define input tensor
    reshaped_inputs = layers.Reshape((1, input_dim))(inputs)  # Reshape to (batch_size, 1, input_dim)

    # Multi-head attention layer
    attention_output = layers.MultiHeadAttention(num_heads=32, key_dim=128)(reshaped_inputs, reshaped_inputs)
    attention_output = layers.Flatten()(attention_output)  # Flatten the output

    # Fully connected layers
    x = layers.BatchNormalization()(attention_output)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(4000*2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2000*2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1000*2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(500*2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)

    # Build the model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model

def optimize_vae(trial):
    """Optimize VAE hyperparameters."""
    input_dim = snp_data.shape[1]

    # Suggest hyperparameters
    latent_dim = trial.suggest_int("latent_dim", 100, 1000, step=50)
    encoder_hidden_dim1 = trial.suggest_int("encoder_hidden_dim1", 512, 2048, step=128)
    encoder_hidden_dim2 = trial.suggest_int("encoder_hidden_dim2", 512, 2048, step=128)
    decoder_hidden_dim1 = trial.suggest_int("decoder_hidden_dim1", 512, 2048, step=128)
    decoder_hidden_dim2 = trial.suggest_int("decoder_hidden_dim2", 512, 2048, step=128)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])

    # Build encoder
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(encoder_hidden_dim1, activation=activation)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(encoder_hidden_dim2, activation=activation)(x)
    x = layers.BatchNormalization()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # Build decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(decoder_hidden_dim1, activation=activation)(latent_inputs)
    x = layers.Dense(decoder_hidden_dim2, activation=activation)(x)
    outputs = layers.Dense(input_dim, activation="sigmoid")(x)
    decoder = models.Model(latent_inputs, outputs, name="decoder")

    # Build VAE
    reconstructed = decoder(encoder(inputs)[2])

    def vae_loss(y_true, y_pred):
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss

    vae = models.Model(inputs, reconstructed, name="vae")
    vae.add_loss(vae_loss(inputs, reconstructed))

    # Train VAE
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    history = vae.fit(snp_data, snp_data, batch_size=32, epochs=10, verbose=0)

    return history.history['loss'][-1]

def optimize_nn(trial):
    """Optimize Feedforward NN hyperparameters."""
    # Suggest hyperparameters
    num_heads = trial.suggest_int("num_heads", 8, 64, step=8)
    key_dim = trial.suggest_int("key_dim", 64, 256, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    layer1_units = trial.suggest_int("layer1_units", 1000, 8000, step=250)
    layer2_units = trial.suggest_int("layer2_units", 200, 4000, step=200)
    layer3_units = trial.suggest_int("layer3_units", 100, 2000, step=150)
    layer4_units = trial.suggest_int("layer4_units", 50, 1000, step=100)

    # Prepare the data
    data = merged_training_df.drop(columns=["Yield_Mg_ha"])
    data.reset_index(drop=True, inplace=True)
    data.set_index(['Hybrid', 'Env'], inplace=True)
    target_column = merged_training_df["Yield_Mg_ha"]

    X_train, X_test, y_train, y_test = train_test_split(data, target_column, test_size=0.3)

    # Build the model
    inputs = layers.Input(shape=(X_train.shape[1],))
    reshaped_inputs = layers.Reshape((1, X_train.shape[1]))(inputs)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(reshaped_inputs, reshaped_inputs)
    attention_output = layers.Flatten()(attention_output)

    # Fully connected layers
    x = layers.BatchNormalization()(attention_output)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(layer1_units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(layer2_units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(layer3_units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(layer4_units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    # Compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=50,
        verbose=0,
        callbacks=[early_stopping, reduce_lr]
    )

    return history.history['val_loss'][-1]

if __name__ == "__main__":
    # File path to SNP data
    file_path = "/home/wimahler/Stalk_Overflow/5_Genotype_Data_All_2014_2025_Hybrids_filtered_letters.csv"

    # Load and preprocess the data
    snp_data, index = load_data(file_path)

    # Merge training data
    merged_training_df = merge_training_data()

    # Optimize VAE
    # study_vae = optuna.create_study(direction="minimize")
    # study_vae.optimize(optimize_vae, n_trials=10)
    # print("Best VAE hyperparameters:", study_vae.best_params)

    # Optimize Feedforward NN
    study_nn = optuna.create_study(direction="minimize")
    study_nn.optimize(optimize_nn, n_trials=15)
    print("Best Feedforward NN hyperparameters:", study_nn.best_params)
