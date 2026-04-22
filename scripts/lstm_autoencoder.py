import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os

def main():
    # STEP 3 — Load Data
    data_path = '../data/features.pkl'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Generating mock data for testing purposes...")
        np.random.seed(42)
        cycles = np.arange(1, 101)
        df_list = []
        for i in range(1, 4):
            engine_df = pd.DataFrame({
                'engine_id': i,
                'cycle': cycles,
                'sensor_1': np.random.normal(0, 1, 100) + (cycles/100)*2
            })
            df_list.append(engine_df)
        df = pd.concat(df_list, ignore_index=True)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_pickle(data_path)
    else:
        df = pd.read_pickle(data_path)
        print("Data loaded successfully.")

    # Create RUL
    if 'RUL' not in df.columns:
        df['RUL'] = df.groupby('engine_id')['cycle'].transform(lambda x: x.max() - x)

    # Normal data only
    normal_df = df[df['RUL'] > 50].copy()
    
    # Full data for testing
    test_df = df.copy()

    # STEP 4 — Select Features + Scale
    features = [col for col in df.columns if 'sensor' in col]
    scaler = StandardScaler()

    normal_df[features] = scaler.fit_transform(normal_df[features].fillna(0))
    test_df[features] = scaler.transform(test_df[features].fillna(0))

    # STEP 5 — Create Sequences 
    # Adjusted to prevent sequences from crossing engine boundaries
    def create_sequences_by_engine(data, feature_cols, seq_len=30):
        sequences = []
        indices = []
        for engine_id, group in data.groupby('engine_id'):
            feat_data = group[feature_cols].values
            idx_data = group.index.values
            if len(feat_data) > seq_len:
                for i in range(len(feat_data) - seq_len):
                    sequences.append(feat_data[i:i+seq_len])
                    # The index corresponding to the prediction at the end of the sequence
                    indices.append(idx_data[i+seq_len-1])
        return np.array(sequences), np.array(indices)

    SEQ_LEN = 30
    X_train_seq, train_indices = create_sequences_by_engine(normal_df, features, SEQ_LEN)
    X_test_seq, test_indices = create_sequences_by_engine(test_df, features, SEQ_LEN)

    # STEP 6 — Convert to PyTorch
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)

    # STEP 7 — Build LSTM Autoencoder
    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_features, hidden_dim=64):
            super().__init__()
            self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim, n_features, batch_first=True)

        def forward(self, x):
            encoded, _ = self.encoder(x)
            decoded, _ = self.decoder(encoded)
            return decoded

    # STEP 8 — Initialize Model
    n_features = X_train_seq.shape[2]
    model = LSTMAutoencoder(n_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # STEP 9 — Train Model
    EPOCHS = 10
    print("\nTraining LSTM Autoencoder...")
    for epoch in range(EPOCHS):
        model.train()
        output = model(X_train_tensor)
        loss = criterion(output, X_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    # STEP 10 — Reconstruction Error
    model.eval()
    with torch.no_grad():
        recon = model(X_test_tensor)

    errors = torch.mean((recon - X_test_tensor) ** 2, dim=(1,2)).numpy()

    # STEP 11 — Thresholding
    threshold = np.percentile(errors, 95)
    anomalies_lstm = (errors > threshold).astype(int)

    # STEP 12 — Statistical Baselines
    X_test_scaled = test_df[features].values
    
    # Z-score
    z_scores = np.abs((X_test_scaled - X_test_scaled.mean(axis=0)) / (X_test_scaled.std(axis=0) + 1e-6))
    z_anomaly = (z_scores > 3).any(axis=1).astype(int)
    
    # MAD
    median = np.median(X_test_scaled, axis=0)
    mad = np.median(np.abs(X_test_scaled - median), axis=0)
    mad_score = np.abs(X_test_scaled - median) / (mad + 1e-6)
    mad_anomaly = (mad_score > 3).any(axis=1).astype(int)

    # STEP 13 — Align Labels
    test_df['lstm_anomaly'] = 0
    test_df.loc[test_indices, 'lstm_anomaly'] = anomalies_lstm
    
    test_df['z_anomaly'] = z_anomaly
    test_df['mad_anomaly'] = mad_anomaly

    # STEP 14 — Compare with RUL
    test_df['true_anomaly'] = (test_df['RUL'] < 20).astype(int)

    # Filter evaluation data to only include indices where we could run LSTM (due to seq length)
    eval_df = test_df.loc[test_indices]

    # STEP 15 — Evaluation
    print("\n--- Evaluation on Valid Sequences ---")
    print("LSTM Autoencoder:")
    print(classification_report(eval_df['true_anomaly'], eval_df['lstm_anomaly'], zero_division=0))

    print("Z-score:")
    print(classification_report(eval_df['true_anomaly'], eval_df['z_anomaly'], zero_division=0))

    print("MAD (Median Absolute Deviation):")
    print(classification_report(eval_df['true_anomaly'], eval_df['mad_anomaly'], zero_division=0))

    # STEP 16 — Save Model
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/lstm_autoencoder.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()
