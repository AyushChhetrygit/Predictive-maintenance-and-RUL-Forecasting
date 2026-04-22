import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from sklearn.metrics import classification_report
import os

def main():
    data_path = '../data/features.pkl'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Generating mock data for testing purposes...")
        np.random.seed(42)
        cycles = np.arange(1, 101)
        # 3 engines
        df_list = []
        for i in range(1, 4):
            engine_df = pd.DataFrame({
                'engine_id': i,
                'cycle': cycles,
                'sensor_1': np.random.normal(0, 1, 100) + (cycles/100)*2
            })
            df_list.append(engine_df)
        df = pd.concat(df_list, ignore_index=True)
        # Create data dir
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_pickle(data_path)

    # STEP 3 — Load Your Data
    df = pd.read_pickle(data_path)
    print("Features loaded successfully.")
    print(df.head())

    # STEP 4 — Define NORMAL DATA
    df['RUL'] = df.groupby('engine_id')['cycle'].transform(lambda x: x.max() - x)

    # Normal = high RUL
    normal_df = df[df['RUL'] > 50]

    # Test = full dataset
    test_df = df.copy()

    # STEP 5 — Select Features
    features = [col for col in df.columns if 'sensor' in col]

    X_train = normal_df[features].fillna(0)
    X_test = test_df[features].fillna(0)

    # STEP 6 — Train Isolation Forest (PyOD)
    print("Training Isolation Forest...")
    iforest = IForest(contamination=0.05)
    iforest.fit(X_train)

    y_pred_iforest = iforest.predict(X_test)

    # STEP 7 — Train Local Outlier Factor (PyOD)
    print("Training Local Outlier Factor...")
    lof = LOF(n_neighbors=20, contamination=0.05)
    lof.fit(X_train)

    y_pred_lof = lof.predict(X_test)

    # STEP 8 — Combine Predictions 
    final_pred = np.logical_or(y_pred_iforest, y_pred_lof).astype(int)

    df['anomaly'] = final_pred

    # STEP 9 — Evaluate 
    df['true_anomaly'] = (df['RUL'] < 20).astype(int)

    print("\nClassification Report:")
    print(classification_report(df['true_anomaly'], df['anomaly']))

    # STEP 10 — Visualize
    sensor_col = 'sensor_1' if 'sensor_1' in df.columns else features[0]
    
    plt.figure()
    plt.plot(df['cycle'], df[sensor_col], label=sensor_col)
    plt.scatter(df['cycle'], df[sensor_col], 
                c=df['anomaly'], cmap='coolwarm', label='anomaly')
    plt.legend()
    
    # Save the plot explicitly instead of just plt.show() which might block
    os.makedirs('../models', exist_ok=True)
    plot_path = '../models/anomaly_plot.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Optional display (commented out to prevent blocking in automated environments)
    # try:
    #     plt.show()
    # except Exception as e:
    #     print(f"Could not display plot interactively: {e}")

    # STEP 11 — Save Model
    joblib.dump(iforest, '../models/iforest.pkl')
    joblib.dump(lof, '../models/lof.pkl')
    print("Models saved successfully in ../models/")

if __name__ == "__main__":
    main()
