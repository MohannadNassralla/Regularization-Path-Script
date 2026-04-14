import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def run_regularization_explorer():
    # 1. Load Dataset
    try:
        # Ensure the filename matches your actual CSV file
        df = pd.read_csv('data/telecom_churn.csv')
        print("✅ Dataset loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: 'telecom_churn.csv' not found. Please check the file path.")
        return

    # 2. Data Preprocessing
    # Convert total_charges to numeric and handle potential missing values
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    df = df.fillna(0)

    # Encode categorical variables (One-Hot Encoding)
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Identify Target and Features
    # This dynamically finds the 'churned' column you identified
    target_col = [col for col in df_encoded.columns if 'churned' in col][0]
    
    # Drop target and non-predictive identifiers
    X = df_encoded.drop(columns=[target_col])
    if 'customer_id' in X.columns:
        X = X.drop(columns=['customer_id'])
    
    y = df_encoded[target_col]
    feature_names = X.columns

    # 3. Standardize Features (CRITICAL for regularization comparison)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Generate Regularization Path Data
    # 20 values of C logarithmically spaced from 0.001 to 100
    C_values = np.logspace(-3, 2, 20)
    penalties = ['l1', 'l2']
    coef_dict = {p: [] for p in penalties}

    print("🔄 Training models and recording coefficient trajectories...")

    for p in penalties:
        for c in C_values:
            # Using 'liblinear' solver as it is robust for L1/L2 on smaller datasets
            model = LogisticRegression(penalty=p, C=c, solver='liblinear', max_iter=1000)
            model.fit(X_scaled, y)
            coef_dict[p].append(model.coef_[0])

    # 5. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for i, p in enumerate(penalties):
        coefs = np.array(coef_dict[p])
        for j in range(coefs.shape[1]):
            axes[i].plot(C_values, coefs[:, j], label=feature_names[j] if i == 0 else "")
        
        axes[i].set_xscale('log')
        axes[i].set_title(f'Regularization Path: {p.upper()}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('C (Inverse Regularization Strength)', fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.6)

    axes[0].set_ylabel('Coefficient Magnitude', fontsize=12)
    
    # Place legend to the right
    fig.legend(loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize='small')
    
    plt.suptitle('Effect of Regularization on Model Coefficients', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot for the GitHub repository
    plt.savefig('regularization_path_results.png', bbox_inches='tight')
    print("🚀 Success! Plot saved as 'regularization_path_results.png'.")
    plt.show()

if __name__ == "__main__":
    run_regularization_explorer()