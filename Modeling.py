import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Function for per-type anomaly detection & evaluation
def process_wine_type(df, wine_type, pdf):
    df_type = df[df["type"] == wine_type].copy()
    df_type["true_anomaly"] = (df_type["quality"] >= np.percentile(df_type["quality"], 95)).astype(int)

    X = df_type.drop(columns=["wine_name", "type", "quality", "true_anomaly"])
    y = df_type["quality"]
    y_true = df_type["true_anomaly"]
    wine_names = df_type["wine_name"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_results = pd.DataFrame()
    df_results["wine_name"] = wine_names
    df_results["true_anomaly"] = y_true

    # --- Z-Score ---
    z_scores = np.abs((X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0))
    df_results["anomaly_zscore"] = (z_scores > 3).any(axis=1).astype(int)

    # --- Linear Regression ---
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    predicted_quality = lr_model.predict(X_scaled)
    lr_error = np.abs(y - predicted_quality)
    lr_threshold = np.percentile(lr_error, 95)
    df_results["linreg_error"] = lr_error
    df_results["anomaly_linreg"] = (lr_error > lr_threshold).astype(int)

    # --- LOF ---
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_pred = lof.fit_predict(X_scaled)
    df_results["anomaly_lof"] = np.where(lof_pred == -1, 1, 0)

    # --- Total Flags ---
    df_results["total_flags"] = df_results[["anomaly_zscore", "anomaly_linreg", "anomaly_lof"]].sum(axis=1)

    # --- Evaluation Metrics (LOF) ---
    precision = precision_score(df_results["true_anomaly"], df_results["anomaly_lof"])
    recall = recall_score(df_results["true_anomaly"], df_results["anomaly_lof"])
    f1 = f1_score(df_results["true_anomaly"], df_results["anomaly_lof"])
    roc_auc = roc_auc_score(df_results["true_anomaly"], df_results["anomaly_lof"])

    # --- Plot 1: % Anomalies ---
    methods = ["Z-Score", "Linear Regression", "Local Outlier Factor"]
    flags = [df_results["anomaly_zscore"], df_results["anomaly_linreg"], df_results["anomaly_lof"]]
    percentages = [(flag.sum() / len(df_results)) * 100 for flag in flags]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=methods, y=percentages, palette="Set2")
    plt.title(f"{wine_type.capitalize()} Wine: Anomalies Detected (%)")
    plt.ylabel("Percentage")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # --- Plot 2: Flag Count ---
    plt.figure(figsize=(8, 5))
    sns.countplot(x="total_flags", data=df_results, palette="pastel")
    plt.title(f"{wine_type.capitalize()} Wine: Total Flag Count per Sample")
    plt.xlabel("Number of Methods that Flagged")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # --- Plot 3: PCA with LOF ---
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    df_results["PCA1"] = pca_data[:, 0]
    df_results["PCA2"] = pca_data[:, 1]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_results, x="PCA1", y="PCA2", hue="anomaly_lof", palette={0: "green", 1: "red"}, alpha=0.6)
    plt.title(f"{wine_type.capitalize()} Wine: LOF Anomalies (PCA Projection)")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # --- Plot 4: t-SNE with LOF ---
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_data = tsne.fit_transform(X_scaled)
    df_results["TSNE1"] = tsne_data[:, 0]
    df_results["TSNE2"] = tsne_data[:, 1]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_results, x="TSNE1", y="TSNE2", hue="anomaly_lof", palette={0: "blue", 1: "red"}, alpha=0.6)
    plt.title(f"{wine_type.capitalize()} Wine: LOF Anomalies (t-SNE Projection)")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # --- Plot 5: Linear Regression Error ---
    plt.figure(figsize=(10, 4))
    plt.hist(lr_error, bins=50, color="orange", edgecolor="black")
    plt.axvline(lr_threshold, color="red", linestyle="--", label="95th Percentile")
    plt.title(f"{wine_type.capitalize()} Wine: Linear Regression Error")
    plt.xlabel("Absolute Error")
    plt.ylabel("Wine Count")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # --- Plot 6: Metrics Summary ---
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics_text = f"""Evaluation Metrics (LOF)
-----------------------------
Precision  : {precision:.3f}
Recall     : {recall:.3f}
F1 Score   : {f1:.3f}
ROC-AUC    : {roc_auc:.3f}"""
    ax.text(0.01, 0.8, metrics_text, fontsize=12, va='top', family='monospace')
    ax.axis('off')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    return df_results


# ========= MAIN SCRIPT =========

# Load the cleaned data
df = pd.read_csv('/Users/aashiyalama/PycharmProjects/WineTesting/venv/combined_wine_cleaned.csv')

# Add wine_name column if missing
df['wine_name'] = df.get('wine_name', [f"wine_{i}" for i in range(len(df))])

# Output PDF path
pdf = PdfPages('/Users/aashiyalama/PycharmProjects/WineTesting/EDA results/wine_anomaly_results.pdf')

# Run for both red and white
results_red = process_wine_type(df, "red", pdf)
results_white = process_wine_type(df, "white", pdf)

pdf.close()
print("✅ All visualizations and evaluation metrics saved in: wine_anomaly_results.pdf")
