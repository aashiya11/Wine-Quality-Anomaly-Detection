import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

# Loading datasets
red_df = pd.read_csv('/Users/aashiyalama/PycharmProjects/WineTesting/winequality-red.csv', sep=';')
white_df = pd.read_csv('/Users/aashiyalama/PycharmProjects/WineTesting/winequality-white.csv', sep=';')

# Defining features
features = ['alcohol', 'sulphates', 'volatile acidity', 'fixed acidity', 'density',
            'free sulfur dioxide', 'total sulfur dioxide']
target = 'quality'

# Splitting wine dataset
X_red = red_df[features]
y_red = red_df[target]
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_red, y_red, test_size=0.2, random_state=42)

X_white = white_df[features]
y_white = white_df[target]
X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(X_white, y_white, test_size=0.2, random_state=42)

# Function for model execution and visualisation
def evaluate_models(X_train, X_test, y_train, y_test, wine_type):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Z-Score anomaly detection
    z_scores = np.abs((X_test_scaled - X_test_scaled.mean(axis=0)) / X_test_scaled.std(axis=0))
    zscore_anomaly = (z_scores > 3).any(axis=1)

    # Linear Regression
    linreg = LinearRegression()
    linreg.fit(X_train_scaled, y_train)
    predictions = linreg.predict(X_test_scaled)
    residuals = np.abs(y_test - predictions)
    threshold = residuals.mean() + 2 * residuals.std()
    regression_anomaly = residuals > threshold
    r2 = r2_score(y_test, predictions)

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_train_scaled)
    iso_preds = iso_forest.predict(X_test_scaled)
    isolation_anomaly = np.where(iso_preds == -1, True, False)

    # Combine results
    results = X_test.copy()
    results['quality'] = y_test.values
    results['zscore_anomaly'] = zscore_anomaly
    results['regression_anomaly'] = regression_anomaly
    results['isolation_forest_anomaly'] = isolation_anomaly
    results['residual'] = residuals
    results['predicted_quality'] = predictions

    # Creating PDF report
    pdf = PdfPages(f'{wine_type}_model_results.pdf')

    # Boxplots for Z-Score anomalies
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=results['zscore_anomaly'], y=results[feature])
        plt.title(f'{feature} by Z-Score Anomaly - {wine_type.title()}')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Residual plot for Linear Regression
    plt.figure(figsize=(10, 6))
    sns.histplot(results['residual'], bins=50, kde=True)
    plt.axvline(threshold, color='red', linestyle='--', label='Anomaly Threshold')
    plt.title(f'Regression Residuals - {wine_type.title()}')
    plt.xlabel('Residual (|Actual - Predicted|)')
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # t-SNE Projection for Isolation Forest Anomalies
    X_test_2d = TSNE(n_components=2, random_state=42).fit_transform(X_test_scaled)
    results['tsne_x'] = X_test_2d[:, 0]
    results['tsne_y'] = X_test_2d[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='tsne_x', y='tsne_y', hue='isolation_forest_anomaly', data=results,
                    palette={0: 'gray', 1: 'red'})
    plt.title(f't-SNE Projection - Isolation Forest Anomalies ({wine_type.title()})')
    plt.xlabel('t-SNE X')
    plt.ylabel('t-SNE Y')
    plt.legend(title='Anomaly')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    pdf.close()
    print(f"PDF report saved as: {wine_type}_model_results.pdf") # Saving the result in a pdf

# Run on both red and white wine datasets
evaluate_models(X_train_red, X_test_red, y_train_red, y_test_red, 'red')
evaluate_models(X_train_white, X_test_white, y_train_white, y_test_white, 'white')




