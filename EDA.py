import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# Loading Datasets
red_df = pd.read_csv('/Users/aashiyalama/PycharmProjects/WineTesting/winequality-red.csv', sep=';')
white_df = pd.read_csv('/Users/aashiyalama/PycharmProjects/WineTesting/winequality-white.csv', sep=';')

red_df['type'] = 'red'
white_df['type'] = 'white'

df = pd.concat([red_df, white_df], axis=0) # Combine for full EDA

# Data Overview
print("Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Stats:\n", df.describe())

# Data Cleaning
duplicates = df.duplicated().sum()
print("\nDuplicate rows:", duplicates)
if duplicates:
    df = df.drop_duplicates()

# Exploratory Data Analysis
sns.set(style="whitegrid")
output_dir = '/Users/aashiyalama/PycharmProjects/WineTesting/EDA results'
os.makedirs(output_dir, exist_ok=True)

pdf_path = os.path.join(output_dir, 'wine_eda_report.pdf')
pdf = PdfPages(pdf_path)

# Features for box plot & pair plot
features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]
selected = ['alcohol', 'sulphates', 'residual sugar', 'citric acid', 'density', 'quality']

wine_palette = {'red': 'crimson', 'white': 'lightblue'} # Color palettes

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.drop(columns='type').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (All Features)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
pdf.savefig()
plt.close()
print("✅ Saved: correlation_heatmap.png")

# Box plots for all features
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='type', y=feature, hue='type', palette=wine_palette, dodge=False)
    plt.title(f"{feature} Comparison - Red vs White")
    plt.xlabel("Wine Type")
    plt.ylabel(feature)
    plt.tight_layout()
    filename = f'boxplot_{feature.replace(" ", "_")}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    pdf.savefig()
    plt.close()
    print(f"✅ Saved: {filename}")

# Pair plot of Selected Features
pairplot = sns.pairplot(df[selected + ['type']], hue='type', palette=wine_palette, height=1.8, corner=True)
pairplot.fig.suptitle("Pairplot of Key Features by Wine Type", y=2)
pairplot.savefig(os.path.join(output_dir, 'pairplot_selected_features.png'), dpi=300)
pdf.savefig()
plt.close()
print("✅ Saved: pairplot_selected_features.png")

# Quality Comparison Bar Plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='quality', hue='type', palette=wine_palette)
plt.title("Wine Quality Comparison: Red vs White")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.legend(title="Wine Type")
plt.tight_layout()

# Save as image
quality_bar_path = os.path.join(output_dir, 'quality_comparison_barplot.png')
plt.savefig(quality_bar_path, dpi=300)
pdf.savefig()  # Add to PDF
plt.close()
print(f"✅ Saved: {quality_bar_path}")

# Average Feature Values by Wine Quality
avg_features = ['alcohol', 'sulphates', 'residual sugar', 'citric acid', 'pH']

avg_by_quality = df.groupby('quality')[avg_features].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
for feature in avg_features:
    plt.plot(avg_by_quality['quality'], avg_by_quality[feature], marker='o', label=feature)

plt.title("Average Feature Values by Wine Quality")
plt.xlabel("Wine Quality Score")
plt.ylabel("Average Value")
plt.legend(title="Feature")
plt.grid(True)
plt.tight_layout()

# Save to file and PDF
avg_feat_path = os.path.join(output_dir, 'avg_feature_values_by_quality.png')
plt.savefig(avg_feat_path, dpi=300)
pdf.savefig()
plt.close()
print(f"✅ Saved: {avg_feat_path}")

pdf.close()
