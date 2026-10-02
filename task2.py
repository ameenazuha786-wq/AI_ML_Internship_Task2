import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)
# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Mean of numerical columns
print("\nMean:")
print(df.mean(numeric_only=True))

# Median of numerical columns
print("\nMedian:")
print(df.median(numeric_only=True))

# Standard deviation
print("\nStandard Deviation:")
print(df.std(numeric_only=True))
# Histograms for numerical columns
import matplotlib.pyplot as plt

df.hist(figsize=(12, 10))

plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()
# Boxplots for numerical columns
plt.figure(figsize=(12, 8))

df.boxplot()

plt.title("Boxplots of Numerical Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Correlation matrix
plt.figure(figsize=(10, 7))

correlation = df.corr(numeric_only=True)

plt.imshow(correlation, cmap="coolwarm", interpolation="none")
plt.colorbar()

plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation.columns)), correlation.columns)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()