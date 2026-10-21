import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# -------------------------------------------------------------
df = pd.read_csv("books_data.csv")

df["Price"] = df["Price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
df["Price"] = pd.to_numeric(df["Price"])

rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
df["Rating"] = df["Rating"].map(rating_map)

# -------------------------------------------------------------
# 2. EXPLORE DATA STRUCTURE & DETECT ISSUES
# -------------------------------------------------------------
print("=" * 50)
print("1. DATA STRUCTURE & INTEGRITY CHECK")
print("=" * 50)
print("Shape of Dataset (rows, cols):", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values per Column:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# -------------------------------------------------------------
# 3. STATISTICAL SUMMARY & ANOMALY DETECTION
# -------------------------------------------------------------
print("\n" + "=" * 50)
print("2. STATISTICAL METRICS & ANOMALY CHECK")
print("=" * 50)
print(df[["Price", "Rating"]].describe())

Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["Price"] < lower_bound) | (df["Price"] > upper_bound)]
print(f"\nPrice Outliers Detected (outside [{lower_bound:.2f}, {upper_bound:.2f}]): {len(outliers)}")

# -------------------------------------------------------------
# 4. HYPOTHESIS TESTING & PATTERN IDENTIFICATION
# -------------------------------------------------------------
print("\n" + "=" * 50)
print("3. PATTERNS & HYPOTHESIS VALIDATION")
print("=" * 50)

avg_price_by_rating = df.groupby("Rating")["Price"].agg(["mean", "median", "count"])
print("Price Summary Grouped by Rating:")
print(avg_price_by_rating)

correlation = df["Rating"].corr(df["Price"])
print(f"\nCorrelation between Price and Rating: {correlation:.4f}")
if abs(correlation) < 0.2:
    print("Conclusion: Weak or no linear relationship. Higher rating does not imply higher price.")
else:
    print("Conclusion: Observable linear trend between rating and price.")

# -------------------------------------------------------------
# 5. VISUALIZATION (PATTERNS, DISTRIBUTIONS & RELATIONSHIPS)
# -------------------------------------------------------------
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(11, 5.8))

try:
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
except Exception:
    pass

sns.histplot(df["Price"], kde=True, ax=axes[0, 0], color="royalblue", bins=20)
axes[0, 0].set_title("Price Distribution & Skewness", fontsize=10, fontweight="bold")
axes[0, 0].set_xlabel("Price (£)", fontsize=8, fontweight="semibold")
axes[0, 0].set_ylabel("Count", fontsize=8, fontweight="semibold")

# Subplot 2: Book Ratings Count
sns.countplot(data=df, x="Rating", ax=axes[0, 1], palette="crest", hue="Rating", legend=False)
axes[0, 1].set_title("Distribution of Book Ratings", fontsize=10, fontweight="bold")
axes[0, 1].set_xlabel("Rating (1 to 5 Stars)", fontsize=8, fontweight="semibold")
axes[0, 1].set_ylabel("Count", fontsize=8, fontweight="semibold")

sns.boxplot(data=df, x="Rating", y="Price", ax=axes[1, 0], palette="Set2", hue="Rating", legend=False)
axes[1, 0].set_title("Price Distribution Across Rating Tiers", fontsize=10, fontweight="bold")
axes[1, 0].set_xlabel("Rating", fontsize=8, fontweight="semibold")
axes[1, 0].set_ylabel("Price (£)", fontsize=8, fontweight="semibold")

df["Title_Length"] = df["Title"].astype(str).apply(len)
sns.scatterplot(data=df, x="Title_Length", y="Price", ax=axes[1, 1], alpha=0.6, color="coral")
axes[1, 1].set_title("Book Title Length vs. Price", fontsize=10, fontweight="bold")
axes[1, 1].set_xlabel("Length of Title (Characters)", fontsize=8, fontweight="semibold")
axes[1, 1].set_ylabel("Price (£)", fontsize=8, fontweight="semibold")

plt.subplots_adjust(top=0.92, bottom=0.10, left=0.07, right=0.96, hspace=0.38, wspace=0.22)

plt.savefig("eda_summary_plots.png", dpi=300)
print("\nEDA plots saved to 'eda_summary_plots.png'")
plt.show()