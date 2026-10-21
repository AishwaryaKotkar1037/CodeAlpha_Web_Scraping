import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("books_data.csv")

print("========================================")
print("       BOOK DATA ANALYSIS")
print("========================================")

# ----------------------------------------
# DATA CLEANING
# ----------------------------------------

df["Price"] = df["Price"].astype(str).str.replace(r"[^\d.]", "", regex=True)

df["Price"] = pd.to_numeric(df["Price"])

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["Rating"] = df["Rating"].map(rating_map)

# ----------------------------------------
# BASIC INFORMATION
# ----------------------------------------

print("\nTotal number of books:", len(df))

print("\nFirst 5 books:")
print(df.head())

print("\nDataset information:")
print(df.info())

# ----------------------------------------
# PRICE ANALYSIS
# ----------------------------------------

average_price = df["Price"].mean()
minimum_price = df["Price"].min()
maximum_price = df["Price"].max()

print("\nAverage Price:", round(average_price, 2))
print("Minimum Price:", minimum_price)
print("Maximum Price:", maximum_price)

# ----------------------------------------
# RATING ANALYSIS
# ----------------------------------------

average_rating = df["Rating"].mean()

print("\nAverage Rating:", round(average_rating, 2))

print("\nNumber of books by rating:")
print(df["Rating"].value_counts().sort_index())

# ----------------------------------------
# MOST EXPENSIVE BOOK
# ----------------------------------------

expensive_book = df.loc[df["Price"].idxmax()]

print("\nMost Expensive Book:")
print(expensive_book["Title"])
print("Price:", expensive_book["Price"])

# ----------------------------------------
# CHEAPEST BOOK
# ----------------------------------------

cheap_book = df.loc[df["Price"].idxmin()]

print("\nCheapest Book:")
print(cheap_book["Title"])
print("Price:", cheap_book["Price"])

# ----------------------------------------
# GRAPH 1 - RATING DISTRIBUTION
# ----------------------------------------

rating_counts = df["Rating"].value_counts().sort_index()

rating_counts.plot(kind="bar")

plt.title("Distribution of Book Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Books")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("rating_distribution.png")
plt.show()

# ----------------------------------------
# GRAPH 2 - PRICE DISTRIBUTION
# ----------------------------------------

df["Price"].plot(
    kind="hist",
    bins=20
)

plt.title("Distribution of Book Prices")
plt.xlabel("Price (£)")
plt.ylabel("Number of Books")
plt.tight_layout()

plt.savefig("price_distribution.png")
plt.show()

print("\nAnalysis completed successfully!")