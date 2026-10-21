import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ------------------------------------------------------------------
df = pd.read_csv("books_data.csv")

df["Price"] = df["Price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
df["Price"] = pd.to_numeric(df["Price"])

rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
df["Rating"] = df["Rating"].map(rating_map)

df["Price_Tier"] = pd.cut(
    df["Price"],
    bins=[0, 20, 40, np.inf],
    labels=["Budget (£0-£20)", "Mid-Tier (£20-£40)", "Premium (£40+)"]
)

# ------------------------------------------------------------------
# 2. DASHBOARD CANVAS SETUP (Fits all laptop screens & DPI scales)
# ------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

fig = plt.figure(figsize=(11, 5.8))
fig.patch.set_facecolor("#F8F9FA")

try:
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
except Exception:
    pass

gs = fig.add_gridspec(
    2, 2,
    hspace=0.45,
    wspace=0.22,
    top=0.86,
    bottom=0.12,
    left=0.08,
    right=0.95
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

fig.suptitle(
    "BOOK CATALOG MARKET INTELLIGENCE DASHBOARD\n"
    f"Analyzed Catalog: {len(df)} Titles | Avg Price: £{df['Price'].mean():.2f} | Avg Rating: {df['Rating'].mean():.2f} Stars",
    fontsize=12,
    fontweight="bold",
    color="#1E293B"
)

# ------------------------------------------------------------------
# VISUAL 1: Price Tier Breakdown (Donut Chart)
# ------------------------------------------------------------------
tier_counts = df["Price_Tier"].value_counts(sort=False)
colors_tier = ["#38BDF8", "#3B82F6", "#1E40AF"]

wedges, texts, autotexts = ax1.pie(
    tier_counts,
    labels=tier_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.75,
    colors=colors_tier,
    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
)
for at in autotexts:
    at.set_color("white")
    at.set_weight("bold")
    at.set_fontsize(9)
ax1.set_title("1. Inventory Share by Price Tier", fontsize=10, fontweight="bold", pad=6)

# ------------------------------------------------------------------
# VISUAL 2: Price Dispersion Across Rating Tiers (Violin Plot)
# ------------------------------------------------------------------
sns.violinplot(
    data=df,
    x="Rating",
    y="Price",
    hue="Rating",
    legend=False,
    ax=ax2,
    palette="Blues",
    inner="quartile",
    cut=0
)
ax2.set_title("2. Price Dispersion Across Rating Tiers", fontsize=10, fontweight="bold")
ax2.set_xlabel("Star Rating (1 to 5 Stars)", fontsize=8, fontweight="semibold")
ax2.set_ylabel("Price (£)", fontsize=8, fontweight="semibold")
ax2.axhline(
    df["Price"].mean(),
    color="red",
    linestyle="--",
    linewidth=1.2,
    label=f"Mean (£{df['Price'].mean():.2f})"
)
ax2.legend(loc="upper right", frameon=True, fontsize=7)

# ------------------------------------------------------------------
# VISUAL 3: Price Distribution & Density
# ------------------------------------------------------------------
sns.histplot(
    df["Price"],
    kde=True,
    ax=ax3,
    color="#0284C7",
    bins=25,
    alpha=0.6,
    line_kws={"linewidth": 1.8}
)
ax3.set_title("3. Price Frequency & Density Curve", fontsize=10, fontweight="bold")
ax3.set_xlabel("Price (£)", fontsize=8, fontweight="semibold")
ax3.set_ylabel("Total Book Count", fontsize=8, fontweight="semibold")
median_val = df["Price"].median()
ax3.axvline(
    median_val,
    color="#0F172A",
    linestyle=":",
    linewidth=1.4,
    label=f"Median (£{median_val:.2f})"
)
ax3.legend(frameon=True, fontsize=7)

# ------------------------------------------------------------------
# VISUAL 4: Rating Volume Share (Horizontal Bar)
# ------------------------------------------------------------------
rating_pct = (df["Rating"].value_counts(normalize=True).sort_index() * 100).reset_index()
rating_pct.columns = ["Rating", "Percentage"]

bars = ax4.barh(
    rating_pct["Rating"].astype(str) + " Star",
    rating_pct["Percentage"],
    color="#60A5FA",
    edgecolor="#1D4ED8",
    height=0.55
)
ax4.set_title("4. Customer Sentiment / Rating Volume Share (%)", fontsize=10, fontweight="bold")
ax4.set_xlabel("Percentage of Total Catalog (%)", fontsize=8, fontweight="semibold")
ax4.set_xlim(0, max(rating_pct["Percentage"]) + 7)

for bar in bars:
    w = bar.get_width()
    ax4.text(
        w + 0.6,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.1f}%",
        va="center",
        fontweight="bold",
        fontsize=8,
        color="#1E293B"
    )

# ------------------------------------------------------------------
# 3. SAVE HIGH-RES IMAGE & DISPLAY
# ------------------------------------------------------------------
plt.savefig("task3_executive_dashboard.png", dpi=300)
print("Executive dashboard saved successfully as 'task3_executive_dashboard.png'!")
plt.show()