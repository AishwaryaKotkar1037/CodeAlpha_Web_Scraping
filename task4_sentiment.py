import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# -------------------------------------------------------------
# 1. SETUP & DOWNLOAD LEXICONS
# -------------------------------------------------------------
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# -------------------------------------------------------------
# 2. LOAD DATASET
# -------------------------------------------------------------
df = pd.read_csv("books_data.csv")

df["Price"] = df["Price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
df["Price"] = pd.to_numeric(df["Price"])

rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
df["Rating"] = df["Rating"].map(rating_map)

# -------------------------------------------------------------
# 3. NLP SENTIMENT & POLARITY EXTRACTION
# -------------------------------------------------------------
def get_vader_scores(text):
    scores = sia.polarity_scores(str(text))
    return pd.Series([scores["compound"], scores["pos"], scores["neg"], scores["neu"]])

df[["Compound", "Pos_Score", "Neg_Score", "Neu_Score"]] = df["Title"].apply(get_vader_scores)

def classify_sentiment(compound):
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Compound"].apply(classify_sentiment)

emotion_keywords = {
    "Joy / Intrigue": ["love", "magic", "wonder", "art", "secret", "happy", "sun", "dream", "star"],
    "Suspense / Fear": ["dark", "murder", "death", "blood", "kill", "shadow", "lost", "fear", "danger"],
    "Adventure / Action": ["journey", "wild", "world", "dragon", "hunt", "chronicles", "king", "war"]
}

def detect_emotion(title):
    title_lower = str(title).lower()
    for emotion, words in emotion_keywords.items():
        if any(word in title_lower for word in words):
            return emotion
    return "General / Informative"

df["Dominant_Emotion"] = df["Title"].apply(detect_emotion)

# -------------------------------------------------------------
# 4. PRINT STATISTICAL SUMMARY & MARKETING INSIGHTS
# -------------------------------------------------------------
print("=" * 60)
print("             TASK 4: SENTIMENT ANALYSIS REPORT")
print("=" * 60)

sentiment_counts = df["Sentiment"].value_counts()
print("\n--- 1. Sentiment Polarity Breakdown ---")
print(sentiment_counts)
print(f"\nPositive Share: {(sentiment_counts.get('Positive', 0)/len(df))*100:.1f}%")
print(f"Neutral Share:  {(sentiment_counts.get('Neutral', 0)/len(df))*100:.1f}%")
print(f"Negative Share: {(sentiment_counts.get('Negative', 0)/len(df))*100:.1f}%")

print("\n--- 2. Emotion Category Distribution ---")
print(df["Dominant_Emotion"].value_counts())

print("\n--- 3. Average Star Rating by Sentiment Class ---")
rating_by_sentiment = df.groupby("Sentiment")["Rating"].mean().round(2)
print(rating_by_sentiment)

# -------------------------------------------------------------
# 5. VISUALIZATIONS (DASHBOARD)
# -------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig = plt.figure(figsize=(12, 6.5))
fig.patch.set_facecolor("#F8F9FA")

try:
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
except Exception:
    pass

gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25, top=0.88, bottom=0.12, left=0.08, right=0.95)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

fig.suptitle(
    "SENTIMENT & EMOTIONAL INTELLIGENCE REPORT\n"
    f"Catalog Size: {len(df)} Records | NLP Lexicon: NLTK VADER",
    fontsize=12,
    fontweight="bold",
    color="#1E293B"
)

colors_sent = ["#22C55E", "#64748B", "#EF4444"]
order = ["Positive", "Neutral", "Negative"]
present_order = [o for o in order if o in sentiment_counts.index]
ax1.pie(
    [sentiment_counts[o] for o in present_order],
    labels=present_order,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.75,
    colors=[colors_sent[order.index(o)] for o in present_order],
    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
)
ax1.set_title("1. Sentiment Polarity Breakdown", fontsize=10, fontweight="bold", pad=8)

emotion_counts = df["Dominant_Emotion"].value_counts().sort_values(ascending=True)
ax2.barh(emotion_counts.index, emotion_counts.values, color="#3B82F6", edgecolor="#1D4ED8", height=0.55)
ax2.set_title("2. Dominant Emotions Detected", fontsize=10, fontweight="bold")
ax2.set_xlabel("Number of Books", fontsize=8, fontweight="semibold")

sns.scatterplot(
    data=df,
    x="Compound",
    y="Price",
    hue="Sentiment",
    palette={"Positive": "#22C55E", "Neutral": "#64748B", "Negative": "#EF4444"},
    alpha=0.7,
    ax=ax3
)
ax3.axvline(0, color="gray", linestyle="--", linewidth=1)
ax3.set_title("3. Sentiment Polarity vs. Retail Price (£)", fontsize=10, fontweight="bold")
ax3.set_xlabel("Compound Score (-1 Negative to +1 Positive)", fontsize=8, fontweight="semibold")
ax3.set_ylabel("Price (£)", fontsize=8, fontweight="semibold")
ax3.legend(loc="upper left", fontsize=7)

all_titles = " ".join(df["Title"].dropna())
wordcloud = WordCloud(
    width=400,
    height=200,
    background_color="white",
    colormap="Blues",
    max_words=50
).generate(all_titles)
ax4.imshow(wordcloud, interpolation="bilinear")
ax4.axis("off")
ax4.set_title("4. High-Frequency Lexical Keywords", fontsize=10, fontweight="bold")

plt.savefig("task4_sentiment_dashboard.png", dpi=300)
print("\nDashboard saved as 'task4_sentiment_dashboard.png'!")
plt.show()

df.to_csv("sentiment_analyzed_books.csv", index=False)
print("Sentiment data exported to 'sentiment_analyzed_books.csv'.")