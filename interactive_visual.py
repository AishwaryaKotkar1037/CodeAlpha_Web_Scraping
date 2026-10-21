import pandas as pd
import plotly.express as px

df = pd.read_csv("books_data.csv")

df["Price"] = df["Price"].astype(str).str.replace(r"[^\d.]", "", regex=True).astype(float)

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}
df["Rating"] = df["Rating"].map(rating_map)

df["Rating_Tier"] = df["Rating"].astype(str) + " Star"

fig = px.strip(
    df,
    x="Rating",
    y="Price",
    hover_name="Title",
    color="Rating_Tier",
    color_discrete_sequence=px.colors.qualitative.Prism,
    title="Interactive Book Catalog: Price vs. Rating Explorer",
    labels={
        "Price": "Price (£)",
        "Rating": "Star Rating",
        "Rating_Tier": "Rating Tier"
    }
)

fig.update_layout(
    template="plotly_white",
    title_font_size=18,
    xaxis=dict(tickmode="linear", tick0=1, dtick=1)
)

fig.write_html("interactive_book_analysis.html")
print("Interactive chart saved: 'interactive_book_analysis.html'")

fig.show()