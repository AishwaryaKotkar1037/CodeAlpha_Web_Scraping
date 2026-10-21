import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin

base_url = "https://books.toscrape.com/"

data = []

print("========================================")
print("   CODEALPHA WEB SCRAPING PROJECT")
print("========================================")
print()

# Scrape 50 pages
for page in range(1, 51):

    # Create page URL
    url = f"https://books.toscrape.com/catalogue/page-{page}.html"

    print("Scraping page", page, "...")

    try:
        # Send request to website
        response = requests.get(url, timeout=10)

        # Check whether webpage was successfully opened
        if response.status_code != 200:
            print("Could not access page", page)
            continue

        # Convert HTML into BeautifulSoup object
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all books on the page
        books = soup.find_all("article", class_="product_pod")

        # Extract information from each book
        for book in books:

            # Book title
            title = book.h3.a["title"]

            # Book price
            price = book.find(
                "p", class_="price_color"
            ).text.strip()

            # Book rating
            rating = book.find(
                "p", class_="star-rating"
            )["class"][1]

            # Availability
            availability = book.find(
                "p", class_="instock availability"
            ).text.strip()

            # Product URL
            relative_url = book.h3.a["href"]
            product_url = urljoin(url, relative_url)

            # Add information to list
            data.append({
                "Title": title,
                "Price": price,
                "Rating": rating,
                "Availability": availability,
                "Product URL": product_url
            })

    except requests.exceptions.RequestException as error:
        print("Error on page", page)
        print(error)

print()
print("========================================")
print("Scraping completed!")
print("========================================")

# Convert list into Pandas DataFrame
df = pd.DataFrame(data)

# Save data to CSV file
df.to_csv("books_data.csv", index=False)

# Display information
print()
print("Total books collected:", len(df))
print()
print("First 5 records:")
print(df.head())

print()
print("Dataset saved successfully as:")
print("books_data.csv")