import requests
from bs4 import BeautifulSoup

keyword = "Rata-rata Harga Pangan Bulanan Tingkat Konsumen Provinsi"
url = "https://data.badanpangan.go.id/datasetpublications"

def scrape():
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for element in soup.find_all(string=lambda text: text and keyword in text):
        parent = element.find_parent()
        if parent:
            results.append({
                "text": element.strip(),
                "tag": parent.name,
                "parent_html": str(parent)[:200]
            })
    
    return results

if __name__ == "__main__":
    data = scrape()
    for item in data:
        print(f"Found in <{item['tag']}>: {item['text'][:100]}...")
