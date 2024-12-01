#%%
import requests
from bs4 import BeautifulSoup
import os

def google_search(query, api_key, cse_id, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": num_results 
    }
    response = requests.get(url, params=params)
    return response.json().get("items", [])

def fetch_page_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def scrape_content(soup):
    content = soup.find_all('p')  # Example: Extracts all paragraphs
    page_text = " ".join([p.get_text() for p in content])
    return page_text

def search_and_scrape(query):
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    # query = "Where is Alabama"
    search_results = google_search(query, api_key, cse_id)
    page_contents = []
#%%
    for result in search_results:
        title = result['title']
        link = result['link']
        # print(f"Title: {title}\nLink: {link}")

        soup = fetch_page_content(link)
        if soup:
            page_content = scrape_content(soup)
            print(f"Content from {link}:\n", page_content[:500], "...\n")
            page_contents.append(page_content)
#%%
def main():
    search_and_scrape("LLM")
#%%
if __name__ == "__main__":
    main()
