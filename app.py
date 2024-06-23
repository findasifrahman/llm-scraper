import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import tiktoken
import json

# Define retry strategy
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def fetch_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def crawl_website(url):
    print(colored(f"Fetching URL: {url}", "green"))
    try:
        page_content = fetch_url(url)
        print(colored("Successfully fetched!", "blue"))
        return page_content
    except Exception as e:
        print(colored(f"Failed to fetch URL: {url}, Error: {e}", "red"))
        return None

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(colored(f"Data saved to {filename}", "green"))

def main():
    # List of URLs to crawl
    urls = [
        "https://amazon.com",
        "https://intricatlab.com",
        # Add more URLs as needed
    ]
    
    crawled_data = []
    
    for url in urls:
        content = crawl_website(url)
        if content:
            # Tokenize content using tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(content)
            
            crawled_data.append({
                "url": url,
                "content": content,
                "tokens": tokens
            })
    
    # Save crawled data to JSON file
    save_to_json(crawled_data, "crawled_data.json")

if __name__ == "__main__":
    main()