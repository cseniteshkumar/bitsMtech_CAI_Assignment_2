import json
import wikipedia




def generate_random_urls(topcis="Science and Technology",ignoreUrls=[] num_urls=100):
    
    generatedUrls = []

    while len(generatedUrls) < num_urls:
        try:
            random_page = wikipedia.random(pages=1)
            url = wikipedia.page(random_page).url

            if url not in generatedUrls and url not in ignoreUrls:
                generatedUrls.append(url)

        except Exception as e:
            print(f"Error fetching page: {e}")

    
    return generatedUrls


def save_fixed_urls_to_json(fixed_urls, filename="fixed_urls.json"):
    with open(filename, "w") as f:
        json.dump(fixed_urls, f, indent=2)


if __name__ == "__main__":
    fixed_urls = generate_fixed_urls()
    storage_path = "fixed_urls.json"
    save_fixed_urls_to_json(fixed_urls, storage_path)

    print(f"fixed_urls.json created with {len(fixed_urls)} URLs")

    print("\nLast 3 entries:")
    for url in fixed_urls[-3:]:
        print(f" - {url}")
