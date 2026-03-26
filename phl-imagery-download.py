import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://www.pasda.psu.edu/download/philacity/data/Imagery2025/tiles/" # noqa
OUTPUT_DIR = "./data/imagery/"
MAX_WORKERS = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

r = requests.get(BASE_URL)
r.raise_for_status()

soup = BeautifulSoup(r.text, "html.parser")


def download_file(url, path):
    '''
    downloads files from the specified url if the output dir exists
    prints the downloading file name in console for debugging
    restricts chunk handling size to avoid memory issues
    '''
    if os.path.exists(path):
        return
    try:
        file = url.rsplit("/")[-1]
        print(f"Downloading: {file}")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
    except Exception as e:
        return f"Error: {url} -> {e}"


jobs = []
for link in soup.find_all("a"):
    href = link.get("href")
    if not href.endswith(".zip"):
        continue

    full_url = urljoin(BASE_URL, href)
    local_path = os.path.join(OUTPUT_DIR, href.rsplit("/")[-1])

    jobs.append((full_url, local_path))

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

try:
    futures = [executor.submit(download_file, url, path) for url, path in jobs]
    for f in futures:
        f.result()  # allows interrupt to surface
except KeyboardInterrupt:
    print("\nStopping downloads...")
    executor.shutdown(wait=False, cancel_futures=True)
