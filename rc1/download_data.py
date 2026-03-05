# Written by ChatGPT

import os
import requests
from urllib.parse import urljoin, urlparse, unquote
from bs4 import BeautifulSoup
import unicodedata

BASE_URL = "https://www.mimuw.edu.pl/~ciebie/rc25-26/"


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    base = urlparse(BASE_URL)
    return parsed.netloc == base.netloc and parsed.path.startswith(base.path)


def clean_filename(url: str) -> str:
    # Extract raw filename from path
    raw = os.path.basename(urlparse(url).path)

    # URL decode
    decoded = unquote(raw)

    # Normalize to NFC (fix combining diacritics)
    normalized = unicodedata.normalize("NFC", decoded)

    return normalized


def download_file(url: str, dest_folder: str):
    filename = clean_filename(url)

    # Skip anything that does not look like a file
    if not filename or filename.endswith("/"):
        return

    os.makedirs(dest_folder, exist_ok=True)
    local_path = os.path.join(dest_folder, filename)

    print(f"Downloading {url}")
    print(f" → saving as {local_path}")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def crawl(url: str, dest_folder: str, visited: set):
    if url in visited:
        return
    visited.add(url)

    print(f"Visiting {url}")

    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href")
        if not href:
            continue
        if href.startswith("?") or href.startswith("#"):
            continue

        full_url = urljoin(url, href)

        if not is_valid_url(full_url):
            continue

        parsed = urlparse(full_url)
        path = parsed.path

        # Directory
        if href.endswith("/"):
            subfolder_name = os.path.basename(path.rstrip("/"))
            subfolder = os.path.join(dest_folder, subfolder_name)
            crawl(full_url, subfolder, visited)
            continue

        # File
        download_file(full_url, dest_folder)


def main():
    visited = set()
    crawl(BASE_URL, "data", visited)


if __name__ == "__main__":
    main()
