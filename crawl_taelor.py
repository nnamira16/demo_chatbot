import asyncio
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from crawl4ai import AsyncWebCrawler
from streamlit import markdown

BLOG_PAGES = [
    f"https://taelor.style/blogs/mens-style?page={page}"
    for page in range(1, 15)
]

TAELOR_URLS = [
    "https://taelor.style/", 
    "https://taelor.style/pages/how-it-works",
    "https://taelor.style/pages/faq", 
    "https://taelor.style/pages/membership",
    "https://taelor.style/pages/taelor-collection-browsing", 
    "https://taelor.style/pages/brands",
    "https://taelor.style/pages/stylist-introduction",
    "https://taelor.style/pages/mission",
    "https://taelor.style/pages/reviews",
    "https://taelor.style/blogs/news",
    "https://taelor.style/pages/faq",
    "https://taelor.style/blogs/mens-style"
] + BLOG_PAGES


def safe_filename(url):
    parsed = urlparse(url)

    path = (
        parsed.path
        .strip("/")
        .replace("/", "_")
        .replace("-", "_")
    )

    page = parse_qs(parsed.query).get("page", [None])[0]

    if page:
        return f"{path}_page_{page}"

    return path or "homepage"


async def crawl_taelor():

    print("Starting Taelor crawl...")

    Path("data").mkdir(exist_ok=True)

    async with AsyncWebCrawler() as crawler:

        for url in TAELOR_URLS:

            print(f"\n🌐 Crawling: {url}")

            result = await crawler.arun(url)

            if not result.success:
                print(f"❌ Failed: {url}")
                print(result.error_message)
                continue

            print(f"✅ Success: {url}")

            filename = safe_filename(url)

            Path(f"data/{filename}.md").write_text(
                result.markdown,
                encoding="utf-8"
            )

            print(f"📄 Saved: data/{filename}.md")
            print(f"Characters: {len(result.markdown)}")


if __name__ == "__main__":
    asyncio.run(crawl_taelor())

