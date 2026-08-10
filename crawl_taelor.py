import asyncio
from pathlib import Path

from crawl4ai import AsyncWebCrawler
from streamlit import markdown


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
]



async def crawl_taelor():

    print("Starting Taelor crawl...")

    async with AsyncWebCrawler() as crawler:

        for url in TAELOR_URLS:

            print(f"\n🌐 Crawling: {url}")

            result = await crawler.arun(url)

            if not result.success:
                print(f"❌ Failed: {url}")
                print(result.error_message)
                continue

            print(f"✅ Success: {url}")

            # Create a safe filename
            filename = (
                url.replace("https://taelor.style/", "")
                   .replace("/", "_")
                   .replace("-", "_")
            )

            if not filename:
                filename = "homepage"

            Path("data").mkdir(exist_ok=True)

            Path(f"data/{filename}.md").write_text(
                result.markdown,
                encoding="utf-8"
            )

            print(f"📄 Saved: data/{filename}.md")
            print(f"Characters: {len(result.markdown)}")


if __name__ == "__main__":
    asyncio.run(crawl_taelor())

