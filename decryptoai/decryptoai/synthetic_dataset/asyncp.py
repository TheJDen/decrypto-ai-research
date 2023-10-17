import asyncio
import aiohttp

__author__ = "Jaden Rodriguez"

async def fetch_url(session, url):
    async with session.get(url) as response:
        data = await response.text()
        return data

async def fetch_responses(urls):
    async with aiohttp.ClientSession() as session:
        api_calls = [fetch_url(session, url) for url in urls]
        return [await response for response in asyncio.as_completed(api_calls)]

my_urls = [
    "https://gopherhack.com",
    "https://maxjytyla.com"
] * 5

responses = asyncio.run(fetch_responses(my_urls))
for response in responses:
    print("="*12)
    print(response[:50])


