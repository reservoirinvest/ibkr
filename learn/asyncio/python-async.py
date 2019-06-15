from aiohttp import ClientSession, TCPConnector
import asyncio
from itertools import islice
import sys

# import nest_asyncio
# nest_asyncio.apply()

async def fetch(url):
    async with ClientSession() as s, s.get(url) as res:
        ret = await res.read()
        print(ret)
        return ret
    
# asyncio.run(fetch("http://example.com"))

def limited_as_completed(coros, limit):
    futures = [
        asyncio.ensure_future(c)
        for c in islice(coros, 0, limit)
    ]
    async def first_to_finish():
        while True:
            await asyncio.sleep(0)
            for f in futures:
                if f.done():
                    futures.remove(f)
                    try:
                        newf = next(coros)
                        futures.append(
                            asyncio.ensure_future(newf))
                    except StopIteration as e:
                        pass
                    return f.result()
    while len(futures) > 0:
        yield first_to_finish()

# code using asyncio.as_completed() for intermittent gather
async def print_when_done(tasks):
    for res in limited_as_completed(tasks, 100):
        print(await res)
        
coros = [
    fetch("http://example.com")
    for i in range(1000)
]

asyncio.run(print_when_done(coros))