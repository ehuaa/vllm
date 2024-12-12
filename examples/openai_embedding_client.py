from openai import AsyncOpenAI

from transformers import AutoTokenizer
import time
import asyncio
import json
import random
from typing import AsyncGenerator, List, Tuple
import numpy as np
import aiohttp


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


task = 'Given a web search query, retrieve relevant passages that answer the query'
max_length = 4096

# data = [json.loads(line)["text"] for line in (open("/nas/czh/sfr/539667.jsonl", "r"))]
# data = data[9:10] * 3
tokenizer = AutoTokenizer.from_pretrained("/workspace/nas/czh/sfr/SFR-Embedding-Mistral")
tokenizer.add_eos_token = True
data = []
with open('/workspace/nas/czh/part_1.tsv', 'r', encoding='utf-8') as file:
    for line in file:
        row = line.strip().split('|')[1].strip()
        # row = "1 L October 7, 1995 1–3 Buffalo Sabres (1995–96) 0–1–0 9,567 1995020005 2 L October 13, 1995 2–6 @ Florida Panthers (1995–96) 0–2–0 10,895 1995020036 3 W October 15, 1995 7–4 @ Tampa Bay Lightning (1995–96) 1–2–0 13,488 1995020049 4 W October 19, 1995 4–2 Calgary Flames (1995–96) 2–2–0 8,424 1995020067 5 L October 21, 1995 1–4 @ New Jersey Devils (1995–96) 2–3–0 17,620 1995020080 6 W October 22, 1995 4–2 @ New York Rangers (1995–96) 3–3–0 18,200 1995020087 7 W October 24, 1995 2–1 @ Detroit Red Wings (1995–96) 4–3–0 19,512 1995020095 8 W October 26, 1995 5–4 Los Angeles Kings (1995–96) 5–3–0 10,575 1995020104 9 L October 28, 1995 1–4 Florida Panthers (1995–96) 5–4–0 8,660 1995020115 10 L October 29, 1995 2–5 @ Philadelphia Flyers (1995–96) 5–5–0 17,328 1995020124 11 W November 2, 1995 5–0 @ Hartford Whalers (1995–96) 6–5–0 10,458 1995020146 12 L November 4, 1995 4–5 Hartford Whalers (1995–96) 6–6–0 8,794 1995020157 13 L November 8, 1995 1–7 Pittsburgh Penguins (1995–96) 6–7–0 10,137 1995020179 14 L November 9, 1995 3–4 @ Boston Bruins (1995–96) 6–8–0 17,261 1995020184 15 L November 11, 1995 2–3 Mighty Ducks of Anaheim (1995–96) 6–9–0 8,988 1995020196 16 L November 15, 1995 2–3 @ Hartford Whalers (1995–96) 6–10–0 7,641 1995020220 17 L November 16, 1995 3–5 @ Philadelphia Flyers (1995–96) 6–11–0 17,220 1995020225 18 L November 18, 1995 1–5 @ Montreal Canadiens (1995–96) 6–12–0 17,302 1995020240 19 L November 19, 1995 0–6 @ Buffalo Sabres (1995–96) 6–13–0 10,697 1995020246 20 L November 22, 1995 1–3 Winnipeg Jets (1995–96) 6–14–0 8,426 1995020261 21 T November 25, 1995 3–3 OT Boston Bruins (1995–96) 6–14–1 9,419 1995020278 22 L November 28, 1995 2–7 @ Pittsburgh Penguins (1995–96) 6–15–1 16,162 1995020291 23 L November 30, 1995 3–5 New York Islanders (1995–96) 6–16–1 8,167 1995020305 24 L December 2, 1995 2–4 New York Rangers (1995–96) 6–17–1 8,194 1995020318 25 L December 5, 1995 1–4 @ Toronto Maple Leafs (1995–96) 6–18–1 15,746 1995020334 26 W December 7, 1995 5–2 @ Chicago Blackhawks (1995–96) 7–18–1 17,552 1995020349 27 L December 9, 1995 3–7 Colorado Avalanche (1995–96) 7–19–1 9,169 1995020357 28 L December 12, 1995 1–2 @ San Jose Sharks (1995–96) 7–20–1 17,190 1995020377 29 L December 13, 1995 2–6 @ Los Angeles Kings (1995–96) 7–21–1 11,221 1995020384 30 L December 15, 1995 2–4 @ Mighty Ducks of Anaheim (1995–96) 7–22–1 17,174 1995020397 31 L December 17, 1995 1–4 @ Vancouver Canucks (1995–96) 7–23–1 16,006 1995020409 32 L December 18, 1995 1–3 @ Edmonton Oilers (1995–96) 7–24–1 8,419 1995020414 33 L December 23, 1995 2–4 Buffalo Sabres (1995–96) 7–25–1 8,615 1995020441 34 L December 26, 1995 4–6 @ New York Rangers (1995–96) 7–26–1 18,200 1995020452 35 W December 27, 1995 4–3 @ Buffalo Sabres (1995–96) 8–26–1 12,175 1995020458 36 L December 30, 1995 1–4 Montreal Canadiens (1995–96) 8–27–1 10,575 1995020477 37 L December 31, 1995 0–3 Tampa Bay Lightning (1995–96) 8–28–1 8,522 1995020482 38 L January 3, 1996 1–4 @ Pittsburgh Penguins (1995–96) 8–29–1 15,632 1995020495 39 L January 5, 1996 2–4 @ Hartford Whalers (1995–96) 8–30–1 12,239 1995020507 40 L January 6, 1996 4–5 @ New York Islanders (1995–96) 8–31–1 12,175 1995020517 41 L January 11, 1996 1–6 @ Washington Capitals (1995–96) 8–32–1 11,511 1995020549 42 L January 13, 1996 1–4 @ Tampa Bay Lightning (1995–96) 8–33–1 21,829 1995020561 43 L January 17, 1996 0–3 Montreal Canadiens (1995–96) 8–34–1 18,500 1995020579 44 L January 22, 1996 3–7 Chicago Blackhawks (1995–96) 8–35–1 13,872 1995020589 45 L January 24, 1996 3–4 Pittsburgh Penguins (1995–96) 8–36–1 17,149 1995020598 46 L January 25, 1996 2–4 Detroit Red Wings (1995–96) 8–37–1 16,882 1995020608 47 T January 27, 1996 2–2 OT Toronto Maple Leafs (1995–96) 8–37–2 18,500 1995020619 48 W January 29, 1996 4–2 St."
        # data.append(get_detailed_instruct(task, row))
        data.append(tokenizer(get_detailed_instruct(task, row), max_length=max_length - 1, padding=True, truncation=True, return_tensors="pt")["input_ids"].numpy().tolist())
        
data = data[:10]
openai_api_key = "EMPTY"
openai_api_base = "http://10.200.99.220:30224/v1"

client = AsyncOpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)


async def get_request(
    input_requests: List[str],
    request_rate: float,
) -> AsyncGenerator[Tuple[str], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(prompt) -> None:
    models = await client.models.list()
    embeddings = await client.embeddings.create(
        model=models.data[0].id,
        input=prompt,
    )
    for data in embeddings.data:
        print(data.embedding[:10])
    

async def benchmark(
    input_requests: List[str],
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt = request
        task = asyncio.create_task(
            send_request(prompt))
        tasks.append(task)
    await asyncio.gather(*tasks)


def main():
    input_requests = data

    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(input_requests, 400))
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.3f} s")

if __name__ == "__main__":
    main()
# for data in responses.data:
#     print(data.embedding)  # list of float of len 4096
# from openai import OpenAI

# # Modify OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:18192/v1"

# client = OpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# models = client.models.list()
# model = models.data[0].id
# import json
# data = []
# with open('/nas/czh/sfr/539667.jsonl', 'r', encoding='utf-8') as file:
#     for line in file:
#         row = json.loads(line)["text"]
#         data.append(row)

# responses = client.embeddings.create(
#     input=data,
#     model=model,
#     encoding_format="float",
# )

# for data in responses.data:
#     print(data.embedding[:10])  # list of float of len 4096