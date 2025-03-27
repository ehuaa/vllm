# from openai import AsyncOpenAI

# from transformers import AutoTokenizer
# import time
# import asyncio
# import json
# import random
# from typing import AsyncGenerator, List, Tuple
# import numpy as np
# import aiohttp

# # data = [json.loads(line)["text"] for line in (open("/nas/czh/sfr/539667.jsonl", "r"))]
# # data = data[9:10] * 3
# data = []
# with open('/root/vllm_test/czh/vllm/part_1.tsv', 'r', encoding='utf-8') as file:
#     for line in file:
#         row = line.strip().split('|')[1].strip()
#         # row = "1 L October 7, 1995 1–3 Buffalo Sabres (1995–96) 0–1–0 9,567 1995020005 2 L October 13, 1995 2–6 @ Florida Panthers (1995–96) 0–2–0 10,895 1995020036 3 W October 15, 1995 7–4 @ Tampa Bay Lightning (1995–96) 1–2–0 13,488 1995020049 4 W October 19, 1995 4–2 Calgary Flames (1995–96) 2–2–0 8,424 1995020067 5 L October 21, 1995 1–4 @ New Jersey Devils (1995–96) 2–3–0 17,620 1995020080 6 W October 22, 1995 4–2 @ New York Rangers (1995–96) 3–3–0 18,200 1995020087 7 W October 24, 1995 2–1 @ Detroit Red Wings (1995–96) 4–3–0 19,512 1995020095 8 W October 26, 1995 5–4 Los Angeles Kings (1995–96) 5–3–0 10,575 1995020104 9 L October 28, 1995 1–4 Florida Panthers (1995–96) 5–4–0 8,660 1995020115 10 L October 29, 1995 2–5 @ Philadelphia Flyers (1995–96) 5–5–0 17,328 1995020124 11 W November 2, 1995 5–0 @ Hartford Whalers (1995–96) 6–5–0 10,458 1995020146 12 L November 4, 1995 4–5 Hartford Whalers (1995–96) 6–6–0 8,794 1995020157 13 L November 8, 1995 1–7 Pittsburgh Penguins (1995–96) 6–7–0 10,137 1995020179 14 L November 9, 1995 3–4 @ Boston Bruins (1995–96) 6–8–0 17,261 1995020184 15 L November 11, 1995 2–3 Mighty Ducks of Anaheim (1995–96) 6–9–0 8,988 1995020196 16 L November 15, 1995 2–3 @ Hartford Whalers (1995–96) 6–10–0 7,641 1995020220 17 L November 16, 1995 3–5 @ Philadelphia Flyers (1995–96) 6–11–0 17,220 1995020225 18 L November 18, 1995 1–5 @ Montreal Canadiens (1995–96) 6–12–0 17,302 1995020240 19 L November 19, 1995 0–6 @ Buffalo Sabres (1995–96) 6–13–0 10,697 1995020246 20 L November 22, 1995 1–3 Winnipeg Jets (1995–96) 6–14–0 8,426 1995020261 21 T November 25, 1995 3–3 OT Boston Bruins (1995–96) 6–14–1 9,419 1995020278 22 L November 28, 1995 2–7 @ Pittsburgh Penguins (1995–96) 6–15–1 16,162 1995020291 23 L November 30, 1995 3–5 New York Islanders (1995–96) 6–16–1 8,167 1995020305 24 L December 2, 1995 2–4 New York Rangers (1995–96) 6–17–1 8,194 1995020318 25 L December 5, 1995 1–4 @ Toronto Maple Leafs (1995–96) 6–18–1 15,746 1995020334 26 W December 7, 1995 5–2 @ Chicago Blackhawks (1995–96) 7–18–1 17,552 1995020349 27 L December 9, 1995 3–7 Colorado Avalanche (1995–96) 7–19–1 9,169 1995020357 28 L December 12, 1995 1–2 @ San Jose Sharks (1995–96) 7–20–1 17,190 1995020377 29 L December 13, 1995 2–6 @ Los Angeles Kings (1995–96) 7–21–1 11,221 1995020384 30 L December 15, 1995 2–4 @ Mighty Ducks of Anaheim (1995–96) 7–22–1 17,174 1995020397 31 L December 17, 1995 1–4 @ Vancouver Canucks (1995–96) 7–23–1 16,006 1995020409 32 L December 18, 1995 1–3 @ Edmonton Oilers (1995–96) 7–24–1 8,419 1995020414 33 L December 23, 1995 2–4 Buffalo Sabres (1995–96) 7–25–1 8,615 1995020441 34 L December 26, 1995 4–6 @ New York Rangers (1995–96) 7–26–1 18,200 1995020452 35 W December 27, 1995 4–3 @ Buffalo Sabres (1995–96) 8–26–1 12,175 1995020458 36 L December 30, 1995 1–4 Montreal Canadiens (1995–96) 8–27–1 10,575 1995020477 37 L December 31, 1995 0–3 Tampa Bay Lightning (1995–96) 8–28–1 8,522 1995020482 38 L January 3, 1996 1–4 @ Pittsburgh Penguins (1995–96) 8–29–1 15,632 1995020495 39 L January 5, 1996 2–4 @ Hartford Whalers (1995–96) 8–30–1 12,239 1995020507 40 L January 6, 1996 4–5 @ New York Islanders (1995–96) 8–31–1 12,175 1995020517 41 L January 11, 1996 1–6 @ Washington Capitals (1995–96) 8–32–1 11,511 1995020549 42 L January 13, 1996 1–4 @ Tampa Bay Lightning (1995–96) 8–33–1 21,829 1995020561 43 L January 17, 1996 0–3 Montreal Canadiens (1995–96) 8–34–1 18,500 1995020579 44 L January 22, 1996 3–7 Chicago Blackhawks (1995–96) 8–35–1 13,872 1995020589 45 L January 24, 1996 3–4 Pittsburgh Penguins (1995–96) 8–36–1 17,149 1995020598 46 L January 25, 1996 2–4 Detroit Red Wings (1995–96) 8–37–1 16,882 1995020608 47 T January 27, 1996 2–2 OT Toronto Maple Leafs (1995–96) 8–37–2 18,500 1995020619 48 W January 29, 1996 4–2 St."
#         # data.append(get_detailed_instruct(task, row))
#         data.append(row)

# data = data[:1000]
# openai_api_key = "EMPTY"
# openai_api_base = "http://10.200.99.220:30410/v1"

# client = AsyncOpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )


# async def get_request(
#     input_requests: List[str],
#     request_rate: float,
# ) -> AsyncGenerator[Tuple[str], None]:
#     input_requests = iter(input_requests)
#     for request in input_requests:
#         yield request

#         if request_rate == float("inf"):
#             # If the request rate is infinity, then we don't need to wait.
#             continue
#         # Sample the request interval from the exponential distribution.
#         interval = np.random.exponential(1.0 / request_rate)
#         # The next request will be sent after the interval.
#         await asyncio.sleep(interval)


# async def send_request(prompt) -> None:
#     models = await client.models.list()
#     embeddings = await client.embeddings.create(
#         model=models.data[0].id,
#         input=prompt,
#         extra_body={"instruction": "i'm czh"}
#     )
#     for data in embeddings.data:
#         print(data.embedding[:10])
    

# async def benchmark(
#     input_requests: List[str],
#     request_rate: float,
# ) -> None:
#     tasks: List[asyncio.Task] = []
#     async for request in get_request(input_requests, request_rate):
#         prompt = request
#         task = asyncio.create_task(
#             send_request(prompt))
#         tasks.append(task)
#     await asyncio.gather(*tasks)


# def main():
#     input_requests = data

#     benchmark_start_time = time.perf_counter()
#     asyncio.run(
#         benchmark(input_requests, 1000))
#     benchmark_end_time = time.perf_counter()
#     benchmark_time = benchmark_end_time - benchmark_start_time
#     print(f"Total time: {benchmark_time:.3f} s")

# if __name__ == "__main__":
#     main()
# for data in responses.data:
#     print(data.embedding)  # list of float of len 4096
from openai import OpenAI
import time

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:18192/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id
import json
data = ["Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: A stop order won't be automatically fired in after-hours trading?", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How can I remove the movement of the stock market as a whole from the movement in price of an individual share?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: May 6, 2010 stock market decline/plunge: Why did it drop 9% in a few minutes?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What to do with small dividends in brokerage account?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Buying insurance (extended warranty or guarantee) on everyday goods / appliances?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Mortgage refinancing fees', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is it ok to have multiple life time free credit cards?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Basic questions about investing in stocks', "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Currently sole owner of a property. My girlfriend is looking to move in with me and is offering to pay 'rent'. Am I at risk here?", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Sage Instant Accounts or Quickbooks?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What are the advantages of a Swiss bank account?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What does “points” mean in such contexts (stock exchange, I believe)?', "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: I'm thinking about selling some original artwork: when does the government start caring about sales tax and income tax and such?", "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Calculate investment's interest rate to break-even insurance cost [duplicate]", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Why would a car company lend me money at a very low interest rate?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Personal Tax Deduction for written work to a recognized 501c3', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What options do I have at 26 years old, with 1.2 million USD?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: ISA trading account options for US citizens living in the UK', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Trouble sticking to a budget when using credit cards for day to day transactions?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is 401k as good as it sounds given the way it is taxed?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How much time would I have to spend trading to turn a profit?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Potential pitfalls of this volume trading strategy', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is there any way to buy a new car directly from Toyota without going through a dealership?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Does a bid and ask price exist for indices like the S&P500?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: If NYSE has market makers, what is the role of NYSE ARCA which is an ECN', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Scam or Real: A woman from Facebook apparently needs my bank account to send money', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How to evaluate stocks? e.g. Whether some stock is cheap or expensive?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Micro-investing: How to effectively invest frequent small amounts of money in equities?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Does Apple have $0 of treasury stock?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Why are there hidden bids and offers in the US stock market for the more illiquid stocks?', "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: The doctor didn't charge the health insurance in time, am I liable?", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: ~$75k in savings - Pay off house before new home?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How can I buy government bonds from foreign countries?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How can one get their FICO/credit scores for free? (really free)', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: UK - How to receive payments in euros', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Why not just invest in the market?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is there a government-mandated resource that lists the shareholders of a public company?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is there a lower threshold for new EU VAT changes coming 1 Jan 2015 related to the sale of digital goods?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: I received $1000 and was asked to send it back. How was this scam meant to work?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: The best credit card for people who pay their balance off every month', "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Do stock splits make one's shares double in voting power?", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Options for Cash only Buyout due to Company Merger', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What size “nest egg” should my husband and I have, and by what age?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: If throwing good money after bad is generally a bad idea, is throwing more money after good Ok?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is a “total stock market” index fund diverse enough alone?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Deal with stock PSEC', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is it common for a new car of about $16k to be worth only $4-6k after three years?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: If I put a large down payment (over 50%) towards a car loan, can I reduce my interest rate and is it smart to even put that much down?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Why are U.S. credit unions not open to everyone?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Single investment across multiple accounts… good, bad, indifferent?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Are online mortgage lenders as good as local brick-and-mortar ones?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Investing small amounts at regular intervals while minimizing fees?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Where to park money while saving for a car', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Canada: New mortgage qualification rules, 2010: Why, what, & when in effect?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Why do car rental companies prefer/require credit over debit cards?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How to keep control of shared expenses inside marriage?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Can you sell stocks/commodities for any price you wish (either direct or market)?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Wardrobe: To Update or Not? How-to without breaking the bank', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Who Can I Hire To Calculate the Value of An Estate?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Purchasing options between the bid and ask prices, or even at the bid price or below?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is gold really an investment or just a hedge against inflation?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is there a good rule of thumb for how much I should have set aside as emergency cash?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How to execute a large stock purchase, relative to the order book?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Diversify or keep current stock to increase capital gains', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is it advisable to go for an auto loan if I can make the full payment for a new car?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Car insurance (UK) excludes commute to and from work, will not pay on claim during non-commute', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What kind of symbol can be shorted?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Facebook buying WhatsApp for 19 Billion. How are existing shareholders affected?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Need something more basic than a financial advisor or planner', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What effect would currency devaluation have on my investments?', "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Filing Form 7004 if an LLC's only members are husband and wife", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is CFD a viable option for long-term trading?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is it normal to think of money in different “contexts”?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Should the price of fuel in Australia at this point be so high?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Working out if I should be registered as self-employed in the UK', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Can I buy and sell a house quickly to access the money in a LISA?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How to represent “out of pocket” purchases in general ledger journal entry?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Does bull/bear market actually make a difference?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: When are investments taxed?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Can preventive health checkup be claimed as a separate expense from medical expenses?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Should I make partial pre-payments on an actuarial loan?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is it worth buying real estate just to safely invest money?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is an interest-only mortgage a bad idea?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Can a stop loss order be triggered by random price?', "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: I thought student loans didn't have interest, or at least very low interest? [UK]", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: should the Market Capitalization be equal to the Equity of the firm', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Are car buying services worth it?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: When would one actually want to use a market order instead of a limit order?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is person-person lending/borrowing protected by law?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Paying off mortgage or invest in annuity', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How to tell if an option is expensive', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Less than a year at my first job out of college, what do I save for first?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Are there alternatives to double currency account to manage payments in different currencies?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: CFD market makers: How is the price coupled to the underlying security?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Can someone explain the Option Chain of AMD for me?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Are Shiller real-estate futures and options catching on with investors?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Why are some long term investors so concerned about their entry price?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How feasible would it be to retire just maxing out a Roth IRA?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Does an employee have the right to pay the federal and state taxes themselves instead of having employer doing it?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Do I have to pay a capital gains tax if I rebuy different stocks?']

t0 = time.perf_counter()
responses = client.embeddings.create(
    input=data,
    model=model,
    encoding_format="float",
)
print(time.perf_counter() - t0)

vllm_res = []
for res_data in responses.data:
    vllm_res.append(res_data.embedding)


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('/root/models/zhangzeqing/embedding_synthetic_data/intfloat/multilingual-e5-large-instruct')

embeddings = model.encode(data, convert_to_tensor=True, normalize_embeddings=True)
hf_res = embeddings
print(vllm_res[26][:10], hf_res[26][:10])

import torch
import torch.nn.functional as F


def check_embeddings_close(
    embeddings_0_lst,
    embeddings_1_lst,
    name_0: str,
    name_1: str,
    tol: float = 1e-3,
) -> None:
    assert len(embeddings_0_lst) == len(embeddings_1_lst)

    for prompt_idx, (embeddings_0, embeddings_1) in enumerate(
            zip(embeddings_0_lst, embeddings_1_lst)):
        assert len(embeddings_0) == len(embeddings_1), (
            f"Length mismatch: {len(embeddings_0)} vs. {len(embeddings_1)}")

        sim = F.cosine_similarity(torch.tensor(embeddings_0),
                                  torch.tensor(embeddings_1).cpu(),
                                  dim=0)
        print(sim)
        fail_msg = (f"Test{prompt_idx}:"
                    f"\n{name_0}:\t{embeddings_0[:16]!r}"
                    f"\n{name_1}:\t{embeddings_1[:16]!r}")

        assert sim >= 1 - tol, fail_msg

check_embeddings_close(vllm_res, hf_res, "vllm", "hf")
