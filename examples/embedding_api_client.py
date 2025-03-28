"""Example Python client for vllm.entrypoints.api_server
NOTE: The API server is used only for demonstration and simple performance
benchmarks. It is not intended for production use.
For production use, we recommend vllm.entrypoints.openai.api_server
and the OpenAI client API
"""

import argparse
import json
from typing import Iterable, List

import struct
import requests
import time


def post_http_request(api_url: str, question: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    # pload = {
    #     "prompt": prompt,
    #     "n": n,
    #     "use_beam_search": True,
    #     "temperature": 0.0,
    #     "max_tokens": 16,
    #     "stream": stream,
    # }
    req_data = {"prompts": question}
    response = requests.post(api_url, headers=headers, json=req_data, stream=True)
    return response


def get_response(response: requests.Response) -> List[str]:
    packed_data = response.content
    rows, cols = struct.unpack('II', packed_data[:8])
    
    # unpack data
    flat_data = struct.unpack(f'{rows*cols}f', packed_data[8:])
    
    # reconstruct
    return [list(flat_data[i*cols:(i+1)*cols]) for i in range(rows)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="10.200.48.45")
    parser.add_argument("--port", type=int, default=18192)
    # parser.add_argument("--host", type=str, default="10.244.127.79")
    # parser.add_argument("--port", type=int, default=8000)
    # parser.add_argument("--n", type=int, default=4)
    # parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    # prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/embeddings"
    # n = args.n
    stream = args.stream

    data = ["Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: A stop order won't be automatically fired in after-hours trading?", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How can I remove the movement of the stock market as a whole from the movement in price of an individual share?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: May 6, 2010 stock market decline/plunge: Why did it drop 9% in a few minutes?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What to do with small dividends in brokerage account?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Buying insurance (extended warranty or guarantee) on everyday goods / appliances?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Mortgage refinancing fees', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is it ok to have multiple life time free credit cards?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Basic questions about investing in stocks', "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Currently sole owner of a property. My girlfriend is looking to move in with me and is offering to pay 'rent'. Am I at risk here?", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Sage Instant Accounts or Quickbooks?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What are the advantages of a Swiss bank account?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What does “points” mean in such contexts (stock exchange, I believe)?', "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: I'm thinking about selling some original artwork: when does the government start caring about sales tax and income tax and such?", "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Calculate investment's interest rate to break-even insurance cost [duplicate]", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Why would a car company lend me money at a very low interest rate?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Personal Tax Deduction for written work to a recognized 501c3', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What options do I have at 26 years old, with 1.2 million USD?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: ISA trading account options for US citizens living in the UK', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Trouble sticking to a budget when using credit cards for day to day transactions?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is 401k as good as it sounds given the way it is taxed?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How much time would I have to spend trading to turn a profit?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Potential pitfalls of this volume trading strategy', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is there any way to buy a new car directly from Toyota without going through a dealership?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Does a bid and ask price exist for indices like the S&P500?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: If NYSE has market makers, what is the role of NYSE ARCA which is an ECN', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Scam or Real: A woman from Facebook apparently needs my bank account to send money', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How to evaluate stocks? e.g. Whether some stock is cheap or expensive?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Micro-investing: How to effectively invest frequent small amounts of money in equities?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Does Apple have $0 of treasury stock?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Why are there hidden bids and offers in the US stock market for the more illiquid stocks?', "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: The doctor didn't charge the health insurance in time, am I liable?", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: ~$75k in savings - Pay off house before new home?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How can I buy government bonds from foreign countries?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How can one get their FICO/credit scores for free? (really free)', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: UK - How to receive payments in euros', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Why not just invest in the market?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is there a government-mandated resource that lists the shareholders of a public company?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is there a lower threshold for new EU VAT changes coming 1 Jan 2015 related to the sale of digital goods?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: I received $1000 and was asked to send it back. How was this scam meant to work?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: The best credit card for people who pay their balance off every month', "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Do stock splits make one's shares double in voting power?", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Options for Cash only Buyout due to Company Merger', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What size “nest egg” should my husband and I have, and by what age?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: If throwing good money after bad is generally a bad idea, is throwing more money after good Ok?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is a “total stock market” index fund diverse enough alone?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Deal with stock PSEC', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is it common for a new car of about $16k to be worth only $4-6k after three years?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: If I put a large down payment (over 50%) towards a car loan, can I reduce my interest rate and is it smart to even put that much down?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Why are U.S. credit unions not open to everyone?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Single investment across multiple accounts… good, bad, indifferent?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Are online mortgage lenders as good as local brick-and-mortar ones?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Investing small amounts at regular intervals while minimizing fees?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Where to park money while saving for a car', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Canada: New mortgage qualification rules, 2010: Why, what, & when in effect?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Why do car rental companies prefer/require credit over debit cards?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How to keep control of shared expenses inside marriage?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Can you sell stocks/commodities for any price you wish (either direct or market)?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Wardrobe: To Update or Not? How-to without breaking the bank', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Who Can I Hire To Calculate the Value of An Estate?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Purchasing options between the bid and ask prices, or even at the bid price or below?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is gold really an investment or just a hedge against inflation?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is there a good rule of thumb for how much I should have set aside as emergency cash?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How to execute a large stock purchase, relative to the order book?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Diversify or keep current stock to increase capital gains', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is it advisable to go for an auto loan if I can make the full payment for a new car?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Car insurance (UK) excludes commute to and from work, will not pay on claim during non-commute', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What kind of symbol can be shorted?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Facebook buying WhatsApp for 19 Billion. How are existing shareholders affected?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Need something more basic than a financial advisor or planner', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: What effect would currency devaluation have on my investments?', "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Filing Form 7004 if an LLC's only members are husband and wife", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is CFD a viable option for long-term trading?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is it normal to think of money in different “contexts”?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Should the price of fuel in Australia at this point be so high?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Working out if I should be registered as self-employed in the UK', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Can I buy and sell a house quickly to access the money in a LISA?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How to represent “out of pocket” purchases in general ledger journal entry?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Does bull/bear market actually make a difference?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: When are investments taxed?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Can preventive health checkup be claimed as a separate expense from medical expenses?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Should I make partial pre-payments on an actuarial loan?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is it worth buying real estate just to safely invest money?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is an interest-only mortgage a bad idea?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Can a stop loss order be triggered by random price?', "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: I thought student loans didn't have interest, or at least very low interest? [UK]", 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: should the Market Capitalization be equal to the Equity of the firm', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Are car buying services worth it?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: When would one actually want to use a market order instead of a limit order?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Is person-person lending/borrowing protected by law?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Paying off mortgage or invest in annuity', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How to tell if an option is expensive', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Less than a year at my first job out of college, what do I save for first?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Are there alternatives to double currency account to manage payments in different currencies?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: CFD market makers: How is the price coupled to the underlying security?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Can someone explain the Option Chain of AMD for me?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Are Shiller real-estate futures and options catching on with investors?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Why are some long term investors so concerned about their entry price?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: How feasible would it be to retire just maxing out a Roth IRA?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Does an employee have the right to pay the federal and state taxes themselves instead of having employer doing it?', 'Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: Do I have to pay a capital gains tax if I rebuy different stocks?']
    # with open('/root/vllm_test/czh/vllm/part_1.tsv', 'r', encoding='utf-8') as file:
    #     for line in file:
    #         row = line.strip().split('|')[1].strip()
    #         data.append(row)
    # with open('/root/models/czh/sfr/539667.jsonl', 'r', encoding='utf-8') as file:
    #     for line in file:
    #         row = json.loads(line)["text"]
    #         data.append(row)
    #         break
    # print(f"Prompt: {prompt!r}\n", flush=True)
    
    # print(data)
    t0 = time.perf_counter()

    response = post_http_request(api_url, data)
    output = get_response(response)
    print(time.perf_counter() - t0)
    # print(output)
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('/root/models/zhangzeqing/embedding_synthetic_data/intfloat/multilingual-e5-large-instruct')

    embeddings = model.encode(data, convert_to_tensor=True, normalize_embeddings=True)
    hf_res = embeddings
    print(output[0][:10])
    print(hf_res[0][:10])
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

    check_embeddings_close(output, hf_res, "vllm", "hf")
   
