import asyncio
import json
from pathlib import Path
from fast_agent.core.prompt import Prompt
from fast_agent import FastAgent
from mcp.types import PromptMessage
from datetime import date
import os

# Optional: Load API key from environment variable
secret = os.environ.get("OPENAI_API_KEY")

current_date = date.today()

instruction = f"""
You are a TradingNews Analyst Agent specializing in helping active traders interpret recent news items and link them to trade ideas, risks, and opportunities. 
Your goal is to **help a trader** who inputs a **ticker symbol** plus **their current idea/intent/context**, by analyzing relevant recent news and delivering actionable, concise insight.

The current date is {current_date.isoformat()}.
"""

fast = FastAgent(name="news-analyzer")

@fast.agent(
        instruction=instruction,
        #api_key=secret
        )

async def main():
    async with fast.run() as agent:
        news_path = Path(__file__).resolve().parent / "decoy_news.json"
        with open(news_path, "r", encoding="utf-8") as f:
            news_text = f.read().strip()

        try:
            json.loads(news_text)
        except Exception as e:
            raise RuntimeError(f"decoy_news.json is not valid JSON: {e}")

        message = f'''
            Here are some recent news items:
            {news_text}
            ---
            '''

        analysis = await agent.send(
                                message=message)

        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
