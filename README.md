This is a demonestration how you can implement a web scraping agent using

firecrawl and OPENAI.

I also used langchain to monitor agents performence and provide score based on score

You can scrape any site with this code

One of the problem of agent scrapping is you have to provide large context to llm and most models cant handle that sort of context

GPT-3.5-turbo cannot handle most of the request as its token per minute limit is very small

You need to have Open AI tier-2 access to handle the request efficiently

Declare Website Name and entity name that you want to scrap in the code and run the code

You will also need Firecrawl, Langsmith and OPENAI API key

GPT-4o performs very well with this agent

You can choos between GPT-4o, GPT-4-turbo or GPT-3.5

GPT-3.5 is less expensive but it does not provide effective output with internet search scraping

Change the entity name and website name inside TEST.PY or pdffinder.py

Good Luck