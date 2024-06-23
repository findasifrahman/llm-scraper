from dotenv import load_dotenv
import os

#from openai import OpenAI
import openai
import re,time
from firecrawl import FirecrawlApp
import json
from tenacity import retry,wait_random_exponential,stop_after_attempt
from termcolor import colored
import tiktoken 
from langsmith import traceable
from langsmith.wrappers import wrap_openai



OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')
max_tokens = 10000

load_dotenv()
client = wrap_openai(openai.Client())#OpenAI()
GPT_MODEL = "gpt-3.5-turbo"#"gpt-4o"#'gpt-3.5-turbo'#"gpt-4-turbo-2024-04-09"#"gpt-4-turbo"

def save_to_json(data, filename):
    with open(filename, 'a+') as f:
        json.dump(data, f, indent=4)
    print(colored(f"Data saved to {filename}", "green"))

# web scrapping
@traceable(run_type="tool", name="Scrape")
def scrape(url):
    app = FirecrawlApp()

    #scrape single url
    try:
        scraped_data = app.scrape_url(url)
    except Exception as e:
        print(e)
        print("Error in scraping")
        return "unable to scrap the url"
    
    
    links_scraped.append(url)
    return scraped_data["markdown"]

# ok
@traceable(run_type="tool", name="Internet Search")
def search(query,entity_name:str):
    app = FirecrawlApp()
    params = {"pageOptions": {"fetchPageContent": True}}

    #Scrape a single url
    search_result = app.search(query, params=params)
    search_result_str = str(search_result)

    data_keys_to_search = [obj["name"] for obj in data_points if obj["value"] is None]
    prompt = f"""{search_result_str}
    ----
    About is some search results from the internet about {query}
    Your goal is to find specific list of information about an entity called {entity_name}
    
    Please extract information from the search results above in JSON format

    {{
        "related urls to scrape furthur": ["url1","url2","url3"],
        'info found': [{{
            research_item_1: 'xxxx',
            "reference": url
        }},
        {{
            research_item_2: 'xxxx',
            "reference": url
        }},
        ...]
    }}

    where research_item_1, research_item_2 are the actual research item names you are looking for:
    Only return research_items that you actually found,
    if no research item information found from the content provider then don't return any json

    Extracted JSON: 
    """
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    result = response.choices[0].message["content"]
    return result

# ok
@traceable(run_type="tool", name="Update Data Points")
def update_data(datas_update):
    """
    Update the state with new data points found
    
    Args:
        state(dict): The current graph state
        datas_update(List[dict]): The new data points found, have to follow the format [{"name":"xxx","value":"xxx"}]
    
    Returns:
        state(dict): The updated graph state
    """
    print(f"Updating the data {datas_update}")

    for data in datas_update:
        for obj in data_points:
            if obj["name"] == data["name"]:
                obj["value"] = data["value"]
                obj["reference"] = data["reference"]

    return f"data updated: {data_points}"

# ok
@traceable(run_type="llm", name="Agent Chat completion")
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tool_choice, tools, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )
        return response
    except Exception as e:
        print("Unable to generate chat completion")
        print(f"Exception: {e}")
        return e
    
# ok
def pretty_print_conversation(message):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tools":"magenta"
    }
    if message["role"] == "system":
        print(colored(f"system: {message['content']}", role_to_color[message["role"]]))
        save_to_json(f"system: {message['content']}", "crawled_data.json")
    elif message["role"] == "user":
        print(colored(f"user: {message['content']}", role_to_color[message["role"]]))
        save_to_json(f"user: {message['content']}", "crawled_data.json")
    elif message["role"] == "assistant" and message.get("tool_calls"):
        print(
            colored(
                f"assistant: {message['tool_calls']}\n",
                role_to_color[message["role"]]
            )
        )
        save_to_json(f"assistant: {message['tool_calls']}\n", "crawled_data.json")
    elif message["role"] == "assistant" and not message.get("tool_calls"):
        print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        save_to_json(f"assistant: {message['content']}\n", "crawled_data.json")
    elif message["role"] == "tools":
        print(colored(f"function: ({message['name']}): {message['content']}\n", 
                      role_to_color[message["role"]]))
        save_to_json(f"function: {message['name']}: {message['content']}\n", "crawled_data.json")


tools_list = {"scrape": scrape, "search": search, "update_data": update_data}

# ok
@traceable(name="Optimize memory")
def memory_optimize(messages: list):
    system_prompt = messages[0]["content"]

    # token count
    encoding = tiktoken.encoding_for_model(GPT_MODEL)

    if len(messages) > 24 or len(encoding.encode(str(messages))) > 10000:
    #if len(encoding.encode(str(messages))) > max_tokens:
        latest_messaages = messages[-12:]

        #latest_messaages = messages
        token_count_latest_message = len(encoding.encode(str(latest_messaages)))
        print(f"Token count latest messages: {token_count_latest_message}")

        #while(token_count_latest_message > max_tokens):
            #latest_messaages.pop(0)
            #token_count_latest_message = len(encoding.encode(str(latest_messaages)))
            #print(f"Token count latest messages: {token_count_latest_message}")

        #print(f"Final Token count latest messages: {token_count_latest_message}")
        save_to_json(f"Final Token count latest messages: {token_count_latest_message}\n", "crawled_data.json")
        index= messages.index(latest_messaages[0])
        early_messages = messages[:index]

        prompt = f""" {early_messages}
        ----
        Above is the past history of conversation between user and AI, including actions AI aleady taken
        Please summarize the past action taken so far, specifically around:
        - what data source have the AI look up already
        - what data points have been found so far

        SUMMARY:
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        system_prompt = f"""{system_prompt}; Here is a summary of past actions taken so far: {response.choices[0].message.content}"""
        messages = [{"role": "system", "content": system_prompt}] + latest_messaages

        return messages
    
    return messages

@traceable(name="Call Agent")
def call_agent(prompt, system_prompt, tools, plan):
    messages = []

    if plan:
        messages.append(
            {
                "role": "user",
                "content":(
                            system_prompt
                            + "  "
                            + prompt
                            + "  Let's think step by step, make a plan first" 
                ),
            }
        )

        #print(messages)
        save_to_json(messages, "crawled_data.json")
        chat_response = chat_completion_request(messages, tool_choice="none", tools=tools)
        print(chat_response.choices[0].message.content)
        messages = [
            {"role": "user","content": (system_prompt + "  " + prompt)},
            {"role": "assistant","content": chat_response.choices[0].message.content},
        ]

    else:
        messages.append({"role": "user", "content": (system_prompt + "  " + prompt)})

    state = "running"

    for message in messages:
        pretty_print_conversation(message)

    i = 0
    while state == "running":
        i += 1
        chat_response = chat_completion_request(messages, tool_choice=None, tools=tools)

        if isinstance(chat_response, Exception):
            print("Error in chat completion request")
            print(f"Exception: {chat_response}")
            state = "finished"
        else:
            current_choice = chat_response.choices[0]
            print(f"index is--{i}")
            #print(f"current choice is: {current_choice}")
            #print(f"current finish reason  is: {current_choice.finish_reason}")
            #print(f"current tool call  is: {current_choice.message.tool_calls}")
            messages.append(
                {"role": "assistant", "content": current_choice.message.content, "tool_calls": current_choice.message.tool_calls}

            )
            pretty_print_conversation(messages[-1])

            if current_choice.finish_reason == "tool_calls":
                tool_calls = current_choice.message.tool_calls
                
                ### missing code
                ### missing code end
                for tool_call in tool_calls:
                    print(f"single tool call  is: {tool_call}")
                    function = tool_call.function.name#tool_call["name"]
                    arguments = json.loads(tool_call.function.arguments)#tool_call["arguments"]
                    save_to_json(f"single tool call  is: {tool_call}", "crawled_data.json")

                    #print(f"functions are: {function}")
                    #print(f"arguments are: {arguments}")

                    #print(f"tools list functions are: {tools_list[function]}")
                    #print(f"tools list functions args are: {tools_list[function](**arguments)}")
                    result = tools_list[function](**arguments)  # unpack the arguments dictionary
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function,
                            "content": result,
                        }
                    )
                    pretty_print_conversation(messages[-1])

                #...............
             

            if current_choice.finish_reason == "stop":
                state = "finished"

            messages = memory_optimize(messages)

    return messages[-1]["content"]
        

# run agent to do website search -- ok
@traceable(name="#1 Website domain research")
def website_search(entity_name: str, website: str):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "scrape",
                    "description": "Scrape a URL for information",
                    "parameters":{
                        "type":"object",
                        "properties": {
                            "url": {"type": "string","description": "The URL of the webpage to scrape"}
                        },
                        "required": ["url"]
                    },
                    
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_data",
                    "description": "Save data points found for later retrieval",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "datas_update": {
                                "type": "array",
                                "description": "The data points to update",
                                "items": {
                                    "type": "object",
                                    "description": "The data point to update, should follow specific json format: {'name':'xxx','value':'xxx','reference':'xxx'}",
                                    "properties": {
                                        "name": {"type": "string","description": "The name of the data point"},
                                        "value": {"type": "string","description": "The value of the data point"},
                                        "reference": {"type": "string","description": "The reference URL of the data point"}
                                    },
                                    "required": ["name","value","reference"]
                                }
                            }
                        },
                        "required": ["datas_update"]
                    }
                }
            }
        ]

        data_keys_to_search = [obj["name"] for obj in data_points if obj["value"] is None]

        system_prompt= """
        You are a world class scraper, you are great at finding information on the internet

        You will keep scraping url based on information you recieved until information is found;

        If you cant find relavent information from the company's domain related urls,
        Whenever you found certain data points, use "update_data" function to save the data points

        You only answer questions based on results from scraper,do not make things up

        you never asks user for inuts and permissions,
        just go ahead do the best things possible without asking for permission or guidence

        """

        prompt = f"""
        Entity to search: {entity_name}

        Company Website: {website}

        Data points to search: {data_keys_to_search}
        """

        response = call_agent(prompt, system_prompt, tools, plan=True)
        return response
    
# step-2 internet search -- ok
@traceable(name="#2 Internet search")
def internet_search(entity_name: str):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the internet for information",  
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string","description": "The search query, should be semantic search query, as we are using search engine"},
                            "entity_name": {"type": "string","description": "The entity name to search for"}
                        },
                        "required": ["query","research_items","entity_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_data",
                    "description": "Save data points found for later retrieval; Only pass on data points that are found",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "datas_update": {
                                "type": "array",
                                "description": "The data points to update",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string","description": "The name of the data point"},
                                        "value": {"type": "string","description": "The value of the data point"},
                                        "reference": {"type": "string","description": "The reference of the data point"}
                                    }
                                }
                            }
                        },
                        "required": ["data_points"]
                    }
                }
            }
        ]

        data_keys_to_search = [{"name":obj["name"],"description": obj["description"]} for obj in data_points if obj["value"] is None]

        if len(data_keys_to_search) > 0:
            system_prompt = """
            You are a world class web researcher
            You will keep scraping url based on information you recieved until information is found

            You will try as hard as possible to search for all sorts of different query & source to find the information

            you do not stop until all the information is found.It is very important that we find all information
            Whenever you found certain data points, use "update_data" function to save the data points

            you only answer questions based on results from scraper, do not make things up
            you never asks user for inputs and permissions, you just do your job and provide the results;
            You only run 1 function at a time, you never run multiple functions at the same time
            """

            prompt = f"""
            Entity to search: {entity_name}
            Links we allready scraped: {links_scraped}
            Data points to find: {data_keys_to_search}
            """

            response = call_agent(prompt, system_prompt, tools, plan=False)

            return response

@traceable(name="Run research")
def run_research(entity_name,website:str):
    response1 = website_search(entity_name, website)
    response2 = internet_search(entity_name)

    return data_points


from langsmith.schemas import Run,Example
from langsmith.evaluation import evaluate

def research_eval(inputs: dict) -> dict:
    entity_name = inputs.get("entity_name")
    website = inputs.get("website") or ""
    data_points = inputs.get("data_points_to_search") or ""

    print(f"Researching about {entity_name} from {website}")

    data_points = run_research(entity_name, website,data_points)

    return data_points

def all_data_collected(run: Run, example: Example) -> dict:
    company = example.inputs.get("entity_name") or ""
    data_points = example.inputs.get("data_points_to_search") or ""
    result = run.outputs
    ground_truth = example.outputs

    system_prompt = f"""
    Yor are an critic of a research system, you are here to evaluate the results research system
    ===
    **Research task**: "Find information about a company called {company}, specifiically aroud {data_points}"
    **Results from research system**: {result}
    **Reference result from human researcher**: {ground_truth}
    ===

    Please evaluate the results from the research system, and output ONLY a JSON format as below:
    {{"all_info_found": "yes/no"}}

    all_info_found means whether they found all information requests
    (answer does not need to be exactly align with human results,
    but if answer is like Not Found or not relavent or didn't answer original question, should consider as no):
    only "yes" or "no"

    OUTPUT (Only the JSON with exact format above)
    """
    messages = [
        {"role": "user", "content": system_prompt}
    ]
    try:
        print("Evaluating the results")
        print(f"Results: {result}")
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
    except Exception as e:
        print(f"Error in evaluating the results: {e}")
        
    json_result = json.loads(result)
    all_info_found = json_result["all_info_found"]

    score = 1 if all_info_found == "yes" else 0

    return {"key": "all_info_found", "score": score}

'''
experiment_results = evaluate(
        research_eval, 
        data="web scraping research agent",#"web research agent test cases",
        evaluators=[all_data_collected],
        experiment_prefix='gpt-3.5-turbo',#"ai_research-gpt4-turbo",
        metadata={"version": "1.0.1"}
    )
'''

links_scraped = []
data_points = [
    {"name": "employees_name_and_phone_number", "value": None, "reference": None},
    {"name":"CEO_name_and_number", "value": None, "reference": None},
    {"name":"office_locations", "value": None, "reference": None},
]

entity_name = "aramco"
website = "https://www.aramco.com/en"#"https://discord.com"

#response1 = website_search(entity_name, website)
#response2 = internet_search(entity_name)

run_research(entity_name,website)

print("______")
print(f"Data points found: {data_points}")
