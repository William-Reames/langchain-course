from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()
# tavily_client = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches the internet based on a provided query.
#     Args:
#         query: The query to search for.
#     Returns:
#         The search result as a string.
#     """
#     print(f"Searching for {query}...")
#     return tavily_client.search(query=query)


def run_search_agent(query: str) -> str:
    llm = ChatOllama(model="mistral")
    tools = [TavilySearch()]
    agent = create_agent(model=llm,tools=tools)
    result = agent.invoke({"messages": HumanMessage(content=query)})
    return result['messages'][-1].content  # Return the final result from the search

if __name__ == "__main__":
    print("Starting script...")

    query = "Please search for 3 job postings for a remote-work AI engineer using LangChain on LinkedIn. Please provide details and links for each posting."

    print(run_search_agent(query))