import os
from typing import TypedDict, Annotated

import operator
import psycopg

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq
from tools.flight_tool import search_flights
from tools.tavily_tool import tavily_search
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model='llama-3.3-70b-versatile')
DATABASE_URL = os.getenv("DATABASE_URL")

# Define a TypedDict for the travel state
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinery: str
    llm_calls: int

def flight_agent(state: TravelState):
    query = state["user_query"]
    flights_data = search_flights(query)
    return {
        'flight_results': flights_data,
        'messages': [
            AIMessage(content=f"Flight search results for '{query}':\n{flights_data}")
        ],
        'llm_calls': state.get('llm_calls', 0) + 1
    }

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state["user_query"]}"
    hotels_data = tavily_search(query)

    return {
        'hotel_results': hotels_data,
        'messages': [
            AIMessage(content=f"Hotel search results for '{query}':\n{hotels_data}")
        ],
        'llm_calls': state.get('llm_calls', 0) + 1
    }

def itinery_agent(state: TravelState):
    prompt = f"""
create a travel itinary based on the following flight and hotel results:
User Query:
{state["user_query"]}
Flight Results:
{state["flight_results"]}
Hotel Results:
{state["hotel_results"]}
"""
    response = llm.invoke([
        SystemMessage(content="You are a travel assistant that creates travel itineraries."),
        HumanMessage(content=prompt)
    ])
    return {
        'itinery': response.content,
        'messages': [
            AIMessage(content=f"Generated travel itinerary:\n{response.content}")
        ],
        'llm_calls': state.get('llm_calls', 0) + 1
    }
    
def final_agent(state: TravelState):
    final_prompt = f"""
    Generate final travel response.
    Flights:
    {state['flight_results']}
    Hotels:
    {state['hotel_results']}
    Itinerary:
    {state['itinerary']}
    """
    response = llm.invoke([
        HumanMessage(content=final_prompt)
    ])
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

graph = StateGraph(TravelState)
# add nodes
graph.add_node('flight_agent', flight_agent)
graph.add_node('hotel_agent', hotel_agent)
graph.add_node('itinery_agent', itinery_agent)
graph.add_node('final_agent', final_agent)

# add edges
graph.add_edge(START, 'flight_agent')
graph.add_edge('flight_agent', 'hotel_agent')
graph.add_edge('hotel_agent', 'itinery_agent')
graph.add_edge('itinery_agent', 'final_agent')
graph.add_edge('final_agent', END)

_conn = psycopg.connect(DATABASE_URL)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

workflow = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config ={
        'configurable': {
            'thread_id': 'travel_workflow',
        }
    }
    user_query = input("Enter your travel query: ")

    result = workflow.invoke({
        'messages': [HumanMessage(content=user_query)],
        'user_query': user_query,
        'flight_results': '',
        'hotel_results': '',
        'itinerary': '',
        'llm_calls': 0
    })
    print('Final response:')

    for msg in result["messages"]:
        print(msg.content)
