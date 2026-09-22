from src.utils.nodes import create_analysts
from src.utils.states import GenerateAnalystState
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

# creating the graph
# GenerateAnalystState state
# create_analysts node
# edge not needed

builder = StateGraph(GenerateAnalystState)

builder.add_node("create_analysts", create_analysts)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", END)

graph = builder.compile()
