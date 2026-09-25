from src.utils.nodes import create_analysts, human_feedback
from src.utils.states import GenerateAnalystState
from src.utils.edges import should_continue
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

# creating the graph
# GenerateAnalystState state
# create_analysts node
# edge not needed

builder = StateGraph(GenerateAnalystState)

builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue)
# builder.add_edge(should_continue, END)

graph = builder.compile()
