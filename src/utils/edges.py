from dotenv import load_dotenv
from src.utils.states import GenerateAnalystState, InterviewState
from typing import Literal
from langgraph.graph import END
from typing import Literal
from langchain.messages import AIMessage

load_dotenv()

# conditional edge
def should_continue(state: GenerateAnalystState)->Literal["create_analysts", END]:
    """Return the next node to execute"""

    human_analyst_feedback = state.get("human_analyst_feedback", None)

    if human_analyst_feedback:
        return "create_analysts"

    return END

def route_messages(state:InterviewState, name:str ="expert"):
    """routes between question and answer"""

    #Get Messages
    messages = state["messages"]
    max_num_turns = state.egt("max_num_turns",2)

    # check the num of expert answers
    num_response = len(m for m in messages if isinstance(m, AIMessage) and m.name==name)

    if num_response>=max_num_turns:
        return "save_interview"

    return "ask_question"

