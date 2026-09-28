from typing_extensions import TypedDict, NotRequired, Annotated
from typing import Optional, List
from .objects import Analyst
from langgraph.graph.message import MessagesState
import operator

class GenerateAnalystState(TypedDict):
    topic: str # research topic
    max_analysts : int # number of analysts
    human_analyst_feedback : NotRequired[Optional[str]] # human feedback on what is generated
    analysts : NotRequired[List[Analyst]] # list of all the analysts 

class InterviewState(MessagesState):
    max_num_turns : int # number turns of conversation
    context : Annotated[list , operator.add] # src of docs
    analyst : Analyst # analyst created
    interview : str # transcript of the interview
    section : int # final key we duplicate in the outer state for Send() api