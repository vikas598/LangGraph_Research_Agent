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

class ResearchGraphState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: NotRequired[Optional[str]] # Human feedback
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report