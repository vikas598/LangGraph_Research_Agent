from typing_extensions import TypedDict, NotRequired
from typing import Optional, List
from .objects import Analyst

class GenerateAnalystState(TypedDict):
    topic: str # research topic
    max_analysts : int # number of analysts
    human_analyst_feedback : NotRequired[Optional[str]] # human feedback on what is generated
    analysts : NotRequired[List[Analyst]] # list of all the analysts 