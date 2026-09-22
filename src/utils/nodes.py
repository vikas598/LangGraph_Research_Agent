from dotenv import load_dotenv
from .states import GenerateAnalystState
from .models import llm
from .objects import Analyst, Perspectives
from .prompts import analyst_instructions
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()


#nodes
def create_analysts(state: GenerateAnalystState):
    """Create Analysts"""

    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    # enforcing structured output
    structured_llm = llm.with_structured_output(Perspectives)

    #system message 
    system_message = analyst_instructions.format(topic=topic,
                                                 human_analyst_feedback=human_analyst_feedback,
                                                 max_analysts=max_analysts)

    # create analyst
    analysts = structured_llm.invoke([SystemMessage(content=system_message)]+ [HumanMessage(content="Please generate the list of analysts.")])

    return {"analysts": analysts.analysts}
    