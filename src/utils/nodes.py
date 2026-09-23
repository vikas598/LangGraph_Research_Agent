from dotenv import load_dotenv
from .states import GenerateAnalystState
from .models import llm
from .objects import Analyst, Perspectives
from .prompts import analyst_instructions
from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt

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

def human_feedback(state: GenerateAnalystState):
    """this is where human gives feedback about the analysts"""

    feedback = interrupt({
        "question": "Are these analysts OK for you?",
        "analysts": [
            analyst.model_dump() if hasattr(analyst, "model_dump") else analyst
                 for analyst in state.get("analyst",[])
        ],
        "isntructions": "Return feedback to regenrate analyst"
        " or return empty/perfect/contunue/okay to approve  and continue the graph."
    })

    if feedback is None:
        return  {"human_analyst_feedback": None}

    if isinstance(feedback, str):
        feedback=feedback.strip()

        if feedback=="":
            return {"human_analyst_feedback": None}

        if feedback.lower() in {"perfect", "okay", "ok", "continue","next"}:
            return {"human_analyst_feedback": None}

        return {"human_analyst_feedback": feedback}

    return {"human_analyst_feedback": None}
