from dotenv import load_dotenv
from .states import GenerateAnalystState, InterviewState
from .models import llm
from .objects import Analyst, Perspectives, SearchQuery
from .prompts import analyst_instructions, question_instructions, search_instructions, answer_instructions
from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from langchain_tavily import TavilySearch
from langchain_core.messages import get_buffer_string

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
                 for analyst in state.get("analysts", [])
        ],
        "instructions": "Return feedback to regenrate analyst"
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

def generate_question(state: InterviewState):
    """Node to generate questions"""

    #get state analyst
    analyst = state["analyst"]

    if isinstance(analyst, dict):
        analyst = analyst.model_validate(analyst)

    messages = state["messages"]

    # generate questions
    system_message = question_instructions.format(goals=analyst.persona)
    questions = llm.invoke([SystemMessage(content=system_message)]+messages)

    return {"messages":[questions]}

def search_web(state: InterviewState):
    """retrieve docs from web"""

    # search query
    structured_llm = llm.with_structured_output(SearchQuery)

    # search instruction
    search_instruction_system_message = SystemMessage(content=search_instructions)
    tavily_search = TavilySearch(max_results = 3)

    search_query = structured_llm.invoke([search_instruction_system_message]+state["messages"])

    # web search
    data = tavily_search.invoke({"querys":search_query.search_query})
    search_docs = data.get("results", data)

    #format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href = "{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

def search_web_2(state: InterviewState):
    """retrieve docs from web"""

    # search query
    structured_llm = llm.with_structured_output(SearchQuery)

    # search instruction
    search_instruction_system_message = SystemMessage(content=search_instructions)
    tavily_search = TavilySearch(max_results = 3)

    search_query = structured_llm.invoke([search_instruction_system_message]+state["messages"])

    # web search
    data = tavily_search.invoke({"querys":search_query.search_query})
    search_docs = data.get("results", data)

    #format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href = "{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

def generate_answer(state: InterviewState):
    """node to answer a query"""

    # get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    #answer question
    system_message = SystemMessage(content=answer_instructions.format(goals=analyst.persona, context = context))
    answer= llm.invoke([system_message]+messages)

    # name the msg as coming from the expert
    answer.name = "expert"

    return {"message":[answer]}

def save_interview(state: InterviewState):
    """save interview"""

    messages = state["messages"]

    interview = get_buffer_string(messages)

    return {"interview":interview}