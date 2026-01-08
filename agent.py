import sys
!{sys.executable} -m pip install -U langchain-openai


from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage, messages_from_dict, messages_to_dict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os
import json
from datetime import datetime





document_content: list[BaseMessage]
system_prompt = SystemMessage(f"""
    You are a versatile, domain-independent AI assistant capable of handling various tasks.
    You can act as a personal assistant, research assistant, scheduler, planner, or any other role the user needs.
    you must grap the histories by reading the file.json using a function tool.but don't tell that you have read the file.json. unless it is asked.

    Available capabilities:
    - Read files (read_file) to have memory of past conversations
    - Save files (save_file)
    - Create plans and break down tasks (create_plan)
    - Create schedules and task boards (create_schedule)
    - Get current time (get_current_time)

    Instructions:
    - Understand the user's role request (e.g., "you're my personal assistant", "you're my research assistant")
    - Use appropriate tools to accomplish the task
    - For planning tasks, use create_plan to break them into chunks
    - For research tasks, use search_information
    - For scheduling tasks, use create_schedule
    - Show the current document state after modifications
    - Be proactive in using tools to complete tasks""")


def initialize_document():
    """Initializes document_content by reading from file.json if it exists."""
    global document_content
    try:
        if os.path.exists("file.json"):
            with open("file.json", "r", encoding="utf-8") as f:

                loaded_data = json.load(f)
                document_content = messages_from_dict(loaded_data)
            print(f"📖 Loaded existing document from file.json ({len(document_content)} messages)")
        else:
            document_content = []
            print("📝 Starting with empty document list")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading or parsing file.json: {e}. Initializing document_content as empty list.")
        document_content = []
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}. Initializing document_content as empty list.")
        document_content = []


# Initialize on import
initialize_document()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    chat_counter:int



@tool
def read_file(filename: str = "file.json") -> str:
    """Read content from a JSON file and update document_content.
    This function reads the entire content from the specified file,
    attempts to parse it as a JSON list of message dictionaries,
    and converts them into BaseMessage objects to update the global
    document_content.

    Args:
        filename: Name of the JSON file to read (default: file.json).

    Returns:
        A confirmation message about the file read operation.
    """
    global document_content
    print("agent is using reading file tool")

    if not filename.endswith(".json"):
        filename = f"{filename}.json"

    try:
        with open(filename, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
            if isinstance(loaded_data, list):
                document_content = messages_from_dict(loaded_data)
                return f"Successfully read and loaded {len(document_content)} messages from {filename} into document_content."
            else:
                return f"File {filename} did not contain a list of messages. document_content remains unchanged."
    except json.JSONDecodeError:
        return f"Error: Could not decode JSON from {filename}. Is it a valid JSON file with message history?"
    except FileNotFoundError:
        return f"Error: File {filename} not found."
    except Exception as e:
        return f"An unexpected error occurred while reading {filename}: {e}"




@tool
def save_file(filename: str = "file.json") -> str:
    """Save the full document_content to JSON, no filtering."""
    global document_content
    print("agent is using saving tool")

    if not filename.endswith(".json"):
        filename += ".json"

    data_to_save = messages_to_dict(document_content)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2)

    return f"Saved {len(document_content)} messages to {filename}"




@tool
def create_plan(task_description: str, chunk_size: str = "medium") -> str:
    """Create a detailed plan and break down tasks into manageable chunks.
    The generated plan is added as an AIMessage to the global document_content.

    Args:
        task_description: Description of the tasks or goals
        chunk_size: Size of chunks - "small", "medium", or "large" (default: "medium")

    Returns:
        A confirmation message about the plan creation.
    """
    global document_content
    print("agent is using creating plan tool")

    chunk_sizes = {
        "small": 3,
        "medium": 5,
        "large": 7
    }

    num_chunks = chunk_sizes.get(chunk_size.lower(), 5)

    plan_structure = f"""
PLAN: {task_description}

TASK BREAKDOWN:
"""

    for i in range(1, num_chunks + 1):
        plan_structure += f"\nChunk {i}: [Task to be defined based on requirements]"

    plan_structure += f"\n\nTotal Chunks: {num_chunks}"
    plan_structure += f"\nCreated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


    document_content.append(AIMessage(content=plan_structure))
    return f"Plan created successfully with {num_chunks} task chunks and added to document_content."




@tool
def create_schedule(items: str, time_format: str = "24h") -> str:
    """Create a schedule or task board with time slots.
    The generated schedule is added as an AIMessage to the global document_content.

    Args:
        items: Comma-separated list of tasks/items or a description of schedule items
        time_format: Time format - "12h" or "24h" (default: "24h")

    Returns:
        A confirmation message about the schedule creation.
    """
    global document_content
    print("agent is using creating schedule tool")

    if "," in items:
        task_list = [item.strip() for item in items.split(",")]
    else:

        task_list = [f"Task {i+1}" for i in range(5)]

    schedule = f"""
SCHEDULE BOARD
Generated: {datetime.now().strftime('%Y-%m-%d')}
Time Format: {time_format}

"""

    start_hour = 9
    for i, task in enumerate(task_list):
        hour = (start_hour + i) % 24
        if time_format == "12h":
            time_str = f"{hour if hour <= 12 else hour-12}:00 {'AM' if hour < 12 else 'PM'}"
        else:
            time_str = f"{hour:02d}:00"

        schedule += f"{time_str} - {task}\n"

    schedule += f"\nTotal Items: {len(task_list)}"
    schedule += f"\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


    document_content.append(AIMessage(content=schedule))
    return f"Schedule created successfully and added to document_content."




@tool
def get_current_time() -> str:
    """Get the current date and time in a formatted string.

    Returns:
        Current date and time
    """
    now = datetime.now()
    return f"Current Date and Time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


tools = [
    read_file,
    save_file,
    create_plan,
    create_schedule,
    get_current_time
]


from langchain_openai import ChatOpenAI


model = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model=MODEL_NAME,
    temperature=0.7,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Project Aether",
    },
)


def our_agent(state: AgentState) -> AgentState:
    global document_content


    if not state["messages"] or not isinstance(state["messages"][-1], ToolMessage):
        user_input = input("\n👤 USER: ").strip()
        msg = HumanMessage(content=user_input)
        state["messages"].append(msg)
        document_content.append(msg)


    all_messages = [system_prompt, SystemMessage(f"Current document content: {len(document_content)} messages.")] + list(state["messages"])
    response = model.bind_tools(tools).invoke(all_messages, stream=False)

    state["messages"].append(response)
    document_content.append(response)

    print(f"\n🤖 AI: {response.content}\n")
    return state




def should_continue(state: AgentState):
    last_human = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human = msg.content.lower()
            break

    if last_human is None:
        return "continue"

    if any(word in last_human for word in ["done", "exit", "no more", "quit"]):
        save_file.func("file.json")
        return "end"

    return "continue"

graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()

def run_document_agent():
    global document_content

    print("\n" + "=" * 60)
    print(f"📖 Current document: {len(document_content)} messages loaded from file.json")
    print("=" * 60)

    state = {
        "messages": [],
        "chat_counter": 0
    }

    while True:
        state = our_agent(state)  
        if should_continue(state) == "end":

            serializable_messages = messages_to_dict(document_content)
            with open("file.json", 'w', encoding='utf-8') as file:
                json.dump(serializable_messages, file, indent=2)
            print(f"\n💾 Document saved to file.json ({len(document_content)} messages)")
            break

    print("\n" + "=" * 60)
    print("✅ AGENT FINISHED")
    print("=" * 60)


if __name__ == "__main__":

    run_document_agent()
