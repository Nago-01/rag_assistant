"""
Agentic RAG Assistant using LangGraph
This module transforms the basic RAG assistant into an agentic system that can:
- Decide when to use RAG vs other tools
- Use web search for current information
- Perform calculations
- Maintain conversation memory
"""

import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain_community.tools.tavily_search import TavilySearchResults

# Import your existing RAG components
from .app import QAAssistant, load_publication

# Load environment variables
load_dotenv()

# Supressing gRPC ALTS warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"


# Defining the state structure for the agent                        
class AgentState(TypedDict):
    """The agent's memory - tracks conversation and context"""
    messages: Annotated[list, add_messages]


# Defining tools 
# Create a global RAG assistant 
_rag_assistant = None

def initialize_rag():
    """Initialize the RAG assistant once"""
    global _rag_assistant
    if _rag_assistant is None:
        print("\nInitializing RAG knowledge base...")
        _rag_assistant = QAAssistant()
        docs = load_publication()
        _rag_assistant.add_doc(docs)
        print(f"✓ Loaded {len(docs)} documents into knowledge base\n")
    return _rag_assistant


@tool
def rag_search(query: str) -> str:
    """
    Search the internal document knowledge base for information.
    
    **ALWAYS USE THIS FIRST** for any factual questions that could be answered 
    from documents about topics like: medical conditions, research papers, 
    company information, or any content in the uploaded documents.
    
    Args:
        query: The search query to find relevant information in documents
        
    Returns:
        Relevant document excerpts that answer the query
    """
    rag = initialize_rag()
    
    # Get search results with metadata
    results = rag.db.search(query, n_results=5)
    
    # Check if we got any results
    if not results["documents"]:
        return "No relevant information found in the documents. You may try web_search for current information."
    
    # Format the results with context
    formatted_results = []
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"], 
        results["metadatas"], 
        results["distances"]
    ), 1):
        source = metadata.get("source", "Unknown")
        formatted_results.append(f"[Source: {source}]\n{doc}")
    
    return "\n\n---\n\n".join(formatted_results)


@tool
def web_search(query: str) -> str:
    """
    Search the web for current, real-time information.
    
    **ONLY USE THIS** when:
    - rag_search returns "No relevant information found"
    - Question is about very recent events (news from last few days)
    - Question needs real-time data (weather, stock prices, current events)
    
    Args:
        query: The search query
        
    Returns:
        Search results from the web
    """
    if not os.getenv("TAVILY_API_KEY"):
        return "Web search is not available. Please set TAVILY_API_KEY in .env file."
    
    try:
        # Initialize Tavily search tool
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        # Execute the search
        search_tool = tavily_client.search(
            query=query,
            max_results=3,
            include_answer=False,
            include_raw_content=False,
            search_depth="advanced"
        )

        # Perform the search
        results = search_tool["results"]
        
        # Handling different result formats
        if isinstance(results, str):
            # if results come as a single string already
            return results
        elif isinstance(results, list):
            # if it's a list of dictionaries
            formatted = []
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    content = result.get("content", result.get("snippet", ""))
                    url = result.get("url", "")
                    formatted.append(f"{i}. {content}\nSource: {url}")
                else:
                    formatted.append(f"{i}. {str(result)}")
            return "\n\n".join(formatted) if formatted else "No results found."
        elif isinstance(results, dict):
            # if it's a single dictionary
            content = results.get("content", results.get("snippet", ""))
            url = results.get("url", "")
            return f"{content}\nSource: {url}"
        else:
            # If it's unknown format, convert to string
            return str(results)
    except Exception as e:
        return f"Error during web search: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.
    Use this when you need to compute numbers, percentages, or do math.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "100 * 0.15", "(500 + 300) / 2")
        
    Returns:
        The calculated result
    """
    try:
        # Use a safer evaluation method in production
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


def get_tools():
    """Return all available tools for the agent"""
    return [
        rag_search,
        web_search,
        calculator
    ]


# Defining the agent nodes
def _initialize_llm():
    """Initialize the LLM based on available API keys"""

    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        print(f"Using Google Gemini model: {model_name}")
        return ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            model=model_name,
            temperature=0.0
        )
    elif os.getenv("GROQ_API_KEY"):
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        print(f"Using Groq model: {model_name}")
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=model_name,
            temperature=0.0
        )
    else:
        raise ValueError(
            "No valid API key found. "
            "Please set GROQ_API_KEY or GEMINI_API_KEY in your .env file"
        )


def llm_node(state: AgentState):
    """
    The agent's brain - decides whether to use tools or respond directly.
    
    This node:
    1. Receives the current conversation state
    2. Analyzes what the user is asking
    3. Decides if it needs tools (RAG, web search, calculator)
    4. Either calls tools OR provides final answer
    """
    llm = _initialize_llm()
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    # ALWAYS include system message for context
    messages = state["messages"]
    
    # Add system message at the beginning if not present
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        system_msg = SystemMessage(content="""You are a helpful AI assistant with access to tools. Follow these rules STRICTLY:

        **TOOL PRIORITY:**
        1. For ANY factual question, ALWAYS use rag_search FIRST
        2. Only use web_search if rag_search explicitly returns "No relevant information found"
        3. Use calculator for any mathematical computations

        **CONVERSATION CONTEXT:**
        - You have full access to the conversation history
        - When user says "it", "that", "this" - refer to the previous topic discussed
        - Maintain context across multiple turns

        **RESPONSE GUIDELINES:**
        - Base ALL answers on tool results ONLY
        - If rag_search finds information, use it - don't search the web
        - Be concise and direct
        - If information is insufficient, clearly state what's missing

        Remember: TRY RAG FIRST BEFORE WEB!""")
        messages = [system_msg] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tools_node(state: AgentState):
    """
    The agent's hands - executes the tools the agent requested.
    
    This node:
    1. Gets the tool calls from the LLM's last message
    2. Executes each tool with the provided arguments
    3. Returns results back to the agent
    """
    tools = get_tools()
    tool_registry = {tool.name: tool for tool in tools}
    
    last_message = state["messages"][-1]
    tool_messages = []
    
    # Execute each tool the agent requested
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool = tool_registry.get(tool_name)
        
        if tool:
            print(f"\nExecuting {tool_name}")
            print(f"   Query: {tool_call['args'].get('query', tool_call['args'].get('expression', ''))}")
            
            try:
                result = tool.invoke(tool_call["args"])
                
                # Show preview of result
                result_preview = str(result)[:150] + "..." if len(str(result)) > 150 else str(result)
                print(f"   ✓ Preview: {result_preview}")
                
                tool_messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_name
                ))
            except Exception as e:
                print(f"   ✗ Error: {e}")
                tool_messages.append(ToolMessage(
                    content=f"Error executing {tool_name}: {str(e)}",
                    tool_call_id=tool_call["id"],
                    name=tool_name
                ))
        else:
            print(f"\n⚠️{tool_name} not found.")
            tool_messages.append(ToolMessage(
                content=f"Error: '{tool_name}' not found",
                tool_call_id=tool_call["id"],
                name=tool_name
            ))
    
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Decision function - determines if agent should use tools or provide final answer.
    
    Returns:
        "tools" if agent wants to use tools
        "end" if agent is ready to provide final answer
    """
    last_message = state["messages"][-1]
    
    # If LLM made tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("\n🤔 Agent is selecting tools...")
        return "tools"
    
    # Otherwise, we're done - agent has provided final answer
    print("\n✅ Agent is ready with answer")
    return "end"

# Defining the complete agentic RAG workflow
def create_agentic_rag():
    """
    Build the complete agentic RAG workflow.
    
    The graph structure:
    START → [LLM] → Decision?
                ↑      ↓
                |   Use tools?
                |      ↓
                └── [Tools]
                
            No tools needed?
                ↓
              END
    """
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tools_node)
    
    # Set entry point
    graph.set_entry_point("llm")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After using tools, go back to LLM for reasoning
    graph.add_edge("tools", "llm")
    
    # Compile the graph
    return graph.compile()


# Main
def main():
    """Main function to run the agentic RAG assistant"""
    print("=" * 60)
    print("🤖 Agentic RAG Assistant with LangGraph")
    print("=" * 60)
    print("\nInitializing agent...")
    
    try:
        # Create the agent
        agent = create_agentic_rag()
        print("\n✓ Agent initialized successfully!")
        print("\n📚 It can:")
        print("  • Search internal documents (RAG)")
        print("  • Search the web for current info")
        print("  • Perform calculations")
        print("  • Remember conversations")
        print("\n💡 And it will automatically choose the best tool to use")
        print("\nType 'exit', or 'x' to quit.\n")
        
        # Maintain conversation state across turns
        conversation_messages = []
        
        # Interactive loop
        while True:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            # Check for exit commands BEFORE processing
            if user_input.lower() in ["exit", "x"]:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Add user message to conversation
            conversation_messages.append(HumanMessage(content=user_input))
            
            # Create state with full conversation history
            current_state = {
                "messages": conversation_messages.copy()
            }
            
            print("\n🔄 Processing...")
            print("-" * 40)
            
            try:
                # Invoke the agent with conversation history
                result = agent.invoke(current_state)
                
                # Update conversation with all new messages (tool calls, results, final answer)
                conversation_messages = result["messages"]
                
                # Get the final response (last message)
                final_message = result["messages"][-1]
                
                print("-" * 60)
                print(f"\n🤖 Assistant: {final_message.content}\n")
                
            except Exception as e:
                error_msg = str(e)
                print(f"\n❌ Error during processing: {error_msg}")
                
                # Provide helpful error context
                if "tool_use_failed" in error_msg or "tool call validation" in error_msg:
                    print("\n💡 This might be a model-specific issue. Try:")
                    print("  - Using Gemini: Set GEMINI_API_KEY in .env")
                    print("  - Or continue with next question")
                else:
                    print("Continuing conversation...\n")
                
                # Remove the failed user message to prevent cascading errors
                if conversation_messages:
                    conversation_messages.pop()
            
    except KeyboardInterrupt:
        print("\n\n👋Goodbye!")
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set GROQ_API_KEY or GEMINI_API_KEY in .env")
        print("  2. (Optional) Set TAVILY_API_KEY for web search")
        print("  3. Documents in the 'data' folder")


if __name__ == "__main__":
    main()