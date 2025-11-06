import os
from typing import Annotated, Literal, List, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from pathlib import Path
from .export_utils import (
    generate_latex_bibliography,
    generate_word_bibliography,
    generate_markdown_bibliography,
    generate_literature_review_document
)

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
        
        # Trying multiple paths
        possible_paths = [
            Path("./data"),
            Path(__file__).resolve().parent/"data",
            Path(__file__).resolve().parent.parent/"data"
        ]

        docs = None
        for data_path in possible_paths:
            if data_path.exists():
                try:
                    docs = load_publication(pub_dir=data_path)
                    break
                except Exception as e:
                    print(f"Error loading from {data_path}: {e}")
                    continue

        if docs is None:
            raise FileNotFoundError(
                f"Could not find data folder. Tried:\n" +
                "\n".join(f"  - {p.absolute()}" for p in possible_paths)
            )
        
        _rag_assistant.add_doc(docs)
        print(f"Loaded {len(docs)} documents into knowledge base.\n")

    return _rag_assistant

        


@tool
def rag_search(query: str) -> str:
    """
    Search the internal document knowledge base for information.
    
    **ALWAYS USE THIS FIRST** for any factual questions about provided documents.
    Returns results with source, page number and section information.
    
    Args:
        query: The search query 
        
    Returns:
        Relevant document excerpts with full citations
    """
    rag = initialize_rag()


    # Expand short queries for better results
    if len(query.split()) <= 2:
        # For acronyms or short terms, expand the query
        expanded_query = f"{query} definition meaning explanation what is"
    else:
        expanded_query = query


    # Get search results with metadata
    results = rag.db.search(expanded_query, n_results=5, use_reranking=True)
    
    
    # Check if we got any results
    if not results["documents"]:
        return "No relevant information found in the documents. You may try web_search for current information."
    
    # Format the results with context
    formatted_results = []
    for i, (doc, metadata, citation, distance) in enumerate(zip(
        results["documents"], 
        results["metadatas"],
        results["citations"], 
        results["distances"]
    ), 1):
        # truncating long documents for readability
        doc_preview = doc[:500] + "..." if len(doc) > 500 else doc

        
        source = metadata.get("source", "Unknown")
        formatted_results.append(
            f"Source: {source}\n{doc}\n"
            f"**Result {i}** (Relevance: {distance:.3f})\n"
            f"Citation: {citation}\n"
            f"Content: {doc_preview}\n"
        )
    
    return "\n\n---\n\n".join(formatted_results)

@tool
def compare_documents(doc1_name: str, doc2_name: str, topic: str) -> str:
    """
    Compare information from two specific documents on a given topic.
    
    Use this when the user explicitly asks to compare two documents, 
    such as "Compare AMR and Dysmenorrhea papers" or "what's the difference
    between document A and document B on topic Y?"

    Args:
        doc1_name: Name of first document (e.g., "amr.pdf")
        doc2_name: Name of second document (e.g., "dysmenorrhea.pdf")
        topic: The topic/aspect to compare (e.g., "treatment", "causes")

    Returns:
        Side-by-side comaprison of the two documents
    """
    rag = initialize_rag()

    try: 
        # Search for topic in both documents
        query = f"{topic}"
        results = rag.db.search(query, n_results=10, use_reranking=True)

        # Filter results by document
        doc1_results = []
        doc2_results = []

        for doc, metadata, citation in zip(
            results["documents"], 
            results["metadatas"],
            results["citations"]
        ):
            source = metadata.get("source", "").lower()

            if doc1_name.lower() in source:
                doc1_results.append({
                    "content": doc[:300] + "..." if len(doc) > 300 else doc,
                    "citation": citation
                })
            elif doc2_name.lower() in source:
                doc2_results.append({
                    "content": doc[:300] + "..." if len(doc) > 300 else doc,
                    "citation": citation
                })
        

        # Format comparison output
        comparison = f"**COMPARISON: {doc1_name} vs {doc2_name} on '{topic}'**\n\n"

        comparison += f"📘 **{doc1_name}:**\n"
        if doc1_results:
            for i, result in enumerate(doc1_results[:3], 1):
                comparison += f"{i}. {result['citation']}\n{result['content']}\n\n"
        else:
            comparison += f"No relevant information found about '{topic}'\n\n"

        comparison += f"📘 **{doc2_name}:**\n"
        if doc2_results:
            for i, result in enumerate(doc2_results[:3], 1):
                comparison += f"{i}. {result['citation']}\n{result['content']}\n\n"
        else:
            comparison += f"No relevant information found about '{topic}'\n\n"
        
        return comparison
    
    except Exception as e:
        return f"Error comparing documents: {str(e)}"
    

@tool
def generate_bibliography() -> str:
    """
    Generate a formatted bibliography of all documents in the knowledge base.

    Use this when the user asks for:
    - "List of documents"
    - "Show me the bibliography"
    - "What sources do you have?"
    - "Generate a reference list"

    Returns:
        Formatted bibliography with document metadata
    """

    rag = initialize_rag()

    try:
        # Get all required documents from the vector database
        all_results = rag.db.collection.get()

        # Extracting unique documents with metadata
        seen_sources = set()
        bibliography = []

        for metadata in all_results.get("metadatas", []):
            source = metadata.get("source")

            if source and source not in seen_sources:
                seen_sources.add(source)

                # Create bibliography entry
                entry = {
                    "source": source,
                    "title": metadata.get("title", "Unknown Title"),
                    "author": metadata.get("author", "Unknown Author"),
                    "page_count": metadata.get("page_count", "N/A"),
                    "type": metadata.get("type", "N/A"),
                    "date": metadata.get("creation_date", metadata.get("date_from_filename", "N/A"))
                }
                bibliography.append(entry)
        
        # Format bibliography
        output = "BIBLIOGRAPHY\n"
        output += "="*50 + "\n\n"

        for i, entry in enumerate(bibliography, 1):
            output += f"{i}. **{entry['title']}**\n"
            output += f"    Author: {entry['author']}\n"
            output += f"    Source: {entry['source']}\n"
            output += f"    Type: {entry['type'].upper()}\n"
            output += f"    Pages: {entry['page_count']}\n\n"

            if entry['date'] != "N/A":
                output += f"    Date: {entry['date']}\n"

            output += "\n"

        output += "="*50 + "\n"
        output += f"**Total Documents: {len(bibliography)}**\n"

        return output
    except Exception as e:
        return f"Error generating bibliography: {str(e)}"


@tool
def generate_literature_review(topic: str, max_sources: int = 10) -> str:
    """
    Generate a structured academic literature review on a specific topic.

    Use when user asks to:
        - "Write a literature review on X"
        - "Summarize research on X"
        - "What does the literature say about X?"

    Args:
        topic: The research topic
        max_sources: Maximum number of sources (default: 10)

    Returns:
        Formatted literature review with citations
    """
    rag = initialize_rag()

    try:
        # Search for relevant content
        results = rag.db.search(topic, n_results=max_sources, use_reranking=True)

        if not results["documents"]:
            return f"No literature found on topic '{topic}'."
        
        # Organize by document source
        docs_content = {}
        for doc, metadata, citation in zip(
            results["documents"], 
            results["metadatas"],
            results["citations"]
        ):
            source = metadata.get("source", "Unknown")
            if source not in docs_content:
                docs_content[source] = {
                    "title": metadata.get("title", source),
                    "author": metadata.get("author", "Unknown Author"),
                    "excerpts": [],
                    "citations": []
                }
            docs_content[source]["excerpts"].append(doc[:500])
            docs_content[source]["citations"].append(citation)

        # Generate structured review
        review = f"# Literature Review: {topic}\n\n"
        review += f"**Documents Analyzed:** {len(docs_content)}\n\n"
        review += "---\n\n"

        # Introduction
        review += "## Overview\n\n"
        review += f"This review synthesizes findings from {len(docs_content)} documents "
        review += f"related to {topic}. The following sections present key findings from each source.\n\n"
        review += "---\n\n"

        # Document-by-document review
        for i, (source, content) in enumerate(docs_content.items(), 1):
            review += f"## {i}. {content['title']}\n\n"
            review += f"**Author:** {content['author']}\n"
            review += f"**Source:** {source}\n\n"

            # Add key excerpts
            review += "**Key Findings:**\n\n"
            for j, excerpt in enumerate(content['excerpts'][:3], 1):
                review += f"{j}. {excerpt}...\n\n"

            # Add citation
            review += f"**Citations:** {', '.join(set(content['citations'][:2]))}\n\n"
            review += "---\n\n"

        # Synthesis section
        review += "## Synthesis\n\n"
        review += f"Across the {len(docs_content)} reviewed sources, several key themes emerges "
        review += f"regarding {topic}. The literature provides comprehensive insights into "
        review += "various aspects of the topic, offering both theoritical frameworks and practical applications.\n\n"


        # References
        review += "## References\n\n"
        all_citations = []
        for content in docs_content.values():
            all_citations.extend(content['citations'])

        unique_citations = list(set(all_citations))
        for i, citation in enumerate(unique_citations, 1):
            review += f"{i}. {citation}\n"

        return review
    except Exception as e:
        return f"Error generating literature review: {str(e)}"
    


@tool
def export_bibliography(format: str = "word") -> str:
    """
    Export bibliography in specified format (word, latex, markdown).

    Use when user asks to:
    - "Export bibliography"
    - "Generate LaTex bibliography"
    - "Save references as markdown"

    Args:
        format: Export format - 'word', 'latex', or 'markdown' (default: 'word')

    Returns:
        Path to the exported file
    """
    rag = initialize_rag()

    try:
        # Get all documents
        all_results = rag.db.collection.get()

        # Extract unique documents
        seen_sources = set()
        bibliography = []

        for metadata in all_results.get("metadatas", []):
            source = metadata.get("source")
            if source and source not in seen_sources:
                seen_sources.add(source)
                bibliography.append({
                    "source": source,
                    "title": metadata.get("title", "Unknown Title"),
                    "author": metadata.get("author", "Unknown Author"),
                    "page_count": metadata.get("page_count", "N/A"),
                    "type": metadata.get("type", "Unknown"),
                    "date": metadata.get("creation_date", "n.d.")
                })

        
        # Sort by source
        bibliography.sort(key=lambda x: x["source"])

        # Export based on format
        if format.lower() == "latex":
            file_path = generate_latex_bibliography(bibliography)
            return f"Bibliography exported to LaTex: {file_path}"
        
        elif format.lower() == "markdown" or format.lower() == "md":
            file_path = generate_markdown_bibliography(bibliography)
            return f"Bibliography exported to Markdown: {file_path}"
        
        else: # Default to word
            file_path = generate_word_bibliography(bibliography)
            return f"Bibliography exported to Word: {file_path}"
    except Exception as e:
        return f"Error exporting bibliography: {str(e)}"
    

@tool
def export_literature_review(topic: str, format: str = "word") -> str:
    """ 
    Generate and export a complete literature review document.
    
    Use when user asks to:
    - "Export literature review on X as Word"
    - "Generate LaTex review on X"
    - "Save review about X as PDF"

    Args:
        topic: The research topic
        format: Export format - 'word', 'latex', or 'markdown' (default: 'word')
    
    Returns:
        Path to the exported document
    """
    rag = initialize_rag()

    try:
        # Search for content
        results = rag.db.search(topic, n_results=10, use_reranking=True)

        if not results.get("documents"):
            return f"No literature found on the topic: {topic}"
        
        # Organize content by source
        sections = []
        docs_content = {}

        for doc, metadata, citation in zip(
            results.get("documents", []),
            results.get("metadatas", []),
            results.get("citations", [])
        ):
            if not isinstance(metadata, dict):
                continue
            source = metadata.get("source", "Unknown")
            if source not in docs_content:
                docs_content[source] = {
                    "source": source,
                    "content": [],
                    "citations": []
                }
            docs_content[source]["content"].append(doc)
            docs_content[source]["citations"].append(citation)

        # Format sections
        for source, data in docs_content.items():
            combined_content = "\n\n".join(data["content"][:3])
            sections.append({
                "source": data["source"],
                "content": combined_content,
                "citations": list(set(data["citations"]))
            })

        # Generate document
        file_path = generate_literature_review_document(
            topic=topic,
            sections=sections,
            format=format.lower()
            )
        
        return f"Literature review exported to {format.upper()}: {file_path}"
    
    except Exception as e:
        return f"Error generating literature review: {str(e)}"                           



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
        compare_documents,
        generate_bibliography,
        generate_literature_review,
        export_bibliography,
        export_literature_review,
        web_search,
        calculator
    ]


# Defining the agent nodes
def _initialize_llm():
    """Initialize the LLM based on available API keys"""

    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        # print(f"Using Google Gemini model: {model_name}")
        return ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            model=model_name,
            temperature=0.0
        )
    elif os.getenv("GROQ_API_KEY"):
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        # print(f"Using Groq model: {model_name}")
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
        2. For comparing documents, use compare_documents
        3. For listing sources, use generate_bibliography
        4. For real-time or recent info, use web_search
        5. For math, use calculator

        **SEARCH OPTIMIZATION:**
        - For acronyms/short terms (like "AMR), expand the query (e.g., "AMR antimicrobial resistance")
        - For better results, add context (e.g., "in healthcare", "treatment options")
        - Always check if results match the intended document

        **CITATION RULES**
        - ALWAYS cite page numbers and sources in your answer
        - Format: "According to [Source: amr.pdf, Page: 5], ..."
        - When comparing, clearly state differences and similarities

        Remember: TRY RAG FIRST BEFORE WEB!""")
        messages = [system_msg] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tools_node(state: AgentState):
    """
    The agent's hands - executes the tools the agent requested.
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
            args_preview = str(tool_call["args"])[:100]
            print(f"   Args: {args_preview}")
            
            try:
                result = tool.invoke(tool_call["args"])
                
                # Show preview of result
                result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                print(f"    {result_preview}")
                
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
        print("\nAgent is selecting tools...")
        return "tools"
    
    # Otherwise, we're done - agent has provided final answer
    print("\nAgent is ready with answer")
    return "end"

# Defining the complete agentic RAG workflow
def create_agentic_rag():
    """Build the complete agentic RAG workflow"""
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
    print("=" * 50)
    print("🤖 Agentic RAG Assistant with LangGraph")
    print("=" * 50)
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
            print("-" * 25)
            
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