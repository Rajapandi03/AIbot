import streamlit as st
from typing import List, Dict
from langgraph.graph import StateGraph, START, END
try:
    from langchain_ollama.llms import OllamaLLM
except ModuleNotFoundError:
    st.error("Required module 'langchain_ollama' is not installed. Please install it using 'pip install langchain-ollama'.")
    raise

# Step 1: Define State
class State(Dict):
    messages: List[Dict[str, str]]

# Step 2: Initialize StateGraph
graph_builder = StateGraph(State)

# Initialize the LLM
llm = OllamaLLM(model="llama3.2")

# Define chatbot function
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    state["messages"].append({"role": "assistant", "content": response})  # Treat response as a string
    return {"messages": state["messages"]}

# Add nodes and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# Streamlit UI
st.title("Chatbot using LangGraph & Ollama")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "", key="user_input")
if st.button("Send") and user_input:
    state = {"messages": st.session_state.chat_history + [{"role": "user", "content": user_input}]} 
    event = next(graph.stream(state))  
    assistant_response = list(event.values())[0]["messages"][-1]["content"]
    
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    st.write(f"Assistant: {assistant_response}")
    
# Display chat history
for msg in st.session_state.chat_history:
    role = "You:" if msg["role"] == "user" else "Assistant:"
    st.write(f"{role} {msg['content']}")
