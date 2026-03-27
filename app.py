import streamlit as st
from query import load_vectorstore, ask

st.set_page_config(
    page_title="MX Records RAG",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Aircraft Maintenance Records Q&A")
st.caption("Ask natural language questions about aircraft maintenance history")

# Load vector store once and cache it
@st.cache_resource
def get_vectorstore():
    with st.spinner("Loading maintenance records index..."):
        return load_vectorstore()

vectorstore = get_vectorstore()

# Sidebar — suggested questions
with st.sidebar:
    st.header("💡 Example Questions")
    example_questions = [
        "What work was done on N8050J?",
        "Were there any fuel system issues found?",
        "What ADs were complied with?",
        "Were there any magneto problems?",
        "What was the total cost of the work order?",
        "When is the next maintenance due?",
        "Were there any safety-critical findings?",
        "What parts were replaced during the annual?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.question = q

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 View source chunks"):
                for i, chunk in enumerate(message["sources"], 1):
                    m = chunk.metadata
                    st.markdown(
                        f"**Source {i}** | "
                        f"Aircraft: `{m.get('aircraft_registration', 'N/A')}` | "
                        f"WO: `{m.get('work_order', 'N/A')}` | "
                        f"Date: `{m.get('date', 'N/A')}`"
                    )
                    st.text(chunk.page_content[:300] + "...")
                    st.divider()

# Question input — supports both sidebar buttons and typed input
question = st.chat_input("Ask a question about the maintenance records...")

# Handle sidebar button clicks
if "question" in st.session_state and st.session_state.question:
    question = st.session_state.question
    st.session_state.question = None

if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching maintenance records..."):
            result = ask(question, vectorstore)

        st.markdown(result["answer"])

        with st.expander("📄 View source chunks"):
            for i, chunk in enumerate(result["sources"], 1):
                m = chunk.metadata
                st.markdown(
                    f"**Source {i}** | "
                    f"Aircraft: `{m.get('aircraft_registration', 'N/A')}` | "
                    f"WO: `{m.get('work_order', 'N/A')}` | "
                    f"Date: `{m.get('date', 'N/A')}`"
                )
                st.text(chunk.page_content[:300] + "...")
                st.divider()

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })