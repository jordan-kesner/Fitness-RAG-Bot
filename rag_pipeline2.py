import os
import dotenv
from time import time
import streamlit as st


# pip install docx2txt, pypdf
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

dotenv.load_dotenv()



# Function to stream the response of the LLM 
def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})



def load_chroma_vectorstore(persist_directory):
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_db


def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(llm):
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = load_chroma_vectorstore(persist_directory=os.getenv("CHROMA_PERSIST_DIR"))
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a knowledgeable and enthusiastic muscle-building assistant focused on nutrition and fitness. Your primary task is to recommend high-protein meals tailored specifically for muscle growth, strength enhancement, and recovery.

Guidelines for your responses:

- Prioritize recommending meals that are high in protein, balanced in carbohydrates and healthy fats, and aligned with the user's dietary preferences (e.g., vegetarian, vegan, keto, or general omnivore).

Interaction instructions:

- **General Queries:** Respond normally and informatively.
- **Ingredient-Based Queries ("I have x ingredients, what can I make?"):** Provide a bulleted list of possible recipes that match the ingredients listed.
- **Recipe Selection Queries ("How do I make [recipe]?" or if a user selects a recipe):** Provide a structured response containing:
    1. A brief summary of the dish
    2. Bulleted ingredients list (with quantities)
    3. Numbered step-by-step preparation instructions
    4. Additional tips highlighting nutritional value and benefits
- **Meal-Time Queries ("What can I make for breakfast/lunch/dinner?"):** Provide a bulleted list of meal options suitable for that specific meal time.

General guidelines for all responses:

- Maintain an encouraging, informative, and supportive tone, motivating users to stay consistent with their muscle-building goals.
- Prioritize bulleted or numbered lists to maintain clarity.
- Provide additional fitness and nutrition insights when appropriate, including meal timing advice (pre-workout, post-workout, rest days), portion control tips, and ingredient substitutions or adjustments.

Example response for recipe selection:

**Meal Name:** Grilled Chicken with Quinoa and Broccoli

**Summary:**
A nutritious and balanced meal ideal for muscle recovery and growth, combining lean protein from chicken, complex carbohydrates from quinoa, and fiber-rich broccoli.

**Ingredients:**
- 200g grilled chicken breast
- 1 cup cooked quinoa
- 1 cup steamed broccoli
- 1 tablespoon olive oil
- Salt, pepper, garlic powder to taste

**Preparation Steps:**
1. Season chicken breast with salt, pepper, and garlic powder.
2. Cook chicken in olive oil over medium heat until fully cooked (about 6-8 minutes per side).
3. Steam broccoli until tender (about 5-6 minutes).
4. Cook quinoa according to package instructions.
5. Combine chicken, broccoli, and cooked quinoa on a plate.

**Additional Tips:**
- Consume this meal within an hour after your workout to optimize muscle recovery.
- This meal provides approximately 45g of protein, supporting muscle synthesis and repair.
- Stay hydrated and maintain consistent protein intake throughout the day for best results.

{context}\n
"""),
    MessagesPlaceholder(variable_name="messages"),
    ("user", "{input}"),
])


    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    
    # Get the content of the last message, handling both dict and object formats
    if isinstance(messages[-1], dict):
        last_message_content = messages[-1]["content"]
    else:
        # Try accessing as an attribute
        try:
            last_message_content = messages[-1].content
        except AttributeError:
            # Add debugging information
            raise TypeError(f"Unsupported message format. Expected dict or object with content attribute, got {type(messages[-1])}. Full message: {messages[-1]}")
    
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": last_message_content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})