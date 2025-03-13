# Bulking Meals Chatbot

A RAG-powered conversational AI assistant that provides personalized high-protein meal recommendations for muscle building and fitness nutrition.



## Features

- **Personalized Recommendations**: Get tailored meal suggestions based on your preferences and dietary needs
- **Recipe Instructions**: Detailed cooking instructions for all recommended meals
- **Nutritional Information**: Protein content and other nutritional details for informed meal planning
- **Meal Planning**: Assistance with weekly meal planning for fitness goals
- **RAG-powered**: Utilizes Retrieval Augmented Generation for accurate, up-to-date information

## Installation

1. Clone this repository:


git clone https://github.com/jordan-kesner/Fitness-RAG-Bot.git cd Fitness-RAG-Bot

2. Create a virtual environment:

   python -m venv venv source venv/bin/activate
   # On Windows: venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

4. Set up your environment variables:

cp .env.example .env

Edit the `.env` file and add your OpenAI API key

## Usage

1. Run the Streamlit app:

streamlit run app2.py

2. Open your browser at `http://localhost:8501`

3. Start chatting with the assistant about high-protein meals and fitness nutrition!

## Building Your Own Knowledge Base

This chatbot uses a vector database to store and retrieve information. To build your own knowledge base:

1. Place text documents in the `data/documents` folder
2. Add URLs to the `knowledge_base_builder.ipynb` notebook
3. Run the notebook to process and vectorize your content

## Project Structure

- `app.py`: Main Streamlit application
- `rag_pipeline.py`: Core RAG implementation for conversational AI
- `preprocess_documents.py`: Utilities for document processing and vectorization
- `batch_processor.py`: Process large document sets efficiently
- `requirements.txt`: Project dependencies
- `.env.example`: Template for environment variables

## Technologies Used

- Streamlit for the web interface
- LangChain for the RAG pipeline
- OpenAI for embeddings and language model
- Chroma for vector storage

Email: jordankesner@proton.me
