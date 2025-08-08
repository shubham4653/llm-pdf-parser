import os
import requests
import io
import hashlib
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# --- AI & DB Libraries ---
import google.generativeai as genai
from pinecone import Pinecone
import pypdf

# --- Pydantic Models for type validation ---
class HackRxRequest(BaseModel):
    documents: str
    questions: list[str]

class HackRxResponse(BaseModel):
    answers: list[str]

# --- Initialize FastAPI App ---
app = FastAPI(title="HackRx RAG API")

# --- Configure AI/DB Clients ---
# This block now correctly loads the keys from the environment variables you set.
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
except KeyError as e:
    print(f"FATAL: Environment variable {e} not found. Please set it before running.")
    exit()

# --- Global Constants and Model Initialization ---
AUTH_TOKEN = "0b1f88435b5f65b252b2d6939de3cb1db86af17a1609045bae3cb1db86af17a1609045bae3cb7295ecce63c"
GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash-latest') # Correct model name
EMBEDDING_MODEL = "models/embedding-001"
PINECONE_INDEX_NAME = "hackrx-index"

# --- Connect to Pinecone Index ---
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"FATAL: Pinecone index '{PINECONE_INDEX_NAME}' does not exist. Please create it first.")
    exit()
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# --- Core Application Functions ---

def get_and_process_document(doc_url: str):
    """
    Fetches a document from a URL, chunks it, and indexes its embeddings in Pinecone.
    """
    doc_id = hashlib.sha256(doc_url.encode()).hexdigest()

    # Checks if the document has been processed before to avoid duplicate work.
    if pinecone_index.fetch(ids=[f"{doc_id}-0"]).vectors:
        print(f"Document {doc_url} is already indexed. Skipping.")
        return

    print(f"Processing new document: {doc_url}")
    # 1. Fetch PDF from URL
    response = requests.get(doc_url)
    response.raise_for_status()
    pdf_file = io.BytesIO(response.content)
    pdf_reader = pypdf.PdfReader(pdf_file)
    
    # 2. Extract Text and create chunks (one chunk per page)
    text_chunks = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
    if not text_chunks:
        print(f"Warning: Could not extract any text from document {doc_url}.")
        return

    # 3. Generate Embeddings for all chunks
    response = genai.embed_content(model=EMBEDDING_MODEL, content=text_chunks, task_type="retrieval_document")
    doc_embeddings = response['embedding']

    # 4. Prepare vectors for Pinecone upsert
    vectors_to_upsert = [
        {
            "id": f"{doc_id}-{i}",
            "values": embedding,
            "metadata": {"doc_url": doc_url, "text": chunk}
        }
        for i, (chunk, embedding) in enumerate(zip(text_chunks, doc_embeddings))
    ]

    # 5. Upsert vectors to Pinecone
    pinecone_index.upsert(vectors=vectors_to_upsert, batch_size=100) # Use batching for efficiency
    print(f"Successfully indexed {len(vectors_to_upsert)} chunks.")


def retrieve_context_from_pinecone(question: str) -> str:
    """
    Generates an embedding for a question and retrieves relevant text from Pinecone.
    """
    # 1. Generate embedding for the user's question
    response = genai.embed_content(model=EMBEDDING_MODEL, content=question, task_type="retrieval_query")
    question_embedding = response['embedding']

    # 2. Query Pinecone for the top 3 most relevant text chunks
    query_result = pinecone_index.query(
        vector=question_embedding,
        top_k=3,
        include_metadata=True
    )

    # 3. Combine the text from the results to form the context
    context = "\n".join([item['metadata']['text'] for item in query_result['matches']])
    return context

# --- API Endpoint Definition ---

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest, authorization: str | None = Header(None)):
    print(f"Received Authorization Header: '{authorization}'")
    
    
    """
    Main API endpoint that orchestrates the RAG pipeline.
    """
    # 1. Authenticate the request
    if not authorization or authorization.split(" ")[1] != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    try:
        # 2. Ensure the document is processed and indexed
        get_and_process_document(request.documents)

        # 3. For each question, retrieve context and generate an answer
        final_answers = []
        for question in request.questions:
            retrieved_context = retrieve_context_from_pinecone(question)

            prompt = f"""
            Answer the following question strictly based on the provided context.
            If the answer is not in the context, say 'Information not available in the provided context'.

            Context:
            ---
            {retrieved_context}
            ---

            Question: {question}
            """

            # Generate the final answer using Gemini
            response = GEMINI_MODEL.generate_content(prompt)
            final_answers.append(response.text.strip())
        
        return HackRxResponse(answers=final_answers)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "HackRx RAG API is running."}