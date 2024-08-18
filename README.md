User uploads a PDF.
Text is extracted from the PDF.
Embeddings are generated using SentenceTransformers.
Embeddings are stored in Pinecone.
User inputs a query.
Relevant text chunks are retrieved from Pinecone.
The query and retrieved text are passed to Llama2 via LangChain.
Llama2 generates a response.
The response is shown to the user.
