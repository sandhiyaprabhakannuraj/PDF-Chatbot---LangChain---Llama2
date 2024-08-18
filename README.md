

Steps involved:
1. **User uploads a PDF.**
2. **Text is extracted from the PDF.**
3. **Embeddings are generated using SentenceTransformers.**
4. **Embeddings are stored in Pinecone.**
5. **User inputs a query.**
6. **Relevant text chunks are retrieved from Pinecone.**
7. **The query and retrieved text are passed to Llama2 via LangChain.**
8. **Llama2 generates a response.**
9. **The response is shown to the user.**
![image](https://github.com/user-attachments/assets/07ae9520-9802-4cd8-a2b9-1f970ae15f80)
