# # llm_rag.py
# import os
# import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")


# def generate_answer(query, retrieved_docs):
#     context = "\n\n".join([doc["content"] for doc in retrieved_docs])
#     prompt = f"Answer the following question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

#     try:
#         # response = openai.ChatCompletion.create(
#         #     # model="gpt-4",
#         #     model="GPT-3.5",
#         #     messages=[{"role": "user", "content": prompt}],
#         #     temperature=0.3,
#         # )

#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",  
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#         )
#         return response['choices'][0]['message']['content'].strip()
#     except Exception as e:
#         return f"Error generating answer: {e}"


import json
import requests

class LLMRag:
    def __init__(self, vector_db_manager, llm_api_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key="):
        self.vector_db_manager = vector_db_manager
        self.llm_api_url = llm_api_url
        self.api_key = "" # As per instructions, API key is empty. Canvas will provide at runtime.

    async def generate_llm_response(self, prompt: str) -> str:
        """
        Generates a response from the LLM based on the given prompt.
        Uses the Gemini API via a direct fetch call.
        """
        chat_history = []
        chat_history.append({"role": "user", "parts": [{"text": prompt}]})
        
        payload = {"contents": chat_history}
        
        try:
            # Using requests for synchronous call in Flask context
            # In a truly async Flask app (e.g., with Quart), you'd use aiohttp
            response = requests.post(
                self.llm_api_url + self.api_key,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            
            result = response.json()
            
            if result.get('candidates') and len(result['candidates']) > 0 and \
               result['candidates'][0].get('content') and \
               result['candidates'][0]['content'].get('parts') and \
               len(result['candidates'][0]['content']['parts']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"LLM response structure unexpected: {result}")
                return "Sorry, I couldn't generate a coherent response."
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            return f"Error communicating with AI: {e}"
        except Exception as e:
            print(f"An unexpected error occurred during LLM call: {e}")
            return "An unexpected error occurred."


    async def answer_question_with_rag(self, user_query: str) -> str:
        """
        Performs Retrieval Augmented Generation (RAG).
        1. Queries vector DB for relevant chunks.
        2. Constructs a prompt with the query and retrieved context.
        3. Calls the LLM to generate an answer.
        """
        # 1. Retrieve relevant chunks from the vector database
        retrieved_chunks_info = self.vector_db_manager.query_chunks(user_query, n_results=5)

        if not retrieved_chunks_info:
            return "I couldn't find any relevant information in the knowledge base to answer your question."

        context_texts = [info["document"] for info in retrieved_chunks_info]
        
        # 2. Construct the prompt for the LLM
        # Instruct the LLM to use the provided context
        prompt = f"""
        You are an AI assistant that answers questions based *only* on the provided context.
        If the answer cannot be found in the context, state that you don't have enough information.

        Context:
        {'-' * 20}
        {"\n\n".join(context_texts)}
        {'-' * 20}

        Question: {user_query}

        Answer:
        """
        
        # 3. Call the LLM to generate the answer
        llm_answer = await self.generate_llm_response(prompt)
        return llm_answer