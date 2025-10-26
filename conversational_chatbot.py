# Imports
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage


from facial import load_model, video_to_base64_frames, video_predict
from speech import load_model, speech_predict, convert_wav_to_base64


# Load your trained FER model
emotion_model = load_model("models/fer_model/model.h5")
frames_list = video_to_base64_frames("sample_video.mp4", max_frames=30)
initial_emotion = video_predict(emotion_model, frames_list)
initial_emotion = str(initial_emotion)   # ensure it’s a string

# # Load the trained SER model
# speech_emotion_model = load_model("models/ser_model/ser_model.keras")
# audio = convert_wav_to_base64("sample_audio.wav")
# speech_emotion = speech_predict(audio)
# speech_emotion = str(speech_emotion)





# Suppress warnings and set env variable
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

class AstroAssistant:
    def __init__(self, file_path="text.txt", model_name="gemma3:1b"):
        """
        Initializes the chatbot, setting up the vector store, retriever, and LLM chain.
        This setup runs only once when the class is instantiated.
        """
        self.vector_db = self._create_vector_db(file_path)
        self.retriever = self.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        self.llm = ChatOllama(model=model_name)
        self.chat_history = []
        self.prompt = self._create_prompt()
        self.chain = self._create_chain()
        

    def _create_vector_db(self, file_path):
        """Loads text, splits it, and creates a Chroma vector database."""
        loader = TextLoader(file_path=file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        chunks = text_splitter.split_documents(data)
        return Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text:latest"),
            collection_name="local-rag-conversational"
        )

    def _format_docs(self, retrieved_docs):
        """Formats retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in retrieved_docs)
        
    def _format_history(self):
        """Formats chat history into a readable string for the prompt."""
        if not self.chat_history:
            return "No previous conversation history."
        
        formatted_history = ""
        for message in self.chat_history:
            if isinstance(message, HumanMessage):
                formatted_history += f"Astronaut Query: {message.content}\n"
            elif isinstance(message, AIMessage):
                formatted_history += f"AI Assistant Response: {message.content}\n\n"
        return formatted_history.strip()

    def _create_prompt(self):
        """Creates the prompt template, now including chat history and a general 'input'."""
        return PromptTemplate(
            template="""
            [SYSTEM PROTOCOL: ASTRO-ASSISTANT COMMS]
            [ROLE: Onboard AI Assistant]
            [RECIPIENT: Astronaut]
            [TASK: Generate a helpful, professional response. Use the provided CONTEXT from the database and the CHAT HISTORY to answer the astronaut's LATEST QUERY. If the query is an emotion, acknowledge it and provide a supportive initial response. Your response should be around 200-400 words.]
            [CRITICAL INSTRUCTION: The output must contain ONLY the response itself. Omit all preambles, introductions, and conversational filler.]

            [CONVERSATION HISTORY]
            {chat_history}

            [CURRENT INPUT DATA]
            Retrieved Context: {context}
            Astronaut's Latest Query: {question}

            [BEGIN RESPONSE]
            """,
            input_variables=['chat_history', 'context', 'question']
        )

    def _create_chain(self):
        """Creates the full LangChain runnable for the chatbot."""
        
        # The retriever now uses the user's question to find relevant documents.
        # The 'RunnablePassthrough' ensures the original question is passed through the chain.
        retrieval_chain = RunnableParallel({
            'context': RunnablePassthrough() | self.retriever | self._format_docs,
            'question': RunnablePassthrough(),
            'chat_history': RunnableLambda(lambda _: self._format_history())
        })
        
        parser = StrOutputParser()
        
        main_chain = retrieval_chain | self.prompt | self.llm | parser
        return main_chain

    def get_response(self, user_input: str):
        """
        Gets a response from the chatbot for a given user input and updates the history.
        The input can be an emotion (for the first message) or a question.
        """
        if not user_input:
            return "Input cannot be empty."

        # Invoke the chain with the user's input (emotion or question)
        response = self.chain.invoke(user_input)
        
        # Update the chat history with the latest interaction
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response))
        
        # Limit history to the last 5 interactions (10 messages) to avoid overly long prompts
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
            
        return response

# --- Main execution block ---
# This is how you run the chatbot.
if __name__ == "__main__":
    # 1. Create an instance of the assistant (this does the one-time setup)
    assistant = AstroAssistant()


    # 2. Get the initial emotion to trigger the conversation
    try:
        initial_emotion = video_predict(emotion_model,frames_list)
        if initial_emotion.lower() == 'quit':
            print("Deactivating Astro Assistant. Goodbye.")
        else:
            # 3. Get the first response based on emotion
            initial_response = assistant.get_response(initial_emotion)
            print("\n--- AI Assistant ---")
            print(initial_response)
            print("--------------------")

            # 4. Start the main conversational loop for follow-up questions
            while True:
                user_query = input("\nYour Query > ")
                
                if user_query.lower() == 'quit':
                    print("Deactivating Astro Assistant. Goodbye.")
                    break
                
                # Get the subsequent responses based on the query
                ai_response = assistant.get_response(user_query)
                
                print("\n--- AI Assistant ---")
                print(ai_response)
                print("--------------------")

    except (KeyboardInterrupt, EOFError):
        print("\nDeactivating Astro Assistant. Goodbye.")

