import fitz
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from multiprocessing import Pool, cpu_count
import time

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from langchain.prompts import ChatPromptTemplate
import onnxruntime_genai as og


class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text


class EmbeddingProcessor:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def generate_embeddings(self, texts):
        # Generate embeddings for each text without using multiprocessing
        return [(text, self.embedding_model.embed_query(text)) for text in texts]


class VectorStoreManager:
    def __init__(self, embedding_processor, vectorstore_path="vectorstore.pkl"):
        self.embedding_processor = embedding_processor
        self.vectorstore_path = vectorstore_path
        self.vectorstore_cache = None

    def load_or_generate_vectorstore(self, texts):
        if self.vectorstore_cache is not None:
            return self.vectorstore_cache

        try:
            if os.path.exists(self.vectorstore_path):
                with open(self.vectorstore_path, 'rb') as f:
                    self.vectorstore_cache = pickle.load(f)
                print("Loaded vector store from cache.")
            else:
                raise FileNotFoundError("No cache file")
        except (FileNotFoundError, pickle.UnpicklingError):
            print("Creating new vector store...")
            self.vectorstore_cache = FAISS.from_texts(texts=texts, embedding=self.embedding_processor.embedding_model)
            # Disabling pickle for now due to thread locking issues
            # with open(self.vectorstore_path, 'wb') as f:
            #     pickle.dump(self.vectorstore_cache, f)
            print("Created new vector store (not saved to disk)")
        
        return self.vectorstore_cache


class DeepSeekModel:
    def __init__(self, model_dir='C:\\Users\\aup\\Desktop\\local_ai_assistant\\DeepSeek-R1-Distill-Quantized', max_tokens=512):
        self.model_dir = model_dir
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None
        self.conversation_memory = []
        self._initialize_model()
        print(f"Model initialized with max_tokens={max_tokens}")
        
    def _initialize_model(self):
        print(f"Loading DeepSeek R1 model from: {self.model_dir}")
        try:
            # First try loading with default settings
            self._load_model()
        except Exception as e:
            error_message = str(e)
            print(f"Error loading model: {error_message}")
            if "hw queue" in error_message or "device cannot be scheduled" in error_message:
                print("Hardware acceleration error detected, falling back to CPU mode")
                self.use_cpu_fallback = True
                try:
                    # Set environment variable to force CPU mode
                    os.environ["ORT_GENAI_DEVICE"] = "cpu" 
                    print("Attempting to load model in CPU-only mode")
                    self._load_model()
                    print("Successfully loaded model in CPU-only mode")
                except Exception as cpu_error:
                    print(f"Failed to load model even in CPU mode: {str(cpu_error)}")
                    raise
            else:
                raise
                
    def _load_model(self):
        # Load the actual model
        model_options = {}
        if self.use_cpu_fallback:
            model_options["provider"] = "CPUExecutionProvider"
            
        self.model = og.Model(self.model_dir, **model_options)
        self.tokenizer, self.tokenizer_stream = self._get_tokenizer()
        print(f"Model loaded successfully using {'CPU' if self.use_cpu_fallback else 'hardware acceleration'}")
        
    def _get_tokenizer(self):
        tokenizer = og.Tokenizer(self.model)
        tokenizer_stream = tokenizer.create_stream()
        return tokenizer, tokenizer_stream
        
    def invoke(self, prompt_text):
        # Format the prompt if it's not already formatted
        if not prompt_text.startswith('<|user|>'):
            chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
            prompt = chat_template.format(input=prompt_text)
        else:
            prompt = prompt_text
            
        # For safety truncate very long prompts
        max_prompt_chars = 4000
        if len(prompt) > max_prompt_chars:
            print(f"[WARNING] Truncating prompt from {len(prompt)} to {max_prompt_chars} characters")
            prompt = prompt[:max_prompt_chars] + "..."
            
        print(f"[DEBUG] Invoking DeepSeek model with prompt: '{prompt[:100]}...'")
            
        # Set up the generator
        search_options = {}
        search_options['max_length'] = self.max_tokens

        # For multiple attempts in case of hardware errors
        for attempt in range(self.retries):
            try:
                input_tokens = self.tokenizer.encode(prompt)
                
                params = og.GeneratorParams(self.model)
                params.set_search_options(**search_options)
                params.input_ids = input_tokens
                
                generator = og.Generator(self.model, params)
                
                # Get stop tokens for </|assistant|>
                stop_sequence = "</|assistant|>"
                stop_tokens = self.tokenizer.encode(stop_sequence)
                
                # Generate response
                print(f"[DEBUG] Starting generation (attempt {attempt+1}/{self.retries})")
                result_text = self._run_generator(generator, stop_tokens)
                print(f"[DEBUG] Generation completed successfully")
                
                # Clean up
                del generator
                
                # Reset error count on success
                self.hardware_error_count = 0
                
                # Create a response object similar to what LangChain expects
                class Response:
                    def __init__(self, content):
                        self.content = content
                        
                return Response(result_text)
            except Exception as e:
                print(f"[ERROR] Error during generation (attempt {attempt+1}): {str(e)}")
                self.hardware_error_count += 1
                
                # Check if this is a hardware error
                error_message = str(e)
                is_hw_error = "hw queue" in error_message or "device cannot be scheduled" in error_message
                
                # If not using CPU fallback yet and we hit a hardware error, try switching to CPU
                if is_hw_error and not self.use_cpu_fallback:
                    print("[DEBUG] Hardware error detected, switching to CPU mode")
                    self.use_cpu_fallback = True
                    try:
                        # Set environment variable to force CPU mode
                        os.environ["ORT_GENAI_DEVICE"] = "cpu" 
                        print("Attempting to reinitialize model in CPU-only mode")
                        self._load_model()
                        print("Successfully reinitialized model in CPU-only mode")
                        # Try again immediately with CPU mode
                        continue
                    except Exception as cpu_error:
                        print(f"Failed to switch to CPU mode: {str(cpu_error)}")
                
                # If this was the last attempt, provide a fallback response
                if attempt == self.retries - 1:
                    print(f"[ERROR] Failed after {self.retries} attempts")
                    
                    # Provide a user-friendly response based on the error
                    if is_hw_error:
                        print("[DEBUG] Hardware error detected, providing fallback response")
                        fallback_content = "I apologize, but I'm experiencing some technical difficulties with my hardware acceleration. I can still try to help you, but my responses might be limited. Could you please try again with a simpler or shorter query?"
                        return Response(fallback_content)
                    else:
                        # For non-hardware errors, return a generic fallback
                        print("[DEBUG] Non-hardware error detected, providing generic fallback")
                        fallback_content = "I apologize, but I encountered an issue while generating a response. Please try again with a different query."
                        return Response(fallback_content)
                
                # Wait before retrying
                time.sleep(1)  # Short delay between retries
        
    def _run_generator(self, generator, stop_tokens):
        result_text = ""
        token_buffer = []
        stop_tokens_len = len(stop_tokens)
        full_text = ""
        
        try:
            token_count = 0
            max_tokens = self.max_tokens  # Safety limit
            
            while not generator.is_done() and token_count < max_tokens:
                try:
                    generator.compute_logits()
                    generator.generate_next_token()
    
                    new_token = generator.get_next_tokens()[0]
                    token_text = self.tokenizer_stream.decode(new_token)
                    
                    token_buffer.append(new_token)
                    if len(token_buffer) > stop_tokens_len:
                        token_buffer.pop(0)
                    
                    full_text += token_text
                    result_text += token_text
                    token_count += 1
                    
                    # Occasionally print progress for long generations
                    if token_count % 50 == 0:
                        print(f"[DEBUG] Generated {token_count} tokens so far")
                    
                    # Check if we've generated the stop sequence
                    if len(token_buffer) == stop_tokens_len:
                        if token_buffer == stop_tokens:
                            break
                        
                    # Alternative check: look for "</|assistant|>" in the full text
                    if "</|assistant|>" in full_text:
                        # Remove the stop token from the result
                        result_text = result_text.replace("</|assistant|>", "").strip()
                        break
                except Exception as e:
                    print(f"[ERROR] Token generation error: {str(e)}")
                    
                    # If hardware error, break and return what we have so far
                    error_message = str(e)
                    if "hw queue" in error_message or "device cannot be scheduled" in error_message:
                        print("[WARNING] Hardware error during generation, returning partial result")
                        # Only add the note if we've generated something substantial
                        if len(result_text.strip()) > 30:
                            result_text += "\n\n(Note: Response was cut short due to technical issues)"
                        break
                    # For other errors, re-raise
                    raise
                    
        except KeyboardInterrupt:
            print("Generation interrupted")
            
        # If we've generated nothing, provide a fallback
        if not result_text.strip():
            result_text = "I apologize, but I'm having trouble generating a response right now. Please try again."
        
        # Log a brief preview of the result
        preview = result_text[:100] + "..." if len(result_text) > 100 else result_text
        print(f"[DEBUG] Generated result: '{preview}'")
            
        return result_text

    def invoke_stream(self, prompt_text):
        """Stream the model response sentence-by-sentence"""
        # Format the prompt if it's not already formatted
        if not prompt_text.startswith('<|user|>'):
            chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
            prompt = chat_template.format(input=prompt_text)
        else:
            prompt = prompt_text
            
        # For safety truncate very long prompts
        max_prompt_chars = 4000
        if len(prompt) > max_prompt_chars:
            print(f"[WARNING] Truncating prompt from {len(prompt)} to {max_prompt_chars} characters")
            prompt = prompt[:max_prompt_chars] + "..."
            
        print(f"[DEBUG] Invoking DeepSeek model with streaming for prompt: '{prompt[:100]}...'")
            
        # Set up the generator
        search_options = {}
        search_options['max_length'] = self.max_tokens

        # For multiple attempts in case of hardware errors
        for attempt in range(self.retries):
            try:
                input_tokens = self.tokenizer.encode(prompt)
                
                params = og.GeneratorParams(self.model)
                params.set_search_options(**search_options)
                params.input_ids = input_tokens
                
                generator = og.Generator(self.model, params)
                
                # Get stop tokens for </|assistant|>
                stop_sequence = "</|assistant|>"
                stop_tokens = self.tokenizer.encode(stop_sequence)
                
                # Generate response
                print(f"[DEBUG] Starting streaming generation (attempt {attempt+1}/{self.retries})")
                
                # Variables for streaming
                sentence_buffer = ""
                sent_eos_chars = ['.', '!', '?', '\n']  # End of sentence characters
                token_count = 0
                stop_pattern_found = False
                result_text = ""
                
                while not generator.is_done() and token_count < self.max_tokens and not stop_pattern_found:
                    try:
                        generator.compute_logits()
                        generator.generate_next_token()
                        
                        new_token = generator.get_next_tokens()[0]
                        token_text = self.tokenizer_stream.decode(new_token)
                        
                        sentence_buffer += token_text
                        result_text += token_text
                        token_count += 1
                        
                        # Occasionally print progress for long generations
                        if token_count % 50 == 0:
                            print(f"[DEBUG] Generated {token_count} tokens so far")
                        
                        # Check if we've generated the stop sequence
                        if stop_sequence in result_text:
                            # Remove the stop token from the result
                            result_text = result_text.replace(stop_sequence, "").strip()
                            sentence_buffer = sentence_buffer.replace(stop_sequence, "").strip()
                            stop_pattern_found = True
                        
                        # Check if we reached the end of a sentence
                        for char in sent_eos_chars:
                            if char in sentence_buffer:
                                # Split on sentence ending and keep the remainder for the next sentence
                                sentences = sentence_buffer.split(char)
                                
                                # The completed sentence includes the terminating character
                                completed_sentence = sentences[0] + char
                                
                                # Yield the completed sentence
                                yield completed_sentence
                                
                                # Keep the remainder (if any) for the next sentence
                                sentence_buffer = char.join(sentences[1:])
                                break
                    
                    except Exception as e:
                        print(f"[ERROR] Token generation error: {str(e)}")
                        
                        # If hardware error, break and return what we have so far
                        error_message = str(e)
                        if "hw queue" in error_message or "device cannot be scheduled" in error_message:
                            print("[WARNING] Hardware error during generation, returning partial result")
                            if sentence_buffer:
                                yield sentence_buffer  # Send the remaining text
                            yield "\n\n(Note: Response was cut short due to technical issues)"
                            break
                        # For other errors, re-raise
                        raise
                
                # If there's anything left in the buffer, send it
                if sentence_buffer:
                    yield sentence_buffer
                
                # Clean up
                del generator
                
                # Reset error count on success
                self.hardware_error_count = 0
                break  # Exit the retry loop
                
            except Exception as e:
                print(f"[ERROR] Error during streaming generation (attempt {attempt+1}): {str(e)}")
                self.hardware_error_count += 1
                
                # Check if this is a hardware error
                error_message = str(e)
                is_hw_error = "hw queue" in error_message or "device cannot be scheduled" in error_message
                
                # If not using CPU fallback yet and we hit a hardware error, try switching to CPU
                if is_hw_error and not self.use_cpu_fallback:
                    print("[DEBUG] Hardware error detected, switching to CPU mode")
                    self.use_cpu_fallback = True
                    try:
                        # Set environment variable to force CPU mode
                        os.environ["ORT_GENAI_DEVICE"] = "cpu" 
                        print("Attempting to reinitialize model in CPU-only mode")
                        self._load_model()
                        print("Successfully reinitialized model in CPU-only mode")
                        # Try again immediately with CPU mode
                        continue
                    except Exception as cpu_error:
                        print(f"Failed to switch to CPU mode: {str(cpu_error)}")
                
                # If this was the last attempt, provide a fallback response
                if attempt == self.retries - 1:
                    print(f"[ERROR] Failed after {self.retries} attempts")
                    
                    # Provide a user-friendly response based on the error
                    if is_hw_error:
                        print("[DEBUG] Hardware error detected, providing fallback response")
                        yield "I apologize, but I'm experiencing some technical difficulties with my hardware acceleration."
                        yield "I can still try to help you, but my responses might be limited."
                        yield "Could you please try again with a simpler or shorter query?"
                    else:
                        # For non-hardware errors, return a generic fallback
                        print("[DEBUG] Non-hardware error detected, providing generic fallback")
                        yield "I apologize, but I encountered an issue while generating a response."
                        yield "Please try again with a different query."
                
                # Wait before retrying
                time.sleep(1)  # Short delay between retries


class LangChain:
    def __init__(self, model_path='C:\\Users\\aup\\Desktop\\local_ai_assistant\\DeepSeek-R1-Distill-Quantized', temperature=0.7):
        self.llm = DeepSeekModel(model_dir=model_path, max_tokens=2048)
        self.conversation_memory = []
        self.prompt_template = ChatPromptTemplate.from_template("""
            You are BuffAdvisor, an AI assistant for University of Colorado Boulder students.
            Provide quick, concise information about UC Boulder.
            BE EXTREMELY BRIEF. Answer in 1-3 sentences only. Don't overthink.

            Advisory Style: {solution_style}
            
            Student Query: {question}
            
            Reference Information (be selective about what you use):
            {context}
            
            Keep your response very concise. Students need quick answers.
        """)

    def choose_solution_style(self, solution_style):
        if solution_style == "brief":
            return (
                "Provide the shortest possible answers. One sentence is ideal."
            )
        elif solution_style == "detailed":
            return (
                "Provide concise but informative answers in 2-3 sentences. No longer."
            )
        elif solution_style == "supportive":
            return (
                "Be encouraging but extremely brief. Keep answers to 1-2 sentences."
            )
        else:
            return (
                "Balance information with brevity. Maximum 2 sentences."
            )

    def generate_response(self, retriever, question, solution_style):
        print(f"[DEBUG] Generating response for question: '{question}'")
        
        # Check if the question is too short, if so provide a direct response
        if len(question.strip()) <= 5:
            print("[DEBUG] Question is very short, providing direct response without context lookup")
            short_question_response = self._generate_short_question_response(question)
            self.conversation_memory.append(f"Student: {question}")
            self.conversation_memory.append(f"BuffAdvisor: {short_question_response}")
            return short_question_response
            
        try:
            self.conversation_memory.append(f"Student: {question}")
            
            # Try to retrieve relevant documents - with error handling
            context_text = self._get_context_text(retriever, question)
            
            # Create the context dictionary with the extracted text
            context = {
                "context": context_text,
                "question": question,
                "solution_style": self.choose_solution_style(solution_style),
                # No longer including memory to keep responses shorter and more focused
            }
            
            # Process with the language model
            print("[DEBUG] Formatting prompt template")
            response = self.prompt_template.format(**context)
            
            print("[DEBUG] Invoking language model")
            # Force brief mode by adding an explicit instruction
            if not response.endswith("be brief."):
                response += " Please keep your answer under 3 sentences and be very direct."
                
            result = self.llm.invoke(response)
            response_text = result.content
            
            # Truncate very long responses
            if len(response_text) > 500:
                print(f"[DEBUG] Truncating long response from {len(response_text)} to 500 characters")
                response_text = response_text[:497] + "..."
            
            self.conversation_memory.append(f"BuffAdvisor: {response_text}")
            print(f"[DEBUG] Response generated successfully - {len(response_text)} characters")
            return response_text
        except Exception as e:
            error_text = f"[ERROR] Failed to generate response: {str(e)}"
            print(error_text)
            
            # Provide a fallback response that doesn't rely on any of the failed components
            fallback = self._generate_fallback_response(question)
            self.conversation_memory.append(f"BuffAdvisor: {fallback}")
            return fallback
            
    def _get_context_text(self, retriever, question):
        """Get context text from the retriever with proper error handling."""
        try:
            print("[DEBUG] Retrieving relevant documents")
            
            # Use the invoke method instead of the deprecated get_relevant_documents
            # Check if the new invoke method is available, otherwise fall back to deprecated method
            if hasattr(retriever, "invoke"):
                print("[DEBUG] Using retriever.invoke() method")
                relevant_docs = retriever.invoke(question)
            else:
                print("[DEBUG] Falling back to deprecated get_relevant_documents() method")
                # Suppress the warning for this specific call
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    relevant_docs = retriever.get_relevant_documents(question)
            
            print(f"[DEBUG] Retrieved {len(relevant_docs)} relevant documents")
            return "\n".join([doc.page_content for doc in relevant_docs])
        except Exception as e:
            print(f"[ERROR] Failed to retrieve context: {str(e)}")
            return "No specific information found. Providing general guidance."
            
    def _generate_short_question_response(self, question):
        """Generate a response for very short questions without context lookup."""
        if question.lower() in ["hi", "hello", "hey"]:
            return "Hello! I'm BuffAdvisor, here to help you with information about CU Boulder. How can I assist you today?"
        elif question.lower() in ["thanks", "thank you", "thx"]:
            return "You're welcome! I'm happy to help. Is there anything else you'd like to know about CU Boulder?"
        else:
            return f"I see you've sent '{question}'. Could you please provide more details about what you'd like to know about CU Boulder? I'm here to help with information about campus resources, academic programs, student life, and more."
            
    def _generate_fallback_response(self, question):
        """Generate a fallback response when normal processing fails."""
        return f"I apologize, but I'm experiencing some technical difficulties in processing your question about '{question}'. As BuffAdvisor, I'm here to help with information about CU Boulder. Could you try rephrasing your question or asking something else about the university?"

    def generate_response_stream(self, retriever, question, solution_style):
        print(f"[DEBUG] Generating streaming response for question: '{question}'")
        
        # Check if the question is too short, if so provide a direct response
        if len(question.strip()) <= 5:
            print("[DEBUG] Question is very short, providing direct response without context lookup")
            short_question_response = self._generate_short_question_response(question)
            self.conversation_memory.append(f"Student: {question}")
            self.conversation_memory.append(f"BuffAdvisor: {short_question_response}")
            # Return the short response as a single chunk
            yield short_question_response
            return
            
        try:
            self.conversation_memory.append(f"Student: {question}")
            
            # Try to retrieve relevant documents - with error handling
            context_text = self._get_context_text(retriever, question)
            
            # Create the context dictionary with the extracted text
            context = {
                "context": context_text,
                "question": question,
                "solution_style": self.choose_solution_style(solution_style),
            }
            
            # Process with the language model
            print("[DEBUG] Formatting prompt template")
            response = self.prompt_template.format(**context)
            
            print("[DEBUG] Invoking language model with streaming")
            # Force brief mode by adding an explicit instruction
            if not response.endswith("be brief."):
                response += " Please keep your answer under 3 sentences and be very direct."
            
            # Variable to store the complete response for conversation memory
            full_response = ""
            
            # Stream the response sentence by sentence
            for sentence in self.llm.invoke_stream(response):
                # Add to the full response
                full_response += sentence
                
                # Stream the sentence to the caller
                yield sentence
            
            # Truncate very long responses
            if len(full_response) > 500:
                print(f"[DEBUG] Full response was {len(full_response)} characters (truncated in memory)")
                full_response = full_response[:497] + "..."
            
            # Add the full response to conversation memory
            self.conversation_memory.append(f"BuffAdvisor: {full_response}")
            print(f"[DEBUG] Streaming response generated successfully - {len(full_response)} characters")
            
        except Exception as e:
            error_text = f"[ERROR] Failed to generate streaming response: {str(e)}"
            print(error_text)
            
            # Provide a fallback response that doesn't rely on any of the failed components
            fallback = self._generate_fallback_response(question)
            self.conversation_memory.append(f"BuffAdvisor: {fallback}")
            yield fallback


class BuffAdvisor:
    def __init__(self, pdf_path, model_path='C:\\Users\\aup\\Desktop\\local_ai_assistant\\DeepSeek-R1-Distill-Quantized'):
        self.pdf_processor = PDFProcessor(pdf_path)
        self.embedding_processor = EmbeddingProcessor(OllamaEmbeddings(model='nomic-embed-text'))
        self.vector_store_manager = VectorStoreManager(self.embedding_processor)
        self.langchain = LangChain(model_path=model_path)
        self.pdf_text = None
        self.vectorstore = None

    def setup(self):
        if self.pdf_text is None:  # Only process if not already done
            print("Extracting text from PDF...")
            self.pdf_text = self.pdf_processor.extract_text()
            print(f"Text extracted. Splitting into chunks...")
            texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(self.pdf_text)
            print(f"Created {len(texts)} text chunks. Generating embeddings (this may take a while)...")
            self.vectorstore = self.vector_store_manager.load_or_generate_vectorstore(texts)
            print("Vector store ready.")

    def advisor_query(self, new_session=True, question="", style="balanced"):
        retriever = self.vectorstore.as_retriever()
        if new_session:
            self.langchain.conversation_memory = []
        response = self.langchain.generate_response(retriever, question, style)
        return response

    def advisor_query_stream(self, new_session=True, question="", style="balanced"):
        retriever = self.vectorstore.as_retriever()
        if new_session:
            self.langchain.conversation_memory = []
        response_generator = self.langchain.generate_response_stream(retriever, question, style)
        return response_generator


# Global instance variable for BuffAdvisor
advisor_instance = None

def initialize_advisor(pdf_path=None, model_path='C:\\Users\\aup\\Desktop\\local_ai_assistant\\DeepSeek-R1-Distill-Quantized'):
    global advisor_instance
    print("[DEBUG] initialize_advisor called")
    
    try:
        if advisor_instance is None:
            # Look for any PDF in the current directory if no path is provided
            if pdf_path is None:
                pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
                if pdf_files:
                    pdf_path = pdf_files[0]
                    print(f"[DEBUG] Using PDF file: {pdf_path}")
                else:
                    print("[ERROR] No PDF files found in the current directory.")
                    print("Please place a CU Boulder information PDF file in the current directory or specify the path.")
                    return
            else:
                # If path was provided, check if it exists
                if not os.path.exists(pdf_path):
                    print(f"[ERROR] PDF file not found: {pdf_path}")
                    print("Please provide a valid PDF path.")
                    return
            
            print(f"[DEBUG] Initializing BuffAdvisor with PDF: {pdf_path}")
            try:
                # First try with default settings
                advisor_instance = BuffAdvisor(pdf_path=pdf_path, model_path=model_path)
                print("[DEBUG] Starting setup process...")
                advisor_instance.setup()
                print("[DEBUG] Setup complete! BuffAdvisor is ready to answer questions.")
            except Exception as e:
                error_message = str(e)
                print(f"[ERROR] Error during initialization: {error_message}")
                
                # If hardware error, try with CPU mode
                if "hw queue" in error_message or "device cannot be scheduled" in error_message:
                    print("[DEBUG] Hardware error detected during initialization, forcing CPU mode")
                    os.environ["ORT_GENAI_DEVICE"] = "cpu"
                    print("[DEBUG] Retrying initialization with CPU mode")
                    advisor_instance = BuffAdvisor(pdf_path=pdf_path, model_path=model_path)
                    print("[DEBUG] Starting setup process in CPU mode...")
                    advisor_instance.setup()
                    print("[DEBUG] Setup complete in CPU mode! BuffAdvisor is ready to answer questions.")
                else:
                    # Re-raise for other errors
                    raise
    except Exception as e:
        print(f"[CRITICAL] Failed to initialize BuffAdvisor: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_advice(new_session=True, question="", style="balanced"):
    if advisor_instance is None:
        print("[ERROR] BuffAdvisor has not been initialized.")
        return "BuffAdvisor has not been initialized. Call initialize_advisor(pdf_path) first."
    
    try:
        print(f"[DEBUG] get_advice called with question: '{question}'")
        response = advisor_instance.advisor_query(new_session=new_session, question=question, style=style)
        print(f"[DEBUG] Response generated - length: {len(response)} characters")
        return response
    except Exception as e:
        error_message = f"[ERROR] Exception in get_advice: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return f"I'm sorry, but I encountered an error while processing your question. Please try again with a different query. Technical details: {str(e)}"

def get_advice_stream(new_session=True, question="", style="balanced"):
    """Stream the response sentence by sentence."""
    if advisor_instance is None:
        print("[ERROR] BuffAdvisor has not been initialized.")
        yield "BuffAdvisor has not been initialized. Call initialize_advisor(pdf_path) first."
        return
    
    try:
        print(f"[DEBUG] get_advice_stream called with question: '{question}'")
        response_generator = advisor_instance.advisor_query_stream(new_session=new_session, question=question, style=style)
        for sentence in response_generator:
            print(f"[DEBUG] Streaming sentence: '{sentence}'")
            yield sentence
    except Exception as e:
        error_message = f"[ERROR] Exception in get_advice_stream: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        yield f"I'm sorry, but I encountered an error while processing your question."
        yield f"Please try again with a different query."
        yield f"Technical details: {str(e)}"


# Testing function for DeepSeek model directly
def test_deepseek_model():
    model = DeepSeekModel()
    print("Testing DeepSeek R1 model...")
    
    # Simple test
    test_prompt = "Tell me about yourself"
    print(f"\nTest prompt: '{test_prompt}'")
    
    response = model.invoke(test_prompt)
    print(f"Response: {response.content}")
    
    # Test with PDF content
    pdf_path = "book.pdf"
    if os.path.exists(pdf_path):
        print("\nTesting with PDF content...")
        pdf_processor = PDFProcessor(pdf_path)
        text = pdf_processor.extract_text()
        sample_text = text[:500] + "..."  # Just take a sample
        
        test_prompt = f"Summarize this text: {sample_text}"
        print(f"\nPrompting with sample text from PDF")
        
        response = model.invoke(test_prompt)
        print(f"Response: {response.content}")


if __name__ == "__main__":
    # Choose which test to run
    print("Select an option:")
    print("1. Test DeepSeek R1 model directly")
    print("2. Run BuffAdvisor with DeepSeek R1")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_deepseek_model()
    else:
        pdf_path = "book.pdf"
        model_path = 'C:\\Users\\aup\\Desktop\\local_ai_assistant\\DeepSeek-R1-Distill-Quantized'
        
        # Initialize the BuffAdvisor instance
        initialize_advisor(pdf_path, model_path)
        
        if advisor_instance is None:
            print("Exiting as no PDF was found or provided.")
            exit(1)
        
        print("\n" + "="*50)
        print("Welcome to BuffAdvisor Interactive Chat with DeepSeek R1!")
        print("Your UC Boulder assistant is ready to help you with questions about")
        print("campus resources, academic programs, student life, and more.")
        print("="*50)
        print("Commands:")
        print("- Type 'exit' or 'quit' to end the chat")
        print("- Type 'new session' to start a fresh conversation")
        print("- Type 'style:brief', 'style:detailed', or 'style:supportive' to change response style")
        print("- Otherwise, just type your question and press Enter")
        print("="*50 + "\n")
        
        current_style = "balanced"
        current_session = True  # Start with a new session
        
        while True:
            user_input = input("\nYour question: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nThank you for using BuffAdvisor. Go Buffs!")
                break
                
            elif user_input.lower() == 'new session':
                current_session = True
                print("\nStarting a new conversation session.")
                continue
                
            elif user_input.lower().startswith('style:'):
                style_option = user_input.lower().split(':', 1)[1].strip()
                if style_option in ['brief', 'detailed', 'supportive', 'balanced']:
                    current_style = style_option
                    print(f"\nSwitched to {current_style} response style.")
                else:
                    print("\nInvalid style. Available styles: brief, detailed, supportive, balanced")
                continue
                
            # Process the user's question and get a response
            try:
                response = get_advice(
                    new_session=current_session,
                    question=user_input,
                    style=current_style
                )
                print(f"\nBuffAdvisor: {response}")
                
                # After the first response, we're no longer in a new session
                if current_session:
                    current_session = False
            except Exception as e:
                print(f"\nError: {e}")
                print("There was an error processing your request. Please try again.")
