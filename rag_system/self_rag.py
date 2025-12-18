"""
Self-RAG implementation for docstring generation.
"""

import time
import re
import requests
from bs4 import BeautifulSoup
import traceback
from typing import Tuple
from .base_rag import BaseRAG, CostMetrics
from .config import get_rag_method_config, get_common_rag_config
from .prompts import (
    get_context_query, get_rewrite_prompt, get_final_generation_prompt, 
    get_system_prompt, get_initial_generation_prompt, get_critique_prompt,
    get_relevance_evaluation_prompt, get_web_search_query
)

try:
    from googlesearch import search as google_search_func
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False

class SelfCorrectionRAG(BaseRAG):
    """
    SelfCorrection-RAG implementation (formerly Self-RAG).
    Uses a heuristic self-critique loop rather than reflection tokens.
    First attempts to generate a docstring without RAG, then self-critiques it, 
    and only triggers RAG if needed.
    """
    
    def __init__(self, index_name: str = None, custom_config: dict = None, namespace: str = None):
        """
        Initialize SelfCorrection-RAG.
        
        Args:
            index_name: Name of the Pinecone index
            custom_config: Custom configuration overrides
            namespace: Optional namespace for the index
        """
        if index_name is None:
            from .config import get_index_name, get_index_namespace
            index_name = get_index_name('self')
            namespace = get_index_namespace('self')
        else:
            namespace = None
        
        super().__init__(index_name, custom_config, namespace)
        
        # Use common configuration for fair benchmarking
        self.top_k = self.common_config.get('top_k', 3)
        self.use_rewrite = self.common_config.get('use_rewrite', True)
        self.rewrite_temperature = self.common_config.get('rewrite_temperature', 0.3)
        self.generation_temperature = self.common_config.get('generation_temperature', 0.5)
        self.web_search_enabled = self.common_config.get('web_search_enabled', True)
        self.web_search_max_results = self.common_config.get('web_search_max_results', 3)
        self.web_search_timeout = self.common_config.get('web_search_timeout', 15)
        self.min_chunk_length = self.common_config.get('min_chunk_length', 10)
        self.chunk_and_regrade = self.common_config.get('chunk_and_regrade', True)
        self.evaluation_enabled = self.common_config.get('evaluation_enabled', True)
        
        # Get Self-Correction specific configuration
        self.self_rag_config = get_rag_method_config('self_correction_rag')
        self.initial_generation_enabled = self.self_rag_config.get('initial_generation_enabled', True)
        self.self_critique_enabled = self.self_rag_config.get('self_critique_enabled', True)
        self.adaptive_rag_enabled = self.self_rag_config.get('adaptive_rag_enabled', True)
        self.critique_temperature = self.self_rag_config.get('critique_temperature', 0.0)
        self.web_search_fallback = self.self_rag_config.get('web_search_fallback', True)
        
        # Store contexts for evaluation
        self.retrieved_contexts = []
        
        self.logger.info("SelfCorrection-RAG initialized successfully")
    
    def generate_docstring(self, user_code: str) -> Tuple[str, CostMetrics]:
        """
        Generate docstring using Self-RAG approach.
        
        Args:
            user_code: Python code to generate docstring for
            
        Returns:
            Tuple of (generated_docstring, cost_metrics)
        """
        start_time = time.time()
        generation_start = time.time()
        
        # Step 1: Initial generation without RAG
        doc_without_rag = ""
        if self.initial_generation_enabled:
            doc_without_rag = self._initial_doc_without_rag(user_code)
        
        # Step 2: Self-critique
        needs_improvement = False
        if self.self_critique_enabled and doc_without_rag:
            needs_improvement = self._self_critique(doc_without_rag, user_code)
        elif not doc_without_rag:
            needs_improvement = True
        
        # Step 3: Adaptive RAG if needed
        final_context = ""
        retrieval_time = 0
        
        if needs_improvement and self.adaptive_rag_enabled:
            retrieval_start = time.time()
            context_query = self._generate_context_query(user_code)
            final_context = self._self_RAG_retrieval(context_query, user_code, needs_improvement)
            retrieval_time = time.time() - retrieval_start
        else:
            self.logger.info("Initial generation passed critique. No RAG needed.")
        
        generation_time = time.time() - generation_start
        
        # Store retrieved context
        self.retrieved_contexts.append(final_context)
        
        # Step 4: Generate final docstring
        if needs_improvement and final_context:
            # Use RAG-enhanced generation
            rewritten_req = self._revise_prompt_with_helper_model(final_context, user_code)
            generated_docstring = self._generate_final_docstring(final_context, user_code, rewritten_req)
        else:
            # Use initial generation
            generated_docstring = doc_without_rag
        
        # Track cost metrics
        api_calls = 1  # Initial generation
        if self.self_critique_enabled:
            api_calls += 1  # Critique
        if needs_improvement and self.adaptive_rag_enabled:
            api_calls += 2  # Helper + Generator
        
        cost_metrics = self._track_cost_metrics(
            start_time=start_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            api_calls=api_calls,
            tokens_used=0
        )
        
        self.logger.info(f"Docstring generated in {cost_metrics.execution_time:.3f}s")
        
        return generated_docstring, cost_metrics
    
    def _initial_doc_without_rag(self, user_code: str) -> str:
        """Generate initial docstring without RAG."""
        initial_docstring = ""
        try:
            initial_prompt = get_initial_generation_prompt(user_code)
            
            response = self.ollama_client.generate(
                model=self.model_config.generator_model,
                prompt=initial_prompt,
                options={'temperature': self.generation_temperature}
            )
            
            initial_docstring = response.get('response', '').strip()
            
        except Exception as e:
            self.logger.error(f"Error during initial generation: {e}")
            initial_docstring = ""
        
        return initial_docstring
    
    def _self_critique(self, initial_docstring: str, user_code: str) -> bool:
        """Self-critique the generated docstring."""
        needs_improvement = False
        
        if not initial_docstring:
            needs_improvement = True
        else:
            try:
                # Extract parameter names for critique
                param_names = []
                param_match = re.search(r'def\s+\w+\s*\((.*?)\):', user_code)
                if param_match:
                    params_str = param_match.group(1)
                    params = [p.strip().split(':')[0].split('=')[0].strip() 
                             for p in params_str.split(',') if p.strip() and p.strip() not in ['self', 'cls']]
                    param_names = [p for p in params if p and p != '*']
                
                critique_prompt = get_critique_prompt(user_code, initial_docstring, param_names)
                
                response = self.ollama_client.generate(
                    model=self.model_config.helper_model,
                    prompt=critique_prompt,
                    options={'temperature': self.critique_temperature}
                )
                
                critique_result = response.get('response', '').strip().upper()
                
                if "GOOD" not in critique_result:
                    needs_improvement = True
                    
            except Exception as e:
                self.logger.error(f"Error during self-critique: {e}")
                needs_improvement = True
        
        return needs_improvement
    
    def _self_RAG_retrieval(self, context_query: str, user_code: str, needs_improvement: bool) -> str:
        """Perform RAG retrieval when initial generation needs improvement."""
        final_context = ""
        final_source_description = "N/A"
        
        if needs_improvement and self.pinecone_index:
            # Initial Pinecone retrieval
            retrieved_matches = []
            initial_docs_for_refinement = []
            
            if context_query.strip():
                try:
                    query_embedding = self.embedding_model.encode(context_query).tolist()
                    search_results = self.pinecone_index.query(
                        vector=query_embedding, 
                        top_k=self.top_k, 
                        include_metadata=True,
                        namespace=self.namespace
                    )
                    retrieved_matches = search_results.matches
                    initial_docs_for_refinement = [
                        {"text": m.metadata.get('text', ''), "source": m.metadata.get('source', 'N/A')} 
                        for m in retrieved_matches if m.metadata.get('text')
                    ]
                except Exception as e:
                    self.logger.error(f"Error querying Pinecone: {e}")
            else:
                self.logger.warning("No context query provided, skipping Pinecone retrieval")
            
            # Web search fallback
            web_context_docs = []
            if context_query.strip() and self.web_search_enabled and GOOGLE_SEARCH_AVAILABLE:
                try:
                    code_first_line = user_code.split('\n', 1)[0].strip()
                    if len(code_first_line) > 50:
                        code_first_line = code_first_line[:50] + "..."
                    
                    web_query = get_web_search_query(context_query, code_first_line)
                    search_urls = []
                    try:
                        import itertools as _it
                        # Limit Google search results (generator might hang)
                        search_urls = list(_it.islice(google_search_func(web_query, lang="en"), self.web_search_max_results))
                    except TypeError:
                        # Fallback for older versions that accept 'num' parameter
                        try:
                            search_urls = list(google_search_func(web_query, lang="en", num=self.web_search_max_results))
                        except Exception:
                            pass
                    except Exception as gs_err:
                        self.logger.warning(f"Error during Google Search: {gs_err}")
                    
                    if search_urls:
                        for i, url in enumerate(search_urls[:self.web_search_max_results]):
                            try:
                                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                                response = requests.get(url, timeout=self.web_search_timeout, headers=headers)
                                response.raise_for_status()
                                
                                soup = BeautifulSoup(response.content, 'html.parser')
                                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                                
                                if main_content:
                                    page_text = main_content.get_text(separator='\n', strip=True)
                                else:
                                    page_text = soup.get_text(separator='\n', strip=True)
                                
                                if page_text:
                                    max_snippet_length = self.common_config.get('max_context_length', 4000)
                                    web_context_docs.append({
                                        "text": page_text[:max_snippet_length],
                                        "source": url,
                                        "type": "web"
                                    })
                                    
                            except requests.exceptions.Timeout:
                                self.logger.warning(f"Timeout fetching URL: {url}")
                            except requests.exceptions.RequestException as req_err:
                                self.logger.warning(f"Failed to fetch URL {url}: {req_err}")
                            except Exception as parse_err:
                                self.logger.warning(f"Failed to parse URL {url}: {parse_err}")
                            
                            time.sleep(1.5)  # Rate limiting
                    else:
                        self.logger.warning("Google Search did not return result URLs")
                        
                except Exception as e:
                    self.logger.error(f"Error during Google Search: {e}")
                    traceback.print_exc()
            elif not context_query.strip():
                self.logger.info("Skipping web search: No context query provided")
            elif not self.web_search_enabled:
                self.logger.info("Skipping web search: Web search disabled")
            elif not GOOGLE_SEARCH_AVAILABLE:
                self.logger.warning("Skipping web search: googlesearch-python not available")
            
            # Process and grade chunks
            relevant_chunks = []
            combined_sources_list = []
            all_potential_docs = initial_docs_for_refinement + web_context_docs
            
            if not all_potential_docs:
                self.logger.warning("No documents from Pinecone or Web to refine")
            else:
                for doc in all_potential_docs:
                    doc_text = doc.get('text', '')
                    doc_source = doc.get('source', 'N/A')
                    doc_type = doc.get('type', 'unknown')
                    
                    if not doc_text:
                        continue
                    
                    if self.chunk_and_regrade:
                        chunks = [chunk.strip() for chunk in doc_text.split('\n\n') if chunk.strip()]
                        
                        for chunk in chunks:
                            if self._grade_chunk_relevance(chunk, user_code, context_query):
                                relevant_chunks.append(chunk)
                                source_tag = f"{doc_type.capitalize()}:{doc_source.split('/')[-1]}"
                                if source_tag not in combined_sources_list:
                                    combined_sources_list.append(source_tag)
                    else:
                        # Use entire document if chunk grading is disabled
                        relevant_chunks.append(doc_text)
                        source_tag = f"{doc_type.capitalize()}:{doc_source.split('/')[-1]}"
                        if source_tag not in combined_sources_list:
                            combined_sources_list.append(source_tag)
            
            if relevant_chunks:
                final_context = "\n\n".join(relevant_chunks)
                final_source_description = " | ".join(combined_sources_list)
                self.logger.info(f"Final refined context compiled from {len(relevant_chunks)} relevant chunks")
                self.logger.info(f"Sources: {final_source_description}")
            else:
                self.logger.warning("No relevant chunks found after refinement")
                final_context = ""
        
        elif not self.pinecone_index:
            self.logger.warning("Skipping RAG pipeline as Pinecone index is unavailable")
        else:
            self.logger.info("Initial generation passed critique. Skipping RAG pipeline")
        
        return final_context
    
    def _grade_chunk_relevance(self, chunk_text: str, user_code: str, context_query: str) -> bool:
        """Grade chunk relevance using rule-based approach."""
        if not chunk_text or len(chunk_text.strip()) < self.min_chunk_length:
            return False
        
        chunk_lower = chunk_text.lower()
        code_lower = user_code.lower()
        query_lower = context_query.lower()
        
        # Core keywords
        core_keywords = ["docstring", "parameter", "argument", "return", "yield", "attribute", 
                        "class", "function", "method", "pep 257", "summary", "description", 
                        "example", "usage", "type hint"]
        
        if any(keyword in chunk_lower for keyword in core_keywords):
            return True
        
        # Query keywords
        if query_lower:
            query_keywords = re.findall(r'\b\w{3,}\b', query_lower)
            stop_words = {"how", "what", "the", "and", "for", "does", "work", "python", "use", "create"}
            query_keywords = [kw for kw in query_keywords if kw not in stop_words]
            if query_keywords and any(keyword in chunk_lower for keyword in query_keywords):
                return True
        
        # Code structure matching
        if ("def " in chunk_lower and "def " in code_lower) or \
           ("class " in chunk_lower and "class " in code_lower):
            return True
        
        # Parameter matching
        param_match = re.search(r'def\s+\w+\s*\((.*?)\):', user_code)
        if param_match:
            params = [p.strip().split('=')[0].strip() for p in param_match.group(1).split(',') if p.strip()]
            if any(f"parameter {p}" in chunk_lower or f"{p}:" in chunk_lower 
                   for p in params if p not in ['self', 'cls']):
                return True
        
        return False
    
    def _revise_prompt_with_helper_model(self, context: str, user_code: str) -> str:
        """Use helper LLM to refine the prompt for the generator model."""
        try:
            rewrite_prompt = get_rewrite_prompt(context)
            
            response = self.ollama_client.generate(
                model=self.model_config.helper_model,
                prompt=rewrite_prompt,
                options={'temperature': self.rewrite_temperature}
            )
            
            return response.get('response', '').strip()
            
        except Exception as e:
            self.logger.error(f"Error during prompt rewriting: {e}")
            return rewrite_prompt
    
    def _generate_final_docstring(self, context: str, user_code: str, rewritten_req: str) -> str:
        """Generate the final docstring using the main generator LLM."""
        messages = [
            {'role': 'system', 'content': get_system_prompt('docstring_generator')}
        ]
        
        if context:
            context_prompt = f"Here is potentially relevant context retrieved from knowledge base:\n{context}\n\nFor the python code:\n{user_code}\n\nBased on this context, generate an appropriate docstring."
            messages.append({'role': 'user', 'content': context_prompt})
        else:
            messages.append({'role': 'user', 'content': "No specific context was retrieved for this request."})
        
        messages.append({'role': 'user', 'content': get_final_generation_prompt(user_code)})
        
        try:
            response = self.ollama_client.chat(
                model=self.model_config.generator_model,
                messages=messages,
                options={'temperature': self.generation_temperature}
            )
            
            generated_docstring = response.get('message', {}).get('content', '').strip()
            generated_docstring = self._clean_docstring_output(generated_docstring)
            
            return generated_docstring
            
        except Exception as e:
            self.logger.error(f"Error communicating with Ollama: {e}")
            return "# ERROR: Docstring generation failed."
    
    def get_retrieved_contexts(self) -> list:
        """Get list of retrieved contexts for evaluation."""
        return self.retrieved_contexts
    
    def clear_history(self):
        """Clear stored contexts."""
        self.retrieved_contexts = []