"""
Simple RAG implementation for docstring generation.
"""

import time
import re
from typing import Tuple
from .base_rag import BaseRAG, CostMetrics
from .config import get_rag_method_config, get_common_rag_config
from .prompts import get_context_query, get_rewrite_prompt, get_final_generation_prompt, get_system_prompt

class SimpleRAG(BaseRAG):
    """
    Simple RAG implementation that performs basic semantic search
    and generates docstrings using retrieved context.
    """
    
    def __init__(self, index_name: str = None, custom_config: dict = None, namespace: str = None):
        """
        Initialize Simple RAG.
        
        Args:
            index_name: Name of the Pinecone index
            custom_config: Custom configuration overrides
            namespace: Optional namespace for the index
        """
        if index_name is None:
            from .config import get_index_name, get_index_namespace
            index_name = get_index_name('simple')
            namespace = get_index_namespace('simple')
        else:
            namespace = None
        
        super().__init__(index_name, custom_config, namespace)
        
        # Use common configuration for fair benchmarking
        self.top_k = self.common_config.get('top_k', 3)
        self.use_rewrite = self.common_config.get('use_rewrite', True)
        self.rewrite_temperature = self.common_config.get('rewrite_temperature', 0.3)
        self.generation_temperature = self.common_config.get('generation_temperature', 0.5)
        
        # Store contexts for evaluation
        self.retrieved_contexts = []
        self.rewritten_prompts = []
        
        self.logger.info("Simple RAG initialized successfully")
    
    def generate_docstring(self, user_code: str) -> Tuple[str, CostMetrics]:
        """
        Generate docstring using Simple RAG approach.
        
        Args:
            user_code: Python code to generate docstring for
            
        Returns:
            Tuple of (generated_docstring, cost_metrics)
        """
        start_time = time.time()
        retrieval_start = time.time()
        
        # Step 1: Code Parser - Extract entities from code
        entities = self._parse_code_for_entities(user_code)
        
        # Step 2: Enriched Query Constructor
        enriched_query = self._construct_enriched_query(entities, user_code)
        
        context = ""
        
        # Step 3: Retrieve context from Pinecone using enriched query
        if self.pinecone_index and enriched_query.strip():
            try:
                # Step 4: Embedding Model - Convert enriched query to embedding
                query_embedding = self.embedding_model.encode(enriched_query).tolist()
                
                search_results = self.pinecone_index.query(
                    vector=query_embedding,
                    top_k=self.top_k,
                    include_metadata=True,
                    namespace=self.namespace
                )
                
                if search_results.matches:
                    match = search_results.matches[0]
                    context = match.metadata.get('text', '')
                    self.logger.debug(f"Retrieved context from: {match.metadata.get('source', 'Unknown')}")
                else:
                    context = user_code  # Fallback if no context found
                    self.logger.warning("No relevant context found, using code as fallback")
                    
            except Exception as e:
                self.logger.error(f"Error during retrieval: {e}")
                context = user_code  # Fallback on error
        
        retrieval_time = time.time() - retrieval_start
        generation_start = time.time()
        
        # Store retrieved context
        self.retrieved_contexts.append(context)
        
        # Rewrite prompt if enabled
        if self.use_rewrite:
            rewritten_req = self._revise_prompt_with_helper_model(context, user_code)
            self.rewritten_prompts.append(rewritten_req)
        else:
            rewritten_req = context
        
        # Generate final docstring
        final_docstring = self._generate_final_docstring(context, user_code, rewritten_req)
        
        generation_time = time.time() - generation_start
        
        # Track cost metrics
        cost_metrics = self._track_cost_metrics(
            start_time=start_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            api_calls=2 if self.use_rewrite else 1,  # Helper + Generator or just Generator
            tokens_used=0  # Placeholder - would need actual token counting
        )
        
        self.logger.info(f"Docstring generated in {cost_metrics.execution_time:.3f}s")
        
        return final_docstring, cost_metrics
    
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
            return rewrite_prompt  # Fallback to original prompt
    
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
                options={'temperature': self.model_config.temperature}
            )
            
            generated_docstring = response.get('message', {}).get('content', '').strip()
            
            # Basic cleaning
            generated_docstring = self._clean_docstring_output(generated_docstring)
            
            return generated_docstring
            
        except Exception as e:
            self.logger.error(f"Error communicating with Ollama: {e}")
            return "# ERROR: Docstring generation failed."
    
    def get_retrieved_contexts(self) -> list:
        """Get list of retrieved contexts for evaluation."""
        return self.retrieved_contexts
    
    def get_rewritten_prompts(self) -> list:
        """Get list of rewritten prompts for analysis."""
        return self.rewritten_prompts
    
    def clear_history(self):
        """Clear stored contexts and prompts."""
        self.retrieved_contexts = []
        self.rewritten_prompts = []
    
    def _parse_code_for_entities(self, user_code: str) -> dict:
        """
        Parse code to extract class name, methods, and other entities.
        Matches Simple RAG flow diagram: Code Parser → Extracted Entities.
        
        Args:
            user_code: Python code to parse
            
        Returns:
            Dictionary containing extracted entities
        """
        entities = {
            "class_name": "",
            "methods": [],
            "code_snippet": user_code
        }
        
        try:
            # Extract class name
            class_name_match = re.search(r"class\s+(\w+)", user_code)
            if class_name_match:
                entities["class_name"] = class_name_match.group(1)
            
            # Extract method names
            method_matches = re.findall(r"def\s+(_?\w+)\s*\(", user_code)
            entities["methods"] = method_matches
            
            self.logger.debug(f"Extracted entities: class={entities['class_name']}, methods={len(entities['methods'])}")
            
        except Exception as e:
            self.logger.error(f"Error parsing code entities: {e}")
        
        return entities
    
    def _construct_enriched_query(self, entities: dict, user_code: str) -> str:
        """
        Construct enriched query from extracted entities.
        Matches Simple RAG flow diagram: Enriched Query Constructor.
        
        Args:
            entities: Extracted code entities from _parse_code_for_entities
            user_code: Original user code
            
        Returns:
            Enriched query string for retrieval
        """
        query_parts = []
        
        # Add class information if available
        if entities.get("class_name"):
            query_parts.append(f"python docstring for class {entities['class_name']}")
        
        # Add method information
        if entities.get("methods"):
            public_methods = [m for m in entities["methods"] if not m.startswith('__') or m == '__init__']
            if public_methods:
                methods_str = ', '.join(public_methods[:5])  # Limit to first 5 methods
                query_parts.append(f"with methods {methods_str}")
        
        # Fallback to basic query if no entities extracted
        if not query_parts:
            # Use the standard context query as fallback
            enriched_query = self._generate_context_query(user_code)
        else:
            enriched_query = " ".join(query_parts)
            # Add code snippet for additional context
            if len(user_code) < 500:  # Only include code if not too long
                enriched_query += f" {user_code[:300]}"
        
        self.logger.debug(f"Enriched query constructed: {enriched_query[:100]}...")
        
        return enriched_query