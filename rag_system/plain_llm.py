"""
Plain LLM implementation (No RAG).
Generates docstrings using only the LLM's internal knowledge base, without retrieval.
Also supports Reasoning variants (CoT, ToT, GoT).
"""

import time
from typing import Tuple
from .base_rag import BaseRAG, CostMetrics
from .prompts import get_final_generation_prompt, get_system_prompt
from .reasoning_mixins import CoTMixin, ToTMixin, GoTMixin
from .config import get_index_name, get_index_namespace

class PlainLLM(BaseRAG):
    """
    Plain LLM implementation that generates docstrings without retrieval.
    This serves as a baseline to measure the impact of RAG.
    """
    
    def __init__(self, index_name: str = None, custom_config: dict = None, namespace: str = None):
        """
        Initialize Plain LLM.
        Args and structure match BaseRAG for compatibility, but index details are ignored.
        """
        # We don't need a real index, but BaseRAG expects one.
        if index_name is None:
            index_name = "plain-no-rag"
        
        super().__init__(index_name, custom_config, namespace)
        
        # Explicitly disable semantic retrieval since this is Plain LLM
        self.semantic_enabled = False
        self.pinecone_client = None
        self.pinecone_index = None
        self.embedding_model = None
        
        self.logger.info("Plain LLM initialized (Retrieval Disabled)")
    
    def generate_docstring(self, user_code: str) -> Tuple[str, CostMetrics]:
        """
        Generate docstring using only the LLM.
        """
        start_time = time.time()
        
        # No retrieval step
        retrieval_time = 0.0
        
        generation_start = time.time()
        
        # Direct generation
        final_docstring = self._generate_final_docstring(None, user_code, None)
        
        generation_time = time.time() - generation_start
        
        # Track cost metrics
        cost_metrics = self._track_cost_metrics(
            start_time=start_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            api_calls=1,
            tokens_used=0
        )
        
        self.logger.info(f"Docstring generated in {cost_metrics.execution_time:.3f}s")
        
        return final_docstring, cost_metrics

    def _generate_final_docstring(self, context: str, user_code: str, rewritten_req: str) -> str:
        """Generate the docstring using the generator model."""
        messages = [
            {'role': 'system', 'content': get_system_prompt('docstring_generator')},
            {'role': 'user', 'content': f"Generate a Python docstring for the following code. Do NOT use any external context, rely only on the code itself.\n\nCode:\n{user_code}"}
        ]
        
        try:
            response = self.ollama_client.chat(
                model=self.model_config.generator_model,
                messages=messages,
                options={'temperature': self.model_config.temperature}
            )
            
            generated_docstring = response.get('message', {}).get('content', '').strip()
            return self._clean_docstring_output(generated_docstring)
            
        except Exception as e:
            self.logger.error(f"Error communicating with Ollama: {e}")
            return "# ERROR: Docstring generation failed."
    
    def clear_history(self):
        pass


# Reasoning Variants for Plain LLM

class CoTPlainLLM(CoTMixin, PlainLLM):
    """Chain-of-Thought Plain LLM (No RAG)."""
    def __init__(self, index_name: str = None, custom_config: dict = None, namespace: str = None):
        PlainLLM.__init__(self, index_name, custom_config, namespace)
        CoTMixin.__init__(self)

class ToTPlainLLM(ToTMixin, PlainLLM):
    """Tree-of-Thought Plain LLM (No RAG)."""
    def __init__(self, index_name: str = None, custom_config: dict = None, namespace: str = None):
        PlainLLM.__init__(self, index_name, custom_config, namespace)
        ToTMixin.__init__(self)

class GoTPlainLLM(GoTMixin, PlainLLM):
    """Graph-of-Thought Plain LLM (No RAG)."""
    def __init__(self, index_name: str = None, custom_config: dict = None, namespace: str = None):
        PlainLLM.__init__(self, index_name, custom_config, namespace)
        GoTMixin.__init__(self)
