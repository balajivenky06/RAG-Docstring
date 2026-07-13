"""
Plain LLM implementation (No RAG).
Generates docstrings using only the LLM's internal knowledge base, without retrieval.
Also supports Reasoning variants (CoT, ToT, GoT).
"""

import os
import time
from typing import Tuple
from .base_rag import BaseRAG, CostMetrics
from .prompts import (get_final_generation_prompt, get_system_prompt, get_few_shot_prompt,
                      get_few_shot_prompt_fixed, get_dynamic_few_shot_prompt)
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
    
    def _initialize_pinecone_index(self):
        """Override BaseRAG method to do nothing for PlainLLM."""
        pass
        
    def _load_data_into_pinecone(self):
        """Override BaseRAG method to do nothing for PlainLLM."""
        pass
    
    def generate_docstring(self, user_code: str) -> Tuple[str, CostMetrics]:
        """
        Generate docstring using only the LLM.
        """
        self.api_call_count = 0  # Reset counter for this run
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
            api_calls=self.api_call_count,
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
            self.api_call_count += 1
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

class FewShotPlainLLM(PlainLLM):
    """Plain LLM utilizing Few-Shot prompt templates instead of bare system prompts."""
    def _generate_final_docstring(self, context: str, user_code: str, rewritten_req: str) -> str:
        """Generate the docstring using the few-shot template."""
        messages = [
            {'role': 'system', 'content': "You are an expert technical writer and Python developer. Follow the structure of the provided examples exactly."},
            {'role': 'user', 'content': get_few_shot_prompt(user_code)}
        ]

        try:
            self.api_call_count += 1
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


class FixedFewShotPlainLLM(FewShotPlainLLM):
    """Static few-shot with the corrected generator persona (revision ablation).

    Same two static exemplars as FewShotPlainLLM, but without the judge-persona
    prompt bug, isolating the persona-conflict confound.
    """
    def _generate_final_docstring(self, context: str, user_code: str, rewritten_req: str) -> str:
        messages = [
            {'role': 'system', 'content': "You are an expert technical writer and Python developer."},
            {'role': 'user', 'content': get_few_shot_prompt_fixed(user_code)}
        ]

        try:
            self.api_call_count += 1
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


class DynamicFewShotPlainLLM(FewShotPlainLLM):
    """Few-shot with dynamically retrieved, structurally matched exemplars (R2.8).

    Exemplars are the most similar OTHER classes from the benchmark corpus
    (leave-one-out), paired with their human-written reference docstrings, so
    the demonstration matches the target's structural profile.
    """

    NUM_EXEMPLARS = 2
    MAX_EXEMPLAR_CODE_CHARS = 1500
    MAX_EXEMPLAR_DOC_CHARS = 800

    def __init__(self, index_name: str = None, custom_config: dict = None, namespace: str = None,
                 exemplar_pool_path: str = None):
        super().__init__(index_name, custom_config, namespace)
        self._pool_path = exemplar_pool_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "class_files_df.pkl")
        self._pool_codes = None
        self._pool_docs = None
        self._pool_embeddings = None

    def _ensure_exemplar_pool(self):
        if self._pool_embeddings is not None:
            return
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        pool = pd.read_pickle(self._pool_path)
        self._pool_codes = pool['Code_without_comments'].fillna('').astype(str).tolist()
        self._pool_docs = pool['Comments'].fillna('').astype(str).tolist()

        cache_folder = os.path.join(os.path.dirname(self._pool_path), os.pardir, "models", "all-MiniLM-L6-v2")
        try:
            encoder = SentenceTransformer(self.model_config.embedding_model, cache_folder=cache_folder)
        except Exception:
            encoder = SentenceTransformer(self.model_config.embedding_model)
        self._encoder = encoder
        self._pool_embeddings = encoder.encode(self._pool_codes, normalize_embeddings=True,
                                               show_progress_bar=False)
        self.logger.info(f"Dynamic few-shot exemplar pool ready: {len(self._pool_codes)} classes")

    def _select_exemplars(self, user_code: str):
        self._ensure_exemplar_pool()
        query = self._encoder.encode([user_code], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self._pool_embeddings @ query

        ranked = sims.argsort()[::-1]
        exemplars = []
        target_norm = " ".join(user_code.split())
        for idx in ranked:
            cand_norm = " ".join(self._pool_codes[idx].split())
            # leave-one-out: skip the target class itself
            if cand_norm == target_norm or cand_norm in target_norm or target_norm in cand_norm:
                continue
            if not self._pool_docs[idx].strip():
                continue
            exemplars.append((self._pool_codes[idx][:self.MAX_EXEMPLAR_CODE_CHARS],
                              self._pool_docs[idx][:self.MAX_EXEMPLAR_DOC_CHARS]))
            if len(exemplars) == self.NUM_EXEMPLARS:
                break
        return exemplars

    def _generate_final_docstring(self, context: str, user_code: str, rewritten_req: str) -> str:
        try:
            exemplars = self._select_exemplars(user_code)
        except Exception as e:
            self.logger.error(f"Exemplar selection failed, falling back to static examples: {e}")
            return super()._generate_final_docstring(context, user_code, rewritten_req)

        messages = [
            {'role': 'system', 'content': "You are an expert technical writer and Python developer."},
            {'role': 'user', 'content': get_dynamic_few_shot_prompt(user_code, exemplars)}
        ]

        try:
            self.api_call_count += 1
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
