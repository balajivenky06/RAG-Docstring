"""
Reasoning Mixins for RAG strategies.
These mixins provide advanced reasoning capabilities (CoT, ToT, GoT)
that can be applied to any RAG implementation.
Includes robust retry logic to prevent silent Ollama timeout failures during heavy generation.
"""

import re
import time
from typing import List, Dict, Any

from .prompts import (
    get_cot_prompt, 
    get_tot_decomposition_prompt, get_tot_generation_prompt, get_tot_evaluation_prompt,
    get_got_axis_analysis_prompt, get_got_aggregation_prompt,
    get_system_prompt
)

class CoTMixin:
    """
    Chain of Thought reasoning mixin.
    Enforces step-by-step reasoning before generating the final docstring.
    """
    
    def _generate_final_docstring(self, context: str, user_code: str, rewritten_req: str) -> str:
        """
        Generate docstring using Chain of Thought prompting with retries.
        """
        messages = [
            {'role': 'system', 'content': get_system_prompt('docstring_generator')}
        ]
        
        cot_prompt = get_cot_prompt(user_code)
        if context:
            cot_prompt = f"Relevant Context:\n{context}\n\n{cot_prompt}"
            
        messages.append({'role': 'user', 'content': cot_prompt})
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.ollama_client.chat(
                    model=self.model_config.generator_model,
                    messages=messages,
                    options={'temperature': self.model_config.temperature}
                )
                if hasattr(self, 'api_call_count'):
                    self.api_call_count += 1
                
                full_response = response.get('message', {}).get('content', '').strip()
                docstring_match = re.search(r'\[DOCSTRING\](.*?)\[/DOCSTRING\]', full_response, re.DOTALL)
                
                if docstring_match:
                    generated_docstring = docstring_match.group(1).strip()
                else:
                    self.logger.warning("CoT tags not found, falling back to cleaning full response")
                    generated_docstring = self._clean_docstring_output(full_response)
                
                return generated_docstring
                
            except Exception as e:
                self.logger.warning(f"Ollama CoT Generation failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Error communicating with Ollama (CoT): {e}")
                    return "# ERROR: CoT Docstring generation failed."
                time.sleep(2)


class ToTMixin:
    """
    Tree of Thought reasoning mixin.
    Decomposes task, generates candidates, evaluates, and selects best path.
    """
    
    def _generate_final_docstring(self, context: str, user_code: str, rewritten_req: str) -> str:
        """
        Generate docstring using Tree of Thought approach with retries.
        """
        subtasks = self._decompose_task(user_code)
        self.logger.info(f"ToT Decomposition: {subtasks}")
        collected_thoughts = []
        current_context = context if context is not None else "" 
        
        for task in subtasks:
            candidates = self._generate_candidates(task, user_code, current_context, k=3)
            best_candidate = self._evaluate_and_select(candidates, task)
            collected_thoughts.append(f"Task: {task}\nResult: {best_candidate}")
            current_context += f"\nCompleted {task}: {best_candidate}"
            
        final_prompt = f"""Based on the following analysis steps:
        {chr(10).join(collected_thoughts)}
        
        Generate the final Python docstring for the code:
        {user_code}
        
        Return ONLY the docstring.
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.ollama_client.generate(
                    model=self.model_config.generator_model,
                    prompt=final_prompt
                )
                if hasattr(self, 'api_call_count'):
                    self.api_call_count += 1
                return self._clean_docstring_output(response.get('response', ''))
                
            except Exception as e:
                self.logger.warning(f"Ollama ToT Synthesis failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Error in ToT synthesis: {e}")
                    return "# ERROR: ToT Generation failed"
                time.sleep(2)

    def _decompose_task(self, code: str) -> List[str]:
        prompt = get_tot_decomposition_prompt(code)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.ollama_client.generate(
                    model=self.model_config.helper_model,
                    prompt=prompt
                )
                if hasattr(self, 'api_call_count'):
                    self.api_call_count += 1
                text = response.get('response', '')
                tasks = [line.strip().lstrip('- 1234567890.').strip() for line in text.split('\n') if line.strip()]
                return tasks[:3]
            except Exception as e:
                self.logger.warning(f"ToT Decomposition failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return ["Analyze parameters", "Identify return values", "Draft summary"]
                time.sleep(2)

    def _generate_candidates(self, task: str, code: str, context: str, k: int = 3) -> List[str]:
        candidates = []
        prompt = get_tot_generation_prompt(task, code, context)
        max_retries = 2 # lower retries here to prevent extreme hanging on K-loops
        
        for _ in range(k):
            for attempt in range(max_retries):
                try:
                    response = self.ollama_client.generate(
                        model=self.model_config.generator_model,
                        prompt=prompt,
                        options={'temperature': 0.7}
                    )
                    if hasattr(self, 'api_call_count'):
                        self.api_call_count += 1
                    candidates.append(response.get('response', '').strip())
                    break # Success, break out of retry loop for this candidate
                except Exception as e:
                    self.logger.warning(f"ToT Candidate Generation failed (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        pass # Silently skip this candidate branch if all retries fail
                    time.sleep(1)
        return candidates

    def _evaluate_and_select(self, candidates: List[str], task: str) -> str:
        if not candidates:
            return "No result"
            
        best_score = -1
        best_candidate = candidates[0]
        max_retries = 2
        
        for cand in candidates:
            eval_prompt = get_tot_evaluation_prompt(cand)
            for attempt in range(max_retries):
                try:
                    response = self.ollama_client.generate(
                        model=self.model_config.helper_model,
                        prompt=eval_prompt
                    )
                    if hasattr(self, 'api_call_count'):
                        self.api_call_count += 1
                    score_match = re.search(r"(\d+(\.\d+)?)", response.get('response', '0'))
                    score = float(score_match.group(1)) if score_match else 0
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = cand
                    break # Success, break out of retry loop for this evaluation
                except Exception as e:
                    if attempt == max_retries - 1:
                        pass # Default to worst score on complete failure
                    time.sleep(1)
                
        return best_candidate


class GoTMixin:
    """
    Graph of Thought reasoning mixin.
    Parallel analysis of axes (Params, Returns, etc.) followed by aggregation.
    """
    
    def _generate_final_docstring(self, context: str, user_code: str, rewritten_req: str) -> str:
        axes = ["Parameters", "Returns", "Functionality", "Exceptions"]
        analyses = {}
        
        for axis in axes:
            analyses[axis] = self._analyze_axis(axis, user_code)
            
        analyses_text = "\n\n".join([f"## {k}\n{v}" for k, v in analyses.items()])
        prompt = get_got_aggregation_prompt(analyses_text, user_code)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.ollama_client.generate(
                    model=self.model_config.generator_model,
                    prompt=prompt
                )
                if hasattr(self, 'api_call_count'):
                    self.api_call_count += 1
                docstring = response.get('response', '').strip()
                return self._clean_docstring_output(docstring)
                
            except Exception as e:
                self.logger.warning(f"Ollama GoT Synthesis failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Error in GoT synthesis: {e}")
                    return "# ERROR: GoT Generation failed"
                time.sleep(2)

    def _analyze_axis(self, axis: str, code: str) -> str:
        prompt = get_got_axis_analysis_prompt(axis, code)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.ollama_client.generate(
                    model=self.model_config.helper_model,
                    prompt=prompt
                )
                if hasattr(self, 'api_call_count'):
                    self.api_call_count += 1
                    self.logger.debug(f"API Call Count Incremented: {self.api_call_count}")
                return response.get('response', '').strip()
            except Exception as e:
                self.logger.warning(f"GoT Axis '{axis}' failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return f"Analysis for {axis} failed."
                time.sleep(2)
