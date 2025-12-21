"""
Reasoning Mixins for RAG strategies.
These mixins provide advanced reasoning capabilities (CoT, ToT, GoT)
that can be applied to any RAG implementation.
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
        Generate docstring using Chain of Thought prompting.
        """
        messages = [
            {'role': 'system', 'content': get_system_prompt('docstring_generator')}
        ]
        
        # Construct detailed CoT prompt
        cot_prompt = get_cot_prompt(user_code)
        
        if context:
            cot_prompt = f"Relevant Context:\n{context}\n\n{cot_prompt}"
            
        messages.append({'role': 'user', 'content': cot_prompt})
        
        try:
            response = self.ollama_client.chat(
                model=self.model_config.generator_model,
                messages=messages,
                options={'temperature': self.model_config.temperature}
            )
            
            full_response = response.get('message', {}).get('content', '').strip()
            
            # Extract the actual docstring part
            docstring_match = re.search(r'\[DOCSTRING\](.*?)\[/DOCSTRING\]', full_response, re.DOTALL)
            
            if docstring_match:
                generated_docstring = docstring_match.group(1).strip()
            else:
                self.logger.warning("CoT tags not found, falling back to cleaning full response")
                generated_docstring = self._clean_docstring_output(full_response)
            
            return generated_docstring
            
        except Exception as e:
            self.logger.error(f"Error communicating with Ollama (CoT): {e}")
            return "# ERROR: CoT Docstring generation failed."


class ToTMixin:
    """
    Tree of Thought reasoning mixin.
    Decomposes task, generates candidates, evaluates, and selects best path.
    """
    
    def _generate_final_docstring(self, context: str, user_code: str, rewritten_req: str) -> str:
        """
        Generate docstring using Tree of Thought approach.
        """
        # Step 1: Decomposition
        subtasks = self._decompose_task(user_code)
        self.logger.info(f"ToT Decomposition: {subtasks}")
        
        collected_thoughts = []
        
        # Step 2: Generation & Evaluation for each subtask
        current_context = context if context is not None else "" 
        
        for task in subtasks:
            # Generate K candidates
            candidates = self._generate_candidates(task, user_code, current_context, k=3)
            
            # Evaluate candidates
            best_candidate = self._evaluate_and_select(candidates, task)
            collected_thoughts.append(f"Task: {task}\nResult: {best_candidate}")
            
            # Update context for next step
            current_context += f"\nCompleted {task}: {best_candidate}"
            
        # Step 3: Final Synthesis
        final_prompt = f"""Based on the following analysis steps:
        {chr(10).join(collected_thoughts)}
        
        Generate the final Python docstring for the code:
        {user_code}
        
        Return ONLY the docstring.
        """
        
        try:
            response = self.ollama_client.generate(
                model=self.model_config.generator_model,
                prompt=final_prompt
            )
            return self._clean_docstring_output(response.get('response', ''))
            
        except Exception as e:
            self.logger.error(f"Error in ToT synthesis: {e}")
            return "# ERROR: ToT Generation failed"

    def _decompose_task(self, code: str) -> List[str]:
        """Ask LLM to decompose the task."""
        prompt = get_tot_decomposition_prompt(code)
        try:
            response = self.ollama_client.generate(
                model=self.model_config.helper_model,
                prompt=prompt
            )
            text = response.get('response', '')
            tasks = [line.strip().lstrip('- 1234567890.').strip() for line in text.split('\n') if line.strip()]
            return tasks[:3]
        except Exception:
            return ["Analyze parameters", "Identify return values", "Draft summary"]

    def _generate_candidates(self, task: str, code: str, context: str, k: int = 3) -> List[str]:
        """Generate k candidate thoughts for a task."""
        candidates = []
        prompt = get_tot_generation_prompt(task, code, context)
        
        for _ in range(k):
            try:
                response = self.ollama_client.generate(
                    model=self.model_config.generator_model,
                    prompt=prompt,
                    options={'temperature': 0.7}
                )
                candidates.append(response.get('response', '').strip())
            except Exception:
                pass
        return candidates

    def _evaluate_and_select(self, candidates: List[str], task: str) -> str:
        """Select the best candidate using LLM evaluation."""
        if not candidates:
            return "No result"
            
        best_score = -1
        best_candidate = candidates[0]
        
        for cand in candidates:
            try:
                eval_prompt = get_tot_evaluation_prompt(cand)
                response = self.ollama_client.generate(
                    model=self.model_config.helper_model,
                    prompt=eval_prompt
                )
                score_match = re.search(r"(\d+(\.\d+)?)", response.get('response', '0'))
                score = float(score_match.group(1)) if score_match else 0
                
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            except Exception:
                continue
                
        return best_candidate


class GoTMixin:
    """
    Graph of Thought reasoning mixin.
    Parallel analysis of axes (Params, Returns, etc.) followed by aggregation.
    """
    
    def _generate_final_docstring(self, context: str, user_code: str, rewritten_req: str) -> str:
        """
        Generate docstring using Graph of Thought approach.
        """
        # Node 1: Parallel Analysis
        axes = ["Parameters", "Returns", "Functionality", "Exceptions"]
        analyses = {}
        
        for axis in axes:
            analyses[axis] = self._analyze_axis(axis, user_code)
            
        # Node 2: Aggregation
        analyses_text = "\n\n".join([f"## {k}\n{v}" for k, v in analyses.items()])
        
        prompt = get_got_aggregation_prompt(analyses_text, user_code)
        
        try:
            response = self.ollama_client.generate(
                model=self.model_config.generator_model,
                prompt=prompt
            )
            docstring = response.get('response', '').strip()
            
            return self._clean_docstring_output(docstring)
            
        except Exception as e:
            self.logger.error(f"Error in GoT synthesis: {e}")
            return "# ERROR: GoT Generation failed"

    def _analyze_axis(self, axis: str, code: str) -> str:
        """Generate analysis for a specific axis."""
        prompt = get_got_axis_analysis_prompt(axis, code)
        try:
            response = self.ollama_client.generate(
                model=self.model_config.helper_model,
                prompt=prompt
            )
            return response.get('response', '').strip()
        except Exception:
            return f"Analysis for {axis} failed."
