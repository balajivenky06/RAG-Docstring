"""
Prompt templates for RAG-based docstring generation system.
All prompts are centralized here for easy management and modification.
"""

from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PromptTemplates:
    """Container for all prompt templates."""
    
    # System prompts - Enhanced versions for better output quality
    SYSTEM_PROMPT_DOCSTRING_GENERATOR = """You are an expert Python programmer and documentation specialist with deep knowledge of PEP 257 standards and Google-style docstring formatting.

**Your Role:**
- Generate comprehensive, accurate docstrings following PEP 257 standards
- Use Google-style docstring format with proper sections
- Include type hints and detailed parameter descriptions
- Document exceptions and return values thoroughly

**Context Usage:**
- Use retrieved context only when directly relevant to the code
- Incorporate context insights to enhance accuracy
- Avoid including irrelevant information from context

**Output Requirements:**
- Return ONLY the docstring content starting with triple quotes
- Do NOT include the original Python code
- Follow proper indentation and formatting
- Include all necessary sections: Args, Returns, Raises, Examples

**Quality Standards:**
- Be concise but comprehensive
- Use clear, professional language
- Include type hints for parameters and returns
- Document all parameters (except self/cls)
- Mention exceptions that may be raised
- Provide helpful examples when appropriate"""

    SYSTEM_PROMPT_CRITIQUE = """You are an expert code reviewer and documentation specialist with extensive experience in Python docstring quality assessment and PEP 257 standards.

**Your Expertise:**
- Python docstring quality evaluation
- PEP 257 compliance assessment
- Code-documentation consistency analysis
- Documentation best practices

**Evaluation Framework:**
Assess docstrings on multiple dimensions:

1. **Completeness**: All parameters, returns, exceptions documented
2. **Accuracy**: Descriptions match actual code functionality
3. **Formatting**: PEP 257 compliance and proper structure
4. **Clarity**: Clear, professional language and organization
5. **Usefulness**: Helps developers understand and use the code

**Quality Standards:**
- HIGH QUALITY: Meets all criteria with excellence
- NEEDS IMPROVEMENT: Fails multiple criteria or has significant issues

**Assessment Process:**
- Analyze code structure and functionality
- Compare docstring against implementation
- Check for missing documentation elements
- Evaluate language clarity and professionalism
- Assess formatting compliance"""

    SYSTEM_PROMPT_REWRITE = """You are an expert prompt engineer specializing in optimizing prompts for AI docstring generation, with deep understanding of context integration and instruction clarity.

**Your Expertise:**
- Prompt optimization for AI systems
- Context integration strategies
- Instruction clarity enhancement
- Quality-focused prompt design

**Optimization Principles:**
1. **Context Integration**: Strategically incorporate relevant context without overwhelming the prompt
2. **Clarity Enhancement**: Make instructions specific, actionable, and unambiguous
3. **Quality Focus**: Emphasize high-quality documentation standards and requirements
4. **Format Specification**: Ensure clear output format and structure requirements

**Rewrite Strategies:**
- Extract key insights and patterns from context
- Add specific formatting and structure requirements
- Include comprehensive quality criteria
- Provide clear examples and guidance
- Maintain conciseness while maximizing clarity
- Focus on actionable, measurable instructions

**Output Requirements:**
- Generate self-contained, optimized prompts
- Focus specifically on docstring generation tasks
- Include relevant context insights strategically
- Ensure clear, actionable instructions
- Maintain professional tone and clarity"""

    # Context query generation prompts - Enhanced for better retrieval
    CONTEXT_QUERY_TEMPLATE = """Generate a comprehensive search query to find relevant documentation for creating Python docstrings.

**Code Analysis:**
Analyze the provided Python code to extract:
- Function/class name and purpose
- Parameter names and types
- Return type and value
- Exception handling
- Key functionality

**Search Strategy:**
Create a query that will retrieve:
- PEP 257 documentation and examples
- Similar function documentation patterns
- Parameter documentation best practices
- Return value documentation examples
- Exception documentation standards

**Query Components:**
- Include function/class name
- Add parameter names
- Mention return type if identifiable
- Include "docstring" and "PEP 257" keywords
- Add "python documentation" for broader context

**Code to analyze:**
{code}

**Generate a focused search query for this code.**"""

    # Initial generation prompts (for Self-RAG)
    INITIAL_GENERATION_TEMPLATE = """Generate a concise and accurate Python docstring for the following code. Focus only on the code provided.
Return only the docstring for the given code, dont give or return the code.
Python Code:
---
{code}
---

Generate only the docstring content, formatted appropriately:"""

    # Rewrite prompts
    REWRITE_PROMPT_TEMPLATE = """You are a helpful assistant that refines prompts. Given the following context from the RAG knowledge base along with python code: {context}, generate an optimized prompt for another AI whose sole task is to create a Python docstring for the code and your output.
The optimized prompt should clearly state the task, and subtly incorporate hints from the context if relevant, without necessarily repeating the entire context.
Focus on creating a self-contained, clear instruction for the next AI.

Generate only the optimized context prompt text for the docstring generation AI."""

    # Critique prompts - Enhanced with comprehensive evaluation criteria
    CRITIQUE_PROMPT_TEMPLATE = """Evaluate the quality of the generated Python docstring against the original code using comprehensive criteria.

**Evaluation Framework:**
Assess the docstring on these dimensions (each 0-2 points):

1. **Completeness (0-2 points)**:
   - Documents all parameters with types
   - Describes return value and type
   - Mentions exceptions if raised
   - Includes extended description if needed

2. **Accuracy (0-2 points)**:
   - Correctly describes code functionality
   - Parameter descriptions match implementation
   - Return type matches actual return
   - Exception documentation is accurate

3. **Formatting (0-2 points)**:
   - Follows PEP 257 standards
   - Proper triple quote usage
   - Correct indentation and structure

4. **Clarity (0-2 points)**:
   - Clear, professional language
   - Concise but comprehensive
   - Easy to understand

**Scoring:**
- EXCELLENT (7-8 points): High-quality docstring
- GOOD (5-6 points): Solid docstring with minor improvements
- NEEDS_IMPROVEMENT (0-4 points): Significant issues requiring revision

**Code to evaluate:**
---
{code}
---

**Generated docstring:**
---
{initial_docstring}
---

**Parameters to check: {param_names}**

**Assessment (EXCELLENT/GOOD/NEEDS_IMPROVEMENT):**"""

    # Relevance evaluation prompts
    RELEVANCE_EVALUATION_TEMPLATE = """Task: Evaluate if the following Context is relevant for generating a Python docstring for the provided Code.
Answer ONLY with YES or NO.

Context (from '{doc_source}'): --- {doc_text}... ---
Code: --- {code} ---
Is the Context relevant for generating a docstring for the Code (YES or NO):"""

    # Web search query templates
    WEB_SEARCH_QUERY_TEMPLATE = "python {context_query} documentation for `{code_first_line}`"

    # Final generation prompts - Enhanced for better quality
    FINAL_GENERATION_TEMPLATE = """Generate a comprehensive Python docstring following PEP 257 standards and Google-style formatting.

**CRITICAL REQUIREMENT:**
- Return ONLY the docstring content (the text between triple quotes)
- Do NOT include the original Python code
- Do NOT include the function/class definition
- Do NOT include any code implementation

**Context Integration:**
- Use any relevant context provided earlier to enhance accuracy
- Incorporate context insights into parameter descriptions
- Ensure context relevance to the specific code

**Docstring Requirements:**
1. **Format**: Use triple quotes with proper indentation
2. **Style**: Follow Google-style docstring format
3. **Sections**: Include all applicable sections:
   - Summary line (one-line description)
   - Extended description (if needed)
   - Args: Parameter descriptions with types
   - Returns: Return value description with type
   - Raises: Exception documentation
   - Examples: Usage examples (if helpful)

**Quality Standards:**
- Be concise but comprehensive
- Use clear, professional language
- Include type hints for parameters and returns
- Document all parameters (except self/cls)
- Mention exceptions that may be raised
- Provide helpful examples when appropriate

**Code to document:**
```python
{code}
```

**IMPORTANT: Your response should start and end with triple quotes and contain ONLY the docstring text. Do not include the original code.**"""

    # Context inclusion prompts
    CONTEXT_INCLUSION_TEMPLATE = """Here is potentially relevant context retrieved from knowledge base:
---
{context}
---

For the python code:
{code}

Based on this context, generate an appropriate docstring."""

    # Error handling prompts
    ERROR_PROMPT_TEMPLATE = """An error occurred during docstring generation: {error_message}
Please provide a basic docstring for the following Code:
{code}"""

    # --- Advanced Reasoning Prompts ---

    # Chain of Thought (CoT)
    COT_GENERATION_TEMPLATE = """Generate a Python docstring for the code below.
    
    **Instructions:**
    1. **Think Step-by-Step**: First, analyze the code logic, parameters, returns, and exceptions. Explain your reasoning clearly.
    2. **Generate Docstring**: Based on your analysis, generate the final docstring.
    
    **Format:**
    [REASONING]
    <Your step-by-step analysis here>
    [/REASONING]
    
    [DOCSTRING]
    <Your final docstring here (triple-quoted)>
    [/DOCSTRING]
    
    **Code:**
    {code}
    """

    # Tree of Thought (ToT)
    TOT_DECOMPOSITION_PROMPT = """Analyze the following code and break down the task of generating a docstring into 3 distinct sub-tasks.
    Return ONLY a list of sub-tasks.
    
    Code:
    {code}
    """
    
    TOT_GENERATION_PROMPT = """Generate a candidate docstring section for the following task: "{task}".
    
    Context:
    {context}
    
    Code:
    {code}
    """
    
    TOT_EVALUATION_PROMPT = """Evaluate the quality of the following docstring candidate on a scale of 0 to 1.
    
    Candidate:
    {candidate}
    
    Criteria: Accuracy, Completeness, Clarity.
    Return ONLY the numeric score (e.g., 0.85).
    """

    # Graph of Thought (GoT)
    GOT_AXIS_ANALYSIS_PROMPT = """Analyze the following code solely from the perspective of: **{axis}**.
    
    Axis Definition:
    - **Parameters**: Focus on arguments, types, and defaults.
    - **Returns**: Focus on return values, types, and yields.
    - **Functionality**: Focus on what the code does, algorithms, and logic.
    - **Exceptions**: Focus on errors raised and edge cases.
    
    Provide a detailed analysis for this specific axis.
    
    Code:
    {code}
    """
    
    GOT_AGGREGATION_PROMPT = """Synthesize the following specific analyses into a single, cohesive Python docstring.
    
    **Analyses:**
    {analyses}
    
    **Code:**
    {code}
    
    Generate the final docstring following PEP 257 standards. Return ONLY the docstring.
    """

    # Parameter extraction prompts
    PARAMETER_EXTRACTION_TEMPLATE = """Extract parameter information from the following Python code:
{code}

Return a JSON object with:
- function_name: name of the function
- parameters: list of parameter names
- return_type: inferred return type
- exceptions: list of raised exceptions"""

    # Code analysis prompts
    CODE_ANALYSIS_TEMPLATE = """Analyze the following Python code and provide:
1. Class/function name
2. Purpose/functionality
3. Parameters and their types
4. Return value and type
5. Exceptions that might be raised
6. Key implementation details

Code:
{code}"""

class PromptManager:
    """Manager class for handling prompts and templates."""
    
    def __init__(self):
        self.templates = PromptTemplates()
    
    def get_context_query(self, code: str) -> str:
        """Generate context query for docstring generation."""
        return self.templates.CONTEXT_QUERY_TEMPLATE.format(code=code)
    
    def get_initial_generation_prompt(self, code: str) -> str:
        """Get initial generation prompt for Self-RAG."""
        return self.templates.INITIAL_GENERATION_TEMPLATE.format(code=code)
    
    def get_rewrite_prompt(self, context: str) -> str:
        """Get prompt rewrite template."""
        return self.templates.REWRITE_PROMPT_TEMPLATE.format(context=context)
    
    def get_critique_prompt(self, code: str, initial_docstring: str, param_names: List[str] = None) -> str:
        """Get critique prompt for Self-RAG."""
        param_names_str = ', '.join(param_names) if param_names else 'None'
        return self.templates.CRITIQUE_PROMPT_TEMPLATE.format(
            code=code,
            initial_docstring=initial_docstring,
            param_names=param_names_str
        )
    
    def get_relevance_evaluation_prompt(self, doc_text: str, doc_source: str, code: str) -> str:
        """Get relevance evaluation prompt."""
        return self.templates.RELEVANCE_EVALUATION_TEMPLATE.format(
            doc_text=doc_text[:1500],  # Truncate for context
            doc_source=doc_source,
            code=code
        )
    
    def get_web_search_query(self, context_query: str, code_first_line: str) -> str:
        """Get web search query."""
        return self.templates.WEB_SEARCH_QUERY_TEMPLATE.format(
            context_query=context_query,
            code_first_line=code_first_line
        )
    
    def get_final_generation_prompt(self, code: str) -> str:
        """Get final generation prompt."""
        return self.templates.FINAL_GENERATION_TEMPLATE.format(code=code)
    
    def get_context_inclusion_prompt(self, context: str, code: str) -> str:
        """Get context inclusion prompt."""
        return self.templates.CONTEXT_INCLUSION_TEMPLATE.format(context=context, code=code)
    
    def get_error_prompt(self, error_message: str, code: str) -> str:
        """Get error handling prompt."""
        return self.templates.ERROR_PROMPT_TEMPLATE.format(
            error_message=error_message,
            code=code
        )
    
    def get_parameter_extraction_prompt(self, code: str) -> str:
        """Get parameter extraction prompt."""
        return self.templates.PARAMETER_EXTRACTION_TEMPLATE.format(code=code)
    
    def get_code_analysis_prompt(self, code: str) -> str:
        """Get code analysis prompt."""
        return self.templates.CODE_ANALYSIS_TEMPLATE.format(code=code)
    
    def get_system_prompt(self, prompt_type: str) -> str:
        """Get system prompt by type."""
        system_prompts = {
            'docstring_generator': self.templates.SYSTEM_PROMPT_DOCSTRING_GENERATOR,
            'critique': self.templates.SYSTEM_PROMPT_CRITIQUE,
            'rewrite': self.templates.SYSTEM_PROMPT_REWRITE
        }
        return system_prompts.get(prompt_type, self.templates.SYSTEM_PROMPT_DOCSTRING_GENERATOR)
    
    # --- Advanced Reasoning Accessors ---
    
    def get_cot_prompt(self, code: str) -> str:
        return self.templates.COT_GENERATION_TEMPLATE.format(code=code)
        
    def get_tot_decomposition_prompt(self, code: str) -> str:
        return self.templates.TOT_DECOMPOSITION_PROMPT.format(code=code)

    def get_tot_generation_prompt(self, task: str, code: str, context: str = "") -> str:
        return self.templates.TOT_GENERATION_PROMPT.format(task=task, code=code, context=context)

    def get_tot_evaluation_prompt(self, candidate: str) -> str:
        return self.templates.TOT_EVALUATION_PROMPT.format(candidate=candidate)

    def get_got_axis_analysis_prompt(self, axis: str, code: str) -> str:
        return self.templates.GOT_AXIS_ANALYSIS_PROMPT.format(axis=axis, code=code)

    def get_got_aggregation_prompt(self, analyses: str, code: str) -> str:
        return self.templates.GOT_AGGREGATION_PROMPT.format(analyses=analyses, code=code)

    def format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with given parameters."""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"Missing parameter for prompt formatting: {e}")
            return template

# Global prompt manager instance
prompt_manager = PromptManager()

# Convenience functions for easy access
def get_context_query(code: str) -> str:
    """Get context query for given code."""
    return prompt_manager.get_context_query(code)

def get_initial_generation_prompt(code: str) -> str:
    """Get initial generation prompt."""
    return prompt_manager.get_initial_generation_prompt(code)

def get_rewrite_prompt(context: str) -> str:
    """Get rewrite prompt."""
    return prompt_manager.get_rewrite_prompt(context)

def get_critique_prompt(code: str, initial_docstring: str, param_names: List[str] = None) -> str:
    """Get critique prompt."""
    return prompt_manager.get_critique_prompt(code, initial_docstring, param_names)

def get_relevance_evaluation_prompt(doc_text: str, doc_source: str, code: str) -> str:
    """Get relevance evaluation prompt."""
    return prompt_manager.get_relevance_evaluation_prompt(doc_text, doc_source, code)

def get_web_search_query(context_query: str, code_first_line: str) -> str:
    """Get web search query."""
    return prompt_manager.get_web_search_query(context_query, code_first_line)

def get_final_generation_prompt(code: str) -> str:
    """Get final generation prompt."""
    return prompt_manager.get_final_generation_prompt(code)

def get_context_inclusion_prompt(context: str, code: str) -> str:
    """Get context inclusion prompt."""
    return prompt_manager.get_context_inclusion_prompt(context, code)

def get_system_prompt(prompt_type: str) -> str:
    """Get system prompt by type."""
    return prompt_manager.get_system_prompt(prompt_type)

# --- Advanced Reasoning Accessors ---
def get_cot_prompt(code: str) -> str:
    return prompt_manager.get_cot_prompt(code)

def get_tot_decomposition_prompt(code: str) -> str:
    return prompt_manager.get_tot_decomposition_prompt(code)

def get_tot_generation_prompt(task: str, code: str, context: str = "") -> str:
    return prompt_manager.get_tot_generation_prompt(task, code, context)

def get_tot_evaluation_prompt(candidate: str) -> str:
    return prompt_manager.get_tot_evaluation_prompt(candidate)

def get_got_axis_analysis_prompt(axis: str, code: str) -> str:
    return prompt_manager.get_got_axis_analysis_prompt(axis, code)

def get_got_aggregation_prompt(analyses: str, code: str) -> str:
    return prompt_manager.get_got_aggregation_prompt(analyses, code)
