"""Enhanced prompts for RAG-based docstring generation."""

ENHANCED_PROMPTS = {
    "system_docstring_generator": """You are an expert Python programmer and documentation specialist with deep knowledge of PEP 257 standards and Google-style docstring formatting.

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
- Provide helpful examples when appropriate""",
    "context_query_enhanced": """Generate a comprehensive search query to find relevant documentation for creating Python docstrings.

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

**Generate a focused search query for this code.**""",
    "final_generation_enhanced": """Generate a comprehensive Python docstring following PEP 257 standards and Google-style formatting.

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

**Generate the complete docstring following these requirements.**""",
    "critique_enhanced": """Evaluate the quality of the generated Python docstring against the original code using comprehensive criteria.

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

**Assessment (EXCELLENT/GOOD/NEEDS_IMPROVEMENT):**""",
}
