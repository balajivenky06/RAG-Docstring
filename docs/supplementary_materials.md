# Supplementary Materials: Implementation Details

## A. Prompt Templates
The following prompt templates were used for the RAG-based docstring generation experiments.

### 1. System Prompts
**Docstring Generator:**
```text
You are an expert Python programmer and documentation specialist with deep knowledge of PEP 257 standards and Google-style docstring formatting.

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
- Provide helpful examples when appropriate
```

**Critique Agent (Self-RAG):**
```text
You are an expert code reviewer and documentation specialist with extensive experience in Python docstring quality assessment and PEP 257 standards.

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
```

### 2. Generation Prompts
**Context Query Generation:**
```text
Generate a comprehensive search query to find relevant documentation for creating Python docstrings.

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

**Generate a focused search query for this code.**
```

**Final Docstring Generation:**
```text
Generate a comprehensive Python docstring following PEP 257 standards and Google-style formatting.

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

**IMPORTANT: Your response should start and end with triple quotes and contain ONLY the docstring text. Do not include the original code.**
```

## B. Hyperparameters and Configuration

### 1. Retrieval Settings
*   **Top-K Retrieved Documents**: 3 (Default)
*   **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
*   **Vector Database**: Pinecone (Cosine Similarity metric)

### 2. Generation Settings
*   **Generator Model**: `deepseek-coder:6.7b` (Default)
*   **Helper Model**: `deepseek-coder:1.3b` (For query rewriting/critique)
*   **Temperature**: 0.2 (Low temperature for deterministic, factual output)
*   **Max Tokens**: Model default

### 3. RAG Specific Settings
*   **Simple RAG**: Direct retrieval + generation.
*   **Code-Aware RAG**: Entity extraction -> Enriched Query -> Retrieval -> Generation.
*   **Corrective RAG**: Retrieval -> Relevance Check -> (Web Search if needed) -> Generation.
*   **Self-RAG**: Retrieval -> Generation -> Critique -> (Rewrite if needed) -> Final Selection.
*   **Fusion RAG**: Multiple Query Generation -> Reciprocal Rank Fusion -> Generation.
