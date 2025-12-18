"""
Simplified prompt validation and enhancement script.
"""

def analyze_current_prompts():
    """Analyze current prompts and provide recommendations."""
    
    print("Prompt Validation Analysis")
    print("=" * 50)
    
    # Current prompt analysis
    analyses = {
        'system_docstring_generator': {
            'current_score': 6,
            'strengths': ['Clear role definition', 'Specifies output format'],
            'weaknesses': ['Lacks PEP 257 specifics', 'No quality requirements', 'Missing examples'],
            'improvements': ['Add PEP 257 compliance', 'Include quality standards', 'Add formatting examples']
        },
        'context_query': {
            'current_score': 4,
            'strengths': ['Comprehensive instructions', 'Mentions PEP 257'],
            'weaknesses': ['Too verbose for query', 'Mixing concerns', 'No search strategy'],
            'improvements': ['Separate query from generation', 'Add search optimization', 'Focus on retrieval']
        },
        'final_generation': {
            'current_score': 5,
            'strengths': ['Clear output format', 'Mentions context usage'],
            'weaknesses': ['Lacks specific format', 'No quality requirements', 'Missing sections'],
            'improvements': ['Add docstring format', 'Include quality standards', 'Specify sections']
        },
        'critique': {
            'current_score': 5,
            'strengths': ['Clear task', 'Binary assessment'],
            'weaknesses': ['Limited criteria', 'No scoring', 'Missing standards'],
            'improvements': ['Add comprehensive criteria', 'Include scoring rubric', 'Add examples']
        }
    }
    
    # Print analysis
    for prompt_name, analysis in analyses.items():
        print(f"\n--- {prompt_name.upper()} ---")
        print(f"Current Score: {analysis['current_score']}/10")
        print(f"Strengths: {', '.join(analysis['strengths'])}")
        print(f"Weaknesses: {', '.join(analysis['weaknesses'])}")
        print(f"Improvements: {', '.join(analysis['improvements'])}")
    
    return analyses

def generate_enhanced_prompts():
    """Generate enhanced versions of key prompts."""
    
    enhanced_prompts = {
        'system_docstring_generator': """You are an expert Python programmer and documentation specialist with deep knowledge of PEP 257 standards and Google-style docstring formatting.

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

        'context_query_enhanced': """Generate a comprehensive search query to find relevant documentation for creating Python docstrings.

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

        'final_generation_enhanced': """Generate a comprehensive Python docstring following PEP 257 standards and Google-style formatting.

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

        'critique_enhanced': """Evaluate the quality of the generated Python docstring against the original code using comprehensive criteria.

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
    }
    
    return enhanced_prompts

def main():
    """Main validation function."""
    print("RAG Prompt Validation and Enhancement")
    print("=" * 60)
    
    # Analyze current prompts
    analyses = analyze_current_prompts()
    
    # Generate enhanced prompts
    enhanced = generate_enhanced_prompts()
    
    # Calculate overall scores
    total_score = sum(analysis['current_score'] for analysis in analyses.values())
    avg_score = total_score / len(analyses)
    
    print(f"\n--- Overall Analysis ---")
    print(f"Average Current Score: {avg_score:.1f}/10")
    print(f"Total Prompts Analyzed: {len(analyses)}")
    
    print(f"\n--- Key Findings ---")
    print("✓ System prompts need more specific role definitions")
    print("✓ Context query generation needs separation from docstring generation")
    print("✓ Final generation prompts lack comprehensive requirements")
    print("✓ Critique prompts need scoring methodology")
    print("✓ All prompts need better context integration strategies")
    
    print(f"\n--- Recommendations ---")
    print("1. Add specific PEP 257 compliance requirements to all prompts")
    print("2. Include comprehensive evaluation criteria for critique prompts")
    print("3. Enhance context integration strategies")
    print("4. Add quality scoring systems")
    print("5. Provide clear formatting examples")
    
    # Save enhanced prompts
    try:
        with open('enhanced_prompts.py', 'w') as f:
            f.write('"""Enhanced prompts for RAG-based docstring generation."""\n\n')
            f.write('ENHANCED_PROMPTS = {\n')
            for name, prompt in enhanced.items():
                f.write(f'    "{name}": """{prompt}""",\n')
            f.write('}\n')
        print(f"\n✓ Enhanced prompts saved to enhanced_prompts.py")
    except Exception as e:
        print(f"✗ Error saving enhanced prompts: {e}")
    
    return enhanced

if __name__ == "__main__":
    main()