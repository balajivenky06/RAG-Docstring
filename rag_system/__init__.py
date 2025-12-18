"""
RAG-based Docstring Generation System

A comprehensive system for generating Python docstrings using different RAG strategies.
"""

from .base_rag import BaseRAG, CostMetrics
from .simple_rag import SimpleRAG

from .self_rag import SelfCorrectionRAG
SelfRAG = SelfCorrectionRAG
from .advanced_rag import CoTRAG, ToTRAG, GoTRAG
from .plain_llm import PlainLLM, CoTPlainLLM, ToTPlainLLM, GoTPlainLLM
from .evaluator import RAGEvaluator
from .cost_analyzer import CostAnalyzer
from .visualizer import RAGVisualizer
from .config import (
    config, SystemConfig, ModelConfig, PineconeConfig, RetrievalConfig, 
    EvaluationConfig, CostAnalysisConfig, PathConfig, LoggingConfig, 
    WebSearchConfig, RAGMethodConfig
)
from .prompts import prompt_manager, PromptTemplates, PromptManager

# Convenience imports
from .config import (
    get_model_config, get_pinecone_config, get_retrieval_config,
    get_evaluation_config, get_path_config, get_rag_method_config,
    get_index_name, get_index_namespace, get_knowledge_base_urls, get_common_rag_config, get_benchmark_config
)
from .prompts import (
    get_context_query, get_initial_generation_prompt, get_rewrite_prompt,
    get_critique_prompt, get_relevance_evaluation_prompt, get_web_search_query,
    get_final_generation_prompt, get_context_inclusion_prompt, get_system_prompt
)

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    # Core RAG classes
    "BaseRAG",
    "SimpleRAG", 

    "SelfCorrectionRAG",
    "SelfRAG",
    "CostMetrics",
    
    # Evaluation and analysis
    "RAGEvaluator",
    "CostAnalyzer",
    "RAGVisualizer",
    
    # Configuration classes
    "config",
    "SystemConfig",
    "ModelConfig",
    "PineconeConfig",
    "RetrievalConfig",
    "EvaluationConfig",
    "CostAnalysisConfig",
    "PathConfig",
    "LoggingConfig",
    "WebSearchConfig",
    "RAGMethodConfig",
    
    # Prompt management
    "CoTRAG",
    "ToTRAG",
    "GoTRAG",
    "PlainLLM",
    "CoTPlainLLM",
    "ToTPlainLLM",
    "GoTPlainLLM",
    "get_cot_prompt", 
    "get_tot_decomposition_prompt", 
    "get_tot_generation_prompt", 
    "get_tot_evaluation_prompt",
    "get_got_axis_analysis_prompt", 
    "get_got_aggregation_prompt",
    "get_initial_generation_prompt",
    "get_critique_prompt",
    "get_final_generation_prompt",
    "get_context_inclusion_prompt",
    "get_system_prompt"
]

# Export advanced variants
from .advanced_rag_variants import (
    CoTSelfCorrectionRAG, ToTSelfCorrectionRAG, GoTSelfCorrectionRAG
)

__all__.extend([
    "CoTSelfCorrectionRAG", "ToTSelfCorrectionRAG", "GoTSelfCorrectionRAG"
])
