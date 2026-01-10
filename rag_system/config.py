"""
Configuration file for RAG-based docstring generation system.
All system configurations are centralized here for easy management.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

# Set environment variable to suppress tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Base directory
BASE_DIR = Path(__file__).parent

@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    embedding_model: str = 'all-MiniLM-L6-v2'
    generator_model: str = 'llama3.1:8b'
    helper_model: str = 'llama3.2:latest'
    temperature: float = 0.5
    max_tokens: int = 2048
    top_p: float = 0.9

@dataclass
class PineconeConfig:
    """Configuration for Pinecone vector database."""
    api_key: str = "pcsk_71bnuL_HGU1YACobTvL5gJNzHsZG1NMNx3RGmz1ohyC7xMiUYoWnuZpEn5SuvWpuTxnuzm"
    environment: str = "us-east-1"
    vector_dimension: int = 384
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"
    namespace: str = ""

@dataclass
class RetrievalConfig:
    """Configuration for retrieval parameters."""
    top_k: int = 3
    max_context_length: int = 4000
    similarity_threshold: float = 0.7
    rerank_top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    rouge_weights: tuple = (0.25, 0.25, 0.25, 0.25)
    bleu_weights: tuple = (0.25, 0.25, 0.25, 0.25)
    bert_model: str = 'bert-base-uncased'
    bert_lang: str = 'en'
    pydocstyle_enabled: bool = True
    faithfulness_threshold: float = 0.3

@dataclass
class CostAnalysisConfig:
    """Configuration for cost analysis."""
    track_memory: bool = True
    track_cpu: bool = True
    track_api_calls: bool = True
    track_tokens: bool = True
    efficiency_weights: Dict[str, float] = field(default_factory=lambda: {
        'execution_time': 0.4,
        'memory_usage': 0.2,
        'cpu_usage': 0.2,
        'api_calls': 0.2
    })

@dataclass
class PathConfig:
    """Configuration for file paths."""
    results_dir: str = "results"
    evaluation_dir: str = "evaluation"
    visualization_dir: str = "visualizations"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    temp_dir: str = "temp"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for attr_name in ['results_dir', 'evaluation_dir', 'visualization_dir', 'logs_dir', 'cache_dir', 'temp_dir']:
            dir_path = getattr(self, attr_name)
            os.makedirs(dir_path, exist_ok=True)

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: bool = True
    console_logging: bool = True
    log_file: str = "rag_system.log"

@dataclass
class WebSearchConfig:
    """Configuration for web search functionality."""
    enabled: bool = True
    max_results: int = 3
    timeout: int = 15
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    max_snippet_length: int = 4000
    delay_between_requests: float = 1.5

@dataclass
class RAGMethodConfig:
    """Configuration for specific RAG methods - all methods use same base parameters for fair comparison."""
    
    # Common parameters for all RAG methods (for fair benchmarking)
    common_config: Dict[str, any] = field(default_factory=lambda: {
        'top_k': 3,  # Same retrieval count for all methods
        'use_rewrite': True,  # All methods use prompt rewriting
        'rewrite_temperature': 0.3,  # Same temperature for rewriting
        'generation_temperature': 0.5,  # Same temperature for generation
        'max_context_length': 4000,  # Same context length limit
        'similarity_threshold': 0.7,  # Same similarity threshold
        'web_search_enabled': True,  # All methods can use web search
        'web_search_max_results': 3,  # Same web search results
        'web_search_timeout': 15,  # Same timeout
        'min_chunk_length': 10,  # Same minimum chunk length
        'chunk_and_regrade': True,  # All methods use chunk grading
        'evaluation_enabled': True,  # All methods use relevance evaluation
    })
    
    # Method-specific parameters (only where absolutely necessary)
    simple_rag: Dict[str, any] = field(default_factory=lambda: {
        'method_specific': 'basic_semantic_search'
    })
    
    code_aware_rag: Dict[str, any] = field(default_factory=lambda: {
        'extract_entities': True,
        'enrich_query': True,
        'include_code_in_query': True,
        'max_methods_in_query': 3
    })
    
    corrective_rag: Dict[str, any] = field(default_factory=lambda: {
        'initial_top_k': 3,  # Same as common top_k
        'web_search_fallback': True
    })
    
    fusion_rag: Dict[str, any] = field(default_factory=lambda: {
        'semantic_top_k': 3,  # Same as common top_k
        'keyword_top_k': 3,  # Same as common top_k
        'rrf_k': 60,
        'bm25_enabled': True,
        'bm25_corpus_urls': [
            "https://peps.python.org/pep-0257/",
            "https://google.github.io/styleguide/pyguide.html",
            "https://www.programiz.com/python-programming/docstrings"
        ]
    })
    
    self_correction_rag: Dict[str, any] = field(default_factory=lambda: {
        'initial_generation_enabled': True,
        'self_critique_enabled': True,
        'adaptive_rag_enabled': True,
        'critique_temperature': 0.0,
        'web_search_fallback': True
    })

@dataclass
class SystemConfig:
    """Main system configuration combining all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cost_analysis: CostAnalysisConfig = field(default_factory=CostAnalysisConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    rag_methods: RAGMethodConfig = field(default_factory=RAGMethodConfig)
    
    # Knowledge base URLs
    knowledge_base_urls: List[str] = field(default_factory=lambda: [
        "https://peps.python.org/pep-0257/",
        "https://www.kaggle.com/code/hagzilla/what-are-docstrings",
        "https://github.com/keleshev/pep257/blob/master/pep257.py",
        "https://github.com/chadrik/doc484",
        "https://zerotomastery.io/blog/python-docstring/",
        "https://google.github.io/styleguide/pyguide.html",
        "https://www.geeksforgeeks.org/python-docstrings/",
        "https://pandas.pydata.org/docs/development/contributing_docstring.html",
        "https://www.coding-guidelines.lftechnology.com/docs/python/docstrings/",
        "https://realpython.com/python-pep8/",
        "https://pypi.org/project/AIDocStringGenerator/",
        "https://www.geeksforgeeks.org/pep-8-coding-style-guide-python/",
        "https://llego.dev/posts/write-python-docstrings-guide-documenting-functions/",
        "https://www.datacamp.com/tutorial/pep8-tutorial-python-code",
        "https://www.programiz.com/python-programming/docstrings",
        "https://marketplace.visualstudio.com/items?itemName=ShanthoshS.docstring-generator-ext",
        "https://stackoverflow.com/questions/3898572/what-are-the-most-common-python-docstring-formats",
        "https://stackoverflow.com/questions/78753860/what-is-the-proper-way-of-including-examples-in-python-docstrings",
        "https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html",
        "https://www.dataquest.io/blog/documenting-in-python-with-docstrings/",
        "https://www.tutorialspoint.com/python/python_docstrings.htm"
    ])
    
    # Index names for different RAG methods - using single index with namespaces
    index_names: Dict[str, str] = field(default_factory=lambda: {
        'simple': 'rag-docstring',  # Use existing index
        'code_aware': 'code-aware-rag-docstring',  # Use existing index
        'corrective': 'corrective-rag-docstring',  # Use existing index
        'fusion': 'fusion-rag-docstring',  # Use existing index
        'self_correction': 'self-correction-rag-docstring'  # Distinct index for SelfCorrectionRAG
    })
    
    # Namespaces for different RAG methods (to partition data in same index)
    index_namespaces: Dict[str, str] = field(default_factory=lambda: {
        'simple': 'simple_rag',
        'code_aware': 'code_aware_rag',
        'corrective': 'corrective_rag',
        'fusion': 'fusion_rag',
        'self_correction': 'self_correction_rag'
    })
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Load Pinecone API key from environment
        pinecone_key = os.getenv('PINECONE_API_KEY')
        if pinecone_key:
            self.pinecone.api_key = pinecone_key
        
        # Load model configurations from environment
        generator_model = os.getenv('GENERATOR_MODEL')
        if generator_model:
            self.model.generator_model = generator_model
            
        helper_model = os.getenv('HELPER_MODEL')
        if helper_model:
            self.model.helper_model = helper_model
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to a file."""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)
    
    def load_from_file(self, file_path: str) -> None:
        """Load configuration from a file."""
        import json
        from dataclasses import is_dataclass

        with open(file_path, 'r') as f:
            config_dict = json.load(f)
            # Update current configuration
            for key, value in config_dict.items():
                if hasattr(self, key):
                    current_attr = getattr(self, key)
                    # If it's a nested dataclass and value is a dict, update fields
                    if is_dataclass(current_attr) and isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if hasattr(current_attr, sub_key):
                                setattr(current_attr, sub_key, sub_value)
                    else:
                        # Simple value or non-dataclass, just set it
                        setattr(self, key, value)

# Global configuration instance
config = SystemConfig()

# 1. Load from config file if exists (Project-level defaults)
config_path = BASE_DIR.parent / "config" / "config.json"
if config_path.exists():
    try:
        config.load_from_file(str(config_path))
        print(f"Loaded configuration from {config_path}")
    except Exception as e:
        print(f"Warning: Failed to load config file: {e}")
else:
    print(f"No config file found at {config_path}, using defaults")

# 2. Load from environment variables (Overrides config file)
config.load_from_env()

# Configuration validation
def validate_config(config: SystemConfig) -> bool:
    """Validate the configuration."""
    errors = []
    
    # Check required fields
    if not config.pinecone.api_key:
        errors.append("Pinecone API key is required")
    
    if not config.model.generator_model:
        errors.append("Generator model is required")
    
    if not config.model.helper_model:
        errors.append("Helper model is required")
    
    # Check numeric ranges
    if not 0 <= config.model.temperature <= 2:
        errors.append("Temperature must be between 0 and 2")
    
    if config.retrieval.top_k <= 0:
        errors.append("Top-k must be positive")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

# Export commonly used configurations
def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return config.model

def get_pinecone_config() -> PineconeConfig:
    """Get Pinecone configuration."""
    return config.pinecone

def get_retrieval_config() -> RetrievalConfig:
    """Get retrieval configuration."""
    return config.retrieval

def get_evaluation_config() -> EvaluationConfig:
    """Get evaluation configuration."""
    return config.evaluation

def get_path_config() -> PathConfig:
    """Get path configuration."""
    return config.paths

def get_rag_method_config(method_name: str) -> Dict[str, any]:
    """Get configuration for a specific RAG method."""
    return getattr(config.rag_methods, method_name, {})

def get_common_rag_config() -> Dict[str, any]:
    """Get common configuration parameters used by all RAG methods for fair benchmarking."""
    return config.rag_methods.common_config

def get_benchmark_config() -> Dict[str, any]:
    """Get standardized configuration for benchmarking all RAG methods."""
    return {
        'common': config.rag_methods.common_config,
        'model': {
            'generator_model': config.model.generator_model,
            'helper_model': config.model.helper_model,
            'embedding_model': config.model.embedding_model,
            'temperature': config.model.temperature
        },
        'retrieval': {
            'top_k': config.retrieval.top_k,
            'max_context_length': config.retrieval.max_context_length,
            'similarity_threshold': config.retrieval.similarity_threshold
        },
        'evaluation': {
            'rouge_weights': config.evaluation.rouge_weights,
            'bleu_weights': config.evaluation.bleu_weights,
            'bert_model': config.evaluation.bert_model
        }
    }

def get_index_name(method_name: str) -> str:
    """Get index name for a specific RAG method."""
    return config.index_names.get(method_name, f"{method_name}-rag-docstring")

def get_index_namespace(method_name: str) -> str:
    """Get namespace for a specific RAG method."""
    return config.index_namespaces.get(method_name, f"{method_name}_rag")

def get_knowledge_base_urls() -> List[str]:
    """Get knowledge base URLs."""
    return config.knowledge_base_urls
