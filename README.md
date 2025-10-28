# RAG-based Docstring Generation System

A comprehensive system for generating Python docstrings using different Retrieval-Augmented Generation (RAG) strategies, with detailed computational cost analysis and evaluation metrics.

## Overview

This system implements and compares 5 different RAG approaches for automated Python docstring generation:

1. **Simple RAG** - Basic RAG without enhancements
2. **Code-Aware RAG** - Analyzes code structure for enriched queries
3. **Corrective RAG** - Evaluates retrieval quality and takes corrective actions
4. **Fusion RAG** - Combines semantic and keyword search using Reciprocal Rank Fusion
5. **Self-RAG** - Critiques its own output and conditionally uses RAG

## Features

- **Modular Architecture**: Each RAG method implemented as a separate class
- **Comprehensive Evaluation**: 10+ metrics including ROUGE, BLEU, BERT scores, parameter coverage, etc.
- **Computational Cost Analysis**: Detailed timing, memory, CPU, and API usage tracking
- **Rich Visualizations**: Performance comparisons, radar charts, heatmaps, and cost analysis
- **Separate Results Storage**: Each RAG method's results stored independently
- **Easy Configuration**: Centralized configuration system

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG-Docstring
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Ollama models:
```bash
ollama pull deepseek-coder:6.7b
ollama pull deepseek-r1:1.5b
```

## Quick Start

### Basic Usage

```python
from rag_system import SimpleRAG, config

# Set configuration (multiple ways available)
config.pinecone.api_key = "your_pinecone_api_key"
config.model.generator_model = "deepseek-coder:6.7b"
config.model.helper_model = "deepseek-r1:1.5b"

# Initialize RAG
rag = SimpleRAG()

# Generate docstring
code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''

docstring, cost_metrics = rag.generate_docstring(code)
print(docstring)
print(f"Generated in {cost_metrics.execution_time:.3f}s")
```

### Running Complete Experiment

```bash
python run_rag_experiment.py \
    --pinecone_api_key "your_api_key" \
    --dataset_path "class_files_df.pkl" \
    --methods simple code_aware corrective fusion self
```

### Running Single Method

```bash
python run_rag_experiment.py \
    --pinecone_api_key "your_api_key" \
    --single_method code_aware \
    --dataset_path "class_files_df.pkl"
```

## System Architecture

```
rag_system/
├── __init__.py              # Package initialization
├── base_rag.py              # Base RAG class with common functionality
├── simple_rag.py            # Simple RAG implementation
├── code_aware_rag.py        # Code-Aware RAG implementation
├── corrective_rag.py        # Corrective RAG implementation
├── fusion_rag.py            # Fusion RAG implementation
├── self_rag.py              # Self-RAG implementation
├── evaluator.py             # Comprehensive evaluation framework
├── cost_analyzer.py         # Computational cost analysis
└── visualizer.py            # Visualization generation
```

## Evaluation Metrics

### Text Quality Metrics
- **ROUGE-1 F1**: Measures overlap with reference docstrings
- **BLEU Score**: N-gram precision evaluation
- **BERT Score**: Semantic similarity using BERT embeddings

### Documentation Quality Metrics
- **Parameter Coverage**: Checks if function parameters are documented
- **Return Coverage**: Verifies return value documentation
- **Exception Coverage**: Checks if raised exceptions are documented
- **Python Style Adherence**: Uses pydocstyle for PEP 257 compliance

### Readability & Efficiency Metrics
- **Flesch Reading Ease**: Measures readability
- **Conciseness**: Compression ratio comparison
- **Faithfulness Score**: Measures alignment with retrieved context

## Computational Cost Analysis

The system tracks comprehensive cost metrics:

- **Execution Time**: Total time for docstring generation
- **Memory Usage**: RAM consumption in MB
- **CPU Usage**: CPU utilization percentage
- **API Calls**: Number of LLM API calls
- **Retrieval Time**: Time spent on context retrieval
- **Generation Time**: Time spent on docstring generation
- **Efficiency Score**: Composite efficiency metric

## Visualizations

The system generates multiple visualization types:

1. **Performance Comparison Charts**: Bar charts comparing all metrics
2. **Radar Charts**: Comprehensive performance overview
3. **Box Plots**: Score distribution analysis
4. **Heatmaps**: Performance across methods and metrics
5. **Cost Analysis Charts**: Computational cost comparisons
6. **Scatter Plots**: Performance vs efficiency trade-offs

## Configuration

The system uses a comprehensive configuration system with multiple components:

### Configuration Components

- **ModelConfig**: LLM model settings (embedding, generator, helper models)
- **PineconeConfig**: Vector database configuration
- **RetrievalConfig**: Retrieval parameters (top_k, context length, etc.)
- **EvaluationConfig**: Evaluation metrics settings
- **CostAnalysisConfig**: Cost tracking configuration
- **PathConfig**: File and directory paths
- **LoggingConfig**: Logging settings
- **WebSearchConfig**: Web search functionality settings
- **RAGMethodConfig**: Method-specific configurations

### Configuration Methods

#### 1. Environment Variables
```bash
export PINECONE_API_KEY="your_api_key"
export GENERATOR_MODEL="deepseek-coder:6.7b"
export HELPER_MODEL="deepseek-r1:1.5b"
```

#### 2. JSON Configuration File
```python
from rag_system.config_loader import load_config_from_file

# Load from file
config = load_config_from_file("my_config.json")
```

#### 3. Command Line Arguments
```bash
python run_rag_experiment.py \
    --pinecone_api_key "your_key" \
    --generator_model "deepseek-coder:6.7b" \
    --temperature 0.7 \
    --top_k 5
```

#### 4. Programmatic Configuration
```python
from rag_system import config

# Modify configuration
config.model.temperature = 0.7
config.retrieval.top_k = 5
config.pinecone.api_key = "your_key"
```

## Results Structure

```
results/
├── SimpleRAG_results.pkl      # Simple RAG results
├── SimpleRAG_costs.pkl       # Simple RAG cost metrics
├── CodeAwareRAG_results.pkl   # Code-Aware RAG results
├── CodeAwareRAG_costs.pkl    # Code-Aware RAG cost metrics
└── ...

evaluation/
├── simple_evaluation.pkl      # Simple RAG evaluation metrics
├── code_aware_evaluation.pkl # Code-Aware RAG evaluation metrics
├── cost_analysis_report.txt  # Detailed cost analysis
└── cost_summary.csv          # Cost summary table

visualizations/
├── performance_comparison.png # Performance comparison chart
├── radar_comparison.png      # Radar chart
├── box_plots.png            # Box plots
├── performance_heatmap.png   # Performance heatmap
├── cost_comparison.png       # Cost comparison chart
└── summary_table.png        # Summary table
```

## API Reference

### BaseRAG Class

```python
class BaseRAG(ABC):
    def __init__(self, config: RAGConfig, index_name: str)
    def generate_docstring(self, code: str) -> Tuple[str, CostMetrics]
    def process_dataset(self, dataset_path: str, output_dir: str = "results")
```

### RAGEvaluator Class

```python
class RAGEvaluator:
    def evaluate_single_sample(self, code: str, ground_truth: str, 
                             generated_docstring: str, retrieved_context: str = "") -> Dict[str, float]
    def evaluate_dataset(self, results_df: pd.DataFrame) -> pd.DataFrame
```

### CostAnalyzer Class

```python
class CostAnalyzer:
    def load_cost_data(self, cost_files: Dict[str, str])
    def generate_summaries(self) -> Dict[str, CostSummary]
    def create_cost_comparison_chart(self, save_path: str)
    def generate_cost_report(self, output_file: str)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={RAG-based Docstring Generation: A Comprehensive Evaluation},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**: Ensure Ollama is running and models are pulled
2. **Pinecone Connection Error**: Verify API key and index names
3. **Memory Issues**: Reduce batch size or use smaller models
4. **Import Errors**: Install all required dependencies

### Performance Tips

1. Use GPU acceleration for embedding models
2. Optimize Pinecone index configuration
3. Implement caching for repeated queries
4. Use smaller models for faster inference

## Future Work

- [ ] Multi-language support
- [ ] Real-time evaluation
- [ ] Integration with IDEs
- [ ] Advanced RAG architectures
- [ ] User studies and human evaluation
