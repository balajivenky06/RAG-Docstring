"""
Base RAG class providing common functionality for all RAG implementations.
"""

import time
import psutil
import os
import uuid
import pickle
import pandas as pd
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import ollama
import warnings
import logging
warnings.filterwarnings('ignore')

from .config import config, get_model_config, get_pinecone_config, get_retrieval_config, get_path_config, get_common_rag_config
from .prompts import prompt_manager, get_context_query, get_rewrite_prompt, get_final_generation_prompt, get_system_prompt

@dataclass
class CostMetrics:
    """Class to store computational cost metrics."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    api_calls: int
    tokens_used: int
    retrieval_time: float
    generation_time: float
    total_cost_usd: float = 0.0

class BaseRAG(ABC):
    """
    Abstract base class for all RAG implementations.
    Provides common functionality and interface for docstring generation.
    """
    
    def __init__(self, index_name: str, custom_config: Optional[Dict] = None, namespace: Optional[str] = None):
        """
        Initialize the RAG system.
        
        Args:
            index_name: Name of the Pinecone index to use
            custom_config: Optional custom configuration overrides
            namespace: Optional namespace for the index
        """
        self.index_name = index_name
        self.namespace = namespace
        self.custom_config = custom_config or {}
        
        # Get configurations
        self.model_config = get_model_config()
        self.pinecone_config = get_pinecone_config()
        self.retrieval_config = get_retrieval_config()
        self.path_config = get_path_config()
        
        # Get common configuration for fair benchmarking
        self.common_config = get_common_rag_config()
        
        # Initialize services
        self.embedding_model = None
        self.pinecone_client = None
        self.pinecone_index = None
        self.ollama_client = None
        
        # Cost tracking
        self.cost_metrics = []
        self.api_call_count = 0  # Initialize dynamic API call counter
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize services
        self._initialize_services()
        self._initialize_pinecone_index()
        self._load_data_into_pinecone()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the RAG system."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(getattr(logging, config.logging.level))
        
        if not logger.handlers:
            # Console handler
            if config.logging.console_logging:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(getattr(logging, config.logging.level))
                formatter = logging.Formatter(config.logging.format)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            # File handler
            if config.logging.file_logging:
                log_file = os.path.join(config.logging.log_file)
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(getattr(logging, config.logging.level))
                formatter = logging.Formatter(config.logging.format)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_services(self):
        """Initialize embedding model, Pinecone client, and Ollama client."""
        self.logger.info(f"Initializing services for {self.index_name}...")
        
        try:
            # Initialize embedding model with explicit local cache and offline fallback
            cache_base = os.environ.get("SENTENCE_TRANSFORMERS_HOME") or str((Path.cwd() / "models").resolve())
            # Place model under a deterministic subfolder to allow offline reuse
            model_subdir = self.model_config.embedding_model.replace("/", "__")
            cache_folder = str(Path(cache_base) / model_subdir)

            self.logger.info(f"Loading embedding model '{self.model_config.embedding_model}' with cache at '{cache_folder}'")
            try:
                # 1) Try project cache (may work if already populated)
                self.embedding_model = SentenceTransformer(
                    self.model_config.embedding_model,
                    cache_folder=cache_folder
                )
            except Exception as project_cache_err:
                self.logger.warning(f"Project cache load failed: {project_cache_err}")
                # 2) Try default HF cache in offline mode (use any previously cached files)
                try:
                    self.logger.info("Attempting default HF cache in offline mode (HF_HUB_OFFLINE=1)...")
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    self.embedding_model = SentenceTransformer(
                        self.model_config.embedding_model
                    )
                except Exception as default_cache_err:
                    # 3) Final attempt: default HF cache online (in case network is actually available)
                    try:
                        self.logger.info("Attempting default HF cache online (HF_HUB_OFFLINE=0)...")
                        os.environ["HF_HUB_OFFLINE"] = "0"
                        self.embedding_model = SentenceTransformer(
                            self.model_config.embedding_model
                        )
                    except Exception as final_err:
                        # As a last resort, continue without semantic embeddings
                        self.logger.warning(
                            "Failed to load embedding model; continuing without semantic search. "
                            "RAG will skip Pinecone semantic retrieval."
                        )
                        self.embedding_model = None
                        # Mark semantic retrieval disabled
                        self.semantic_enabled = False
                        # Skip Pinecone entirely in this mode
                        self.pinecone_client = None
                        self.pinecone_index = None

            if self.embedding_model is not None:
                self.logger.info("Embedding model loaded successfully")
                self.semantic_enabled = True
            else:
                self.semantic_enabled = False
            
            # Initialize Pinecone client
            if self.semantic_enabled:
                if not self.pinecone_config.api_key:
                    raise ValueError("Pinecone API key is required")
                self.pinecone_client = Pinecone(api_key=self.pinecone_config.api_key)
                self.logger.info("Pinecone client initialized successfully")
            else:
                self.pinecone_client = None
            
            # Initialize Ollama client
            self.ollama_client = ollama.Client()
            self.logger.info(f"Ollama client initialized. Generator: {self.model_config.generator_model}, Helper: {self.model_config.helper_model}")
            
        except Exception as e:
            self.logger.error(f"Error initializing services: {e}")
            raise
    
    def _initialize_pinecone_index(self):
        """Initialize or connect to Pinecone index."""
        try:
            existing_indexes = [index_info["name"] for index_info in self.pinecone_client.list_indexes()]
            
            if self.index_name not in existing_indexes:
                self.logger.info(f"Index '{self.index_name}' not found. Creating new index...")
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.pinecone_config.vector_dimension,
                    metric=self.pinecone_config.metric,
                    spec=ServerlessSpec(
                        cloud=self.pinecone_config.cloud,
                        region=self.pinecone_config.region
                    )
                )
                
                # Wait for index to be ready
                while not self.pinecone_client.describe_index(self.index_name).status["ready"]:
                    self.logger.info(f"Waiting for index '{self.index_name}' to become ready...")
                    time.sleep(5)
                
                self.logger.info(f"Index '{self.index_name}' created and ready")
            else:
                self.logger.info(f"Connecting to existing index '{self.index_name}'")
            
            self.pinecone_index = self.pinecone_client.Index(self.index_name)
            stats = self.pinecone_index.describe_index_stats()
            self.logger.info(f"Successfully connected to index '{self.index_name}' with namespace '{self.namespace}'. Stats: {stats}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone index '{self.index_name}': {e}")
            raise
    
    def _load_data_into_pinecone(self):
        """Load knowledge base data into Pinecone index."""
        index_stats = self.pinecone_index.describe_index_stats()
        
        if index_stats.total_vector_count == 0:
            self.logger.info(f"Index '{self.index_name}' is empty. Loading data from knowledge base...")
            total_docs_loaded = 0
            
            for url in config.knowledge_base_urls:
                try:
                    response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    main_content = soup.find('main') or soup.find('article') or soup.find('body')
                    
                    if main_content:
                        page_text = main_content.get_text(separator='\n', strip=True)
                    else:
                        page_text = soup.get_text(separator='\n', strip=True)
                    
                    if not page_text or len(page_text) < 50:
                        self.logger.warning(f"Insufficient text content from {url}. Skipping.")
                        continue
                    
                    # Generate embedding
                    embedding = self.embedding_model.encode(page_text).tolist()
                    doc_id = str(uuid.uuid4())
                    metadata = {"text": page_text, "source": url}
                    
                    # Upsert to Pinecone
                    self.pinecone_index.upsert(vectors=[(doc_id, embedding, metadata)], namespace=self.namespace)
                    total_docs_loaded += 1
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Error fetching URL {url}: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing data for {url}: {e}")
            
            if total_docs_loaded > 0:
                self.logger.info("Waiting for indexing to complete...")
                time.sleep(2)
                stats = self.pinecone_index.describe_index_stats()
                self.logger.info(f"Indexing complete. Stats: {stats}")
            else:
                self.logger.warning("No documents were loaded into the index")
        else:
            self.logger.info(f"Index '{self.index_name}' already contains {index_stats.total_vector_count} vectors")
    
    def _generate_context_query(self, user_code: str) -> str:
        """Generate a detailed query for context retrieval."""
        return get_context_query(user_code)
    
    def _revise_prompt_with_helper_model(self, context: str, user_code: str) -> str:
        """Use helper LLM to refine the prompt for the generator model."""
        try:
            rewrite_prompt = get_rewrite_prompt(context)
            
            response = self.ollama_client.generate(
                model=self.model_config.helper_model,
                prompt=rewrite_prompt,
                options={'temperature': self.model_config.temperature * 0.6}  # Lower temperature for rewriting
            )
            self.api_call_count += 1  # Increment API call counter
            
            return response.get('response', '').strip()
            
        except Exception as e:
            self.logger.error(f"Error during prompt rewriting: {e}")
            if "timeout" in str(e).lower():
                self.logger.warning("⏰ Timeout during prompt rewriting, using original prompt")
            return rewrite_prompt  # Fallback to original prompt
    
    def _generate_final_docstring(self, context: str, user_code: str, rewritten_req: str) -> str:
        """Generate the final docstring using the main generator LLM."""
        messages = [
            {'role': 'system', 'content': get_system_prompt('docstring_generator')}
        ]
        
        if context:
            context_prompt = prompt_manager.get_context_inclusion_prompt(context, user_code)
            messages.append({'role': 'user', 'content': context_prompt})
        else:
            messages.append({'role': 'user', 'content': "No specific context was retrieved for this request."})
        
        messages.append({'role': 'user', 'content': get_final_generation_prompt(user_code)})
        
        try:
            response = self.ollama_client.chat(
                model=self.model_config.generator_model,
                messages=messages,
                options={'temperature': self.model_config.temperature}
            )
            self.api_call_count += 1  # Increment API call counter
            
            generated_docstring = response.get('message', {}).get('content', '').strip()
            
            # Basic cleaning
            generated_docstring = self._clean_docstring_output(generated_docstring)
            
            return generated_docstring
            
        except Exception as e:
            self.logger.error(f"Error communicating with Ollama during final docstring generation: {e}", exc_info=True)
            if "timeout" in str(e).lower():
                self.logger.warning("⏰ Timeout during docstring generation, using fallback")
            return "# ERROR: Docstring generation failed."
    
    def _clean_docstring_output(self, docstring_text: str) -> str:
        """Clean and format the generated docstring."""
        if not isinstance(docstring_text, str):
            return docstring_text
        
        if docstring_text.startswith("# ERROR:") or docstring_text.startswith("# SKIPPED:"):
            return docstring_text
        
        text = docstring_text.strip()
        
        # Remove code block markers
        if text.startswith("```python"):
            text = text[len("```python"):].strip()
        elif text.startswith("```"):
            text = text[len("```"):].strip()
        
        if text.endswith("```"):
            text = text[:-len("```")].strip()
        
        # Extract content inside quotes
        content_inside_quotes = None
        first_double_quotes = text.find('"""')
        if first_double_quotes != -1:
            last_double_quotes = text.rfind('"""')
            if last_double_quotes > first_double_quotes and (last_double_quotes + 3) <= len(text):
                content_inside_quotes = text[first_double_quotes + 3: last_double_quotes].strip()
        
        if content_inside_quotes is None or not content_inside_quotes.strip():
            first_single_quotes = text.find("'''")
            if first_single_quotes != -1:
                last_single_quotes = text.rfind("'''")
                if last_single_quotes > first_single_quotes and (last_single_quotes + 3) <= len(text):
                    content_inside_quotes = text[first_single_quotes + 3: last_single_quotes].strip()
        
        if content_inside_quotes is not None and content_inside_quotes.strip():
            final_text_to_clean = content_inside_quotes
        else:
            final_text_to_clean = text
            if final_text_to_clean.startswith('"""') and final_text_to_clean.endswith('"""') and len(final_text_to_clean) >= 6:
                final_text_to_clean = final_text_to_clean[3:-3].strip()
            elif final_text_to_clean.startswith("'''") and final_text_to_clean.endswith("'''") and len(final_text_to_clean) >= 6:
                final_text_to_clean = final_text_to_clean[3:-3].strip()
        
        # Remove class definition if present
        final_text_to_clean = re.sub(r"(?i)^class\s+\w+:\s*\n?", "", final_text_to_clean).strip()
        
        # Remove function definitions and code implementations
        final_text_to_clean = re.sub(r"(?i)^def\s+\w+.*?:\s*\n?", "", final_text_to_clean, flags=re.MULTILINE).strip()
        
        # Remove any remaining code patterns (lines that start with spaces and contain code)
        lines = final_text_to_clean.split('\n')
        cleaned_lines = []
        in_code_section = False
        
        for line in lines:
            # Skip lines that look like code implementations
            if (line.strip().startswith('def ') or 
                line.strip().startswith('class ') or
                line.strip().startswith('self.') or
                line.strip().startswith('return ') or
                line.strip().startswith('if ') or
                line.strip().startswith('for ') or
                line.strip().startswith('while ') or
                line.strip().startswith('try:') or
                line.strip().startswith('except') or
                line.strip().startswith('with ') or
                line.strip().startswith('import ') or
                line.strip().startswith('from ') or
                (line.strip() and not line.strip().startswith(('Args:', 'Returns:', 'Raises:', 'Examples:', 'Note:', 'Warning:', 'Todo:')) and 
                 not line.strip().startswith(('    ', '\t')) and 
                 ':' in line and not line.strip().endswith(':'))):
                continue
            cleaned_lines.append(line)
        
        final_text_to_clean = '\n'.join(cleaned_lines).strip()
        
        return final_text_to_clean
    
    def _track_cost_metrics(self, start_time: float, retrieval_time: float, generation_time: float, 
                           api_calls: int = None, tokens_used: int = 0) -> CostMetrics:
        """Track computational cost metrics."""
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Use accumulated API calls if not explicitly provided
        if api_calls is None:
            api_calls = self.api_call_count
        
        # Get memory/CPU usage (robust to env-specific issues)
        try:
            process = psutil.Process()
            memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except Exception:
            memory_usage_mb = 0.0
        
        try:
            cpu_usage_percent = psutil.cpu_percent()  # system-wide percent is more robust
        except Exception:
            cpu_usage_percent = 0.0
        
        cost_metrics = CostMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            api_calls=api_calls,
            tokens_used=tokens_used,
            retrieval_time=retrieval_time,
            generation_time=generation_time
        )
        
        self.cost_metrics.append(cost_metrics)
        return cost_metrics
    
    @abstractmethod
    def generate_docstring(self, user_code: str) -> Tuple[str, CostMetrics]:
        """
        Abstract method to generate docstring for given code.
        
        Args:
            user_code: Python code to generate docstring for
            
        Returns:
            Tuple of (generated_docstring, cost_metrics)
        """
        pass
    
    def process_dataset(self, dataset_path: str, output_dir: str = None) -> str:
        """
        Process a dataset of code samples and generate docstrings.
        
        Args:
            dataset_path: Path to the dataset pickle file
            output_dir: Directory to save results
            
        Returns:
            Path to the results file
        """
        if output_dir is None:
            output_dir = self.path_config.results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        df = pd.read_pickle(dataset_path)
        self.logger.info(f"Loaded dataset with {len(df)} samples")
        
        generated_docstrings = []
        retrieved_contexts = []
        cost_metrics_list = []
        
        # Start timing for progress tracking
        start_time = time.time()
        
        for i, row in df.iterrows():
            sample_start_time = time.time()
            self.logger.info(f"Processing sample {i+1}/{len(df)}")
            
            user_code = row["Code_without_comments"]
            
            # Reset API call counter for each sample
            self.api_call_count = 0
            
            # Generate docstring
            docstring, cost_metrics = self.generate_docstring(user_code)
            
            generated_docstrings.append(self._clean_docstring_output(docstring))
            # Ensure faithfulness receives a string, not a list structure
            latest_ctx = getattr(self, 'retrieved_contexts', [''])[-1] if hasattr(self, 'retrieved_contexts') else ''
            if isinstance(latest_ctx, list):
                try:
                    latest_ctx_text = ' '.join([c.get('text', '') for c in latest_ctx if isinstance(c, dict)])
                except Exception:
                    latest_ctx_text = str(latest_ctx)
            else:
                latest_ctx_text = str(latest_ctx) if latest_ctx is not None else ''
            retrieved_contexts.append(latest_ctx_text)
            cost_metrics_list.append(cost_metrics)
            
            # Progress logging every 5 samples
            if (i + 1) % 5 == 0:
                elapsed_time = time.time() - sample_start_time
                total_elapsed = time.time() - start_time
                avg_time_per_sample = total_elapsed / (i + 1)
                remaining_samples = len(df) - (i + 1)
                estimated_remaining_time = remaining_samples * avg_time_per_sample
                
                self.logger.info(f"📊 PROGRESS UPDATE - Sample {i+1}/{len(df)}")
                self.logger.info(f"   ⏱️  Current sample time: {elapsed_time:.2f}s")
                self.logger.info(f"   📈 Average time per sample: {avg_time_per_sample:.2f}s")
                self.logger.info(f"   ⏳ Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
                self.logger.info(f"   📊 Progress: {((i+1)/len(df)*100):.1f}%")
                self.logger.info(f"   🎯 Generated docstring length: {len(docstring)} chars")
                
                # Log cost metrics for this batch
                if cost_metrics:
                    self.logger.info(f"   💰 Cost metrics - Execution: {cost_metrics.execution_time:.2f}s, "
                                   f"Memory: {cost_metrics.memory_usage_mb:.1f}MB, "
                                   f"API Calls: {cost_metrics.api_calls}")
            
            # Additional logging for every sample (less verbose)
            else:
                elapsed_time = time.time() - sample_start_time
                self.logger.info(f"✅ Sample {i+1} completed in {elapsed_time:.2f}s")
        
        # Create results DataFrame
        results_df = df.copy()
        results_df["Generated_Docstring"] = generated_docstrings
        results_df["Retrieved_Context"] = retrieved_contexts
        results_df["RAG_Method"] = self.__class__.__name__
        
        # Save results
        results_file = os.path.join(output_dir, f"{self.__class__.__name__}_results.pkl")
        results_df.to_pickle(results_file)
        
        # Save results as Excel file
        excel_file = os.path.join(output_dir, f"{self.__class__.__name__}_results.xlsx")
        results_df.to_excel(excel_file, index=False, engine='openpyxl')
        
        # Save cost metrics
        cost_file = os.path.join(output_dir, f"{self.__class__.__name__}_costs.pkl")
        pd.DataFrame(cost_metrics_list).to_pickle(cost_file)
        
        # Save cost metrics as Excel file
        cost_excel_file = os.path.join(output_dir, f"{self.__class__.__name__}_costs.xlsx")
        pd.DataFrame(cost_metrics_list).to_excel(cost_excel_file, index=False, engine='openpyxl')

        # --- CONSOLIDATION: Combine RAG Results + Cost Metrics into one file ---
        # NOTE: This creates a single DataFrame with everything
        try:
            consolidated_excel = os.path.join(output_dir, f"{self.__class__.__name__}_consolidated.xlsx")
            
            # Create a dataframe from cost metrics
            cost_metrics_df = pd.DataFrame(cost_metrics_list)
            # Prefix cost columns to avoid confusion
            cost_metrics_df.columns = [f"Cost_{c}" for c in cost_metrics_df.columns]
            
            # Merge side-by-side (assuming same index order, which is true here)
            consolidated_df = pd.concat([results_df.reset_index(drop=True), cost_metrics_df.reset_index(drop=True)], axis=1)
            
            consolidated_df.to_excel(consolidated_excel, index=False, engine='openpyxl')
            self.logger.info(f"💾 Consolidated results saved to {consolidated_excel}")
        except Exception as e:
            self.logger.error(f"Failed to save consolidated Excel: {e}")
        # -----------------------------------------------------------------------
        
        # Final summary logging
        total_time = time.time() - start_time
        avg_time_per_sample = total_time / len(df)
        
        self.logger.info("🎉 PROCESSING COMPLETED!")
        self.logger.info(f"📊 Total samples processed: {len(df)}")
        self.logger.info(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
        self.logger.info(f"📈 Average time per sample: {avg_time_per_sample:.2f} seconds")
        self.logger.info(f"💾 Results saved to {results_file}")
        self.logger.info(f"💾 Results saved to {excel_file}")
        self.logger.info(f"💾 Cost metrics saved to {cost_file}")
        self.logger.info(f"💾 Cost metrics saved to {cost_excel_file}")

        # --- Visualization (shared across methods) ---
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            # Ensure a writable MPL cache in constrained environments
            os.environ.setdefault('MPLCONFIGDIR', os.path.join(self.path_config.results_dir, '.mplcache'))
            os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

            viz_dir = os.path.join(self.path_config.results_dir, 'visualization', self.__class__.__name__)
            os.makedirs(viz_dir, exist_ok=True)

            # Build cost metrics DataFrame safely
            try:
                cost_df = pd.DataFrame([cm.__dict__ if hasattr(cm, '__dict__') else cm for cm in cost_metrics_list])
            except Exception:
                cost_df = pd.DataFrame()

            # 1) Docstring length distribution
            try:
                lengths = [len(s or '') for s in generated_docstrings]
                plt.figure(figsize=(6,4))
                plt.hist(lengths, bins=min(30, max(5, len(lengths)//2)), color='#4C78A8')
                plt.title(f'{self.__class__.__name__}: Generated Docstring Lengths')
                plt.xlabel('Characters')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'docstring_lengths.png'), dpi=150)
                plt.close()
            except Exception as _:
                pass

            if not cost_df.empty:
                # 2) Execution time per sample
                try:
                    plt.figure(figsize=(7,4))
                    plt.plot(cost_df.get('execution_time', pd.Series([None]*len(cost_df))), marker='o', linestyle='-', color='#F58518')
                    plt.title(f'{self.__class__.__name__}: Execution Time per Sample')
                    plt.xlabel('Sample Index')
                    plt.ylabel('Seconds')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'execution_time_per_sample.png'), dpi=150)
                    plt.close()
                except Exception as _:
                    pass

                # 3) Memory usage per sample
                try:
                    plt.figure(figsize=(7,4))
                    plt.plot(cost_df.get('memory_usage_mb', pd.Series([None]*len(cost_df))), marker='o', linestyle='-', color='#54A24B')
                    plt.title(f'{self.__class__.__name__}: Memory Usage per Sample')
                    plt.xlabel('Sample Index')
                    plt.ylabel('MB')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, 'memory_usage_per_sample.png'), dpi=150)
                    plt.close()
                except Exception as _:
                    pass

                # 4) Retrieval vs Generation time stacked bar (if available)
                try:
                    if 'retrieval_time' in cost_df and 'generation_time' in cost_df:
                        plt.figure(figsize=(7,4))
                        ind = range(len(cost_df))
                        plt.bar(ind, cost_df['retrieval_time'], label='Retrieval', color='#72B7B2')
                        plt.bar(ind, cost_df['generation_time'], bottom=cost_df['retrieval_time'], label='Generation', color='#E45756')
                        plt.title(f'{self.__class__.__name__}: Time Breakdown per Sample')
                        plt.xlabel('Sample Index')
                        plt.ylabel('Seconds')
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(viz_dir, 'time_breakdown_per_sample.png'), dpi=150)
                        plt.close()
                except Exception as _:
                    pass

            self.logger.info(f"🖼️ Visualizations saved to {viz_dir}")
        except Exception as viz_err:
            self.logger.warning(f"Visualization step skipped due to error: {viz_err}")
        
        return results_file