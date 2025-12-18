"""
Cost analysis module for RAG implementations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

@dataclass
class CostSummary:
    """Summary of cost metrics for a RAG method."""
    method_name: str
    avg_execution_time: float
    avg_memory_usage: float
    avg_cpu_usage: float
    total_api_calls: int
    avg_retrieval_time: float
    avg_generation_time: float
    efficiency_score: float

class CostAnalyzer:
    """Analyzes computational costs of different RAG methods."""
    
    def __init__(self):
        self.cost_data = {}
        self.summaries = {}
    
    def load_cost_data(self, cost_files: Dict[str, str]) -> None:
        """Load cost data from pickle files."""
        for method_name, file_path in cost_files.items():
            try:
                df = pd.read_pickle(file_path)
                self.cost_data[method_name] = df
                print(f"Loaded cost data for {method_name}: {len(df)} samples")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def calculate_efficiency_score(self, execution_time: float, memory_usage: float, 
                                 cpu_usage: float, api_calls: int) -> float:
        """Calculate efficiency score (higher is better)."""
        # Normalize metrics (lower values are better)
        time_score = 1.0 / (1.0 + execution_time)
        memory_score = 1.0 / (1.0 + memory_usage / 100)  # Normalize to ~100MB
        cpu_score = 1.0 / (1.0 + cpu_usage / 100)  # Normalize to ~100%
        api_score = 1.0 / (1.0 + api_calls / 10)  # Normalize to ~10 calls
        
        # Weighted average
        return (0.4 * time_score + 0.2 * memory_score + 0.2 * cpu_score + 0.2 * api_score)
    
    def generate_summaries(self) -> Dict[str, CostSummary]:
        """Generate cost summaries for all methods."""
        for method_name, df in self.cost_data.items():
            summary = CostSummary(
                method_name=method_name,
                avg_execution_time=df['execution_time'].mean(),
                avg_memory_usage=df['memory_usage_mb'].mean(),
                avg_cpu_usage=df['cpu_usage_percent'].mean(),
                total_api_calls=df['api_calls'].sum(),
                avg_retrieval_time=df['retrieval_time'].mean(),
                avg_generation_time=df['generation_time'].mean(),
                efficiency_score=self.calculate_efficiency_score(
                    df['execution_time'].mean(),
                    df['memory_usage_mb'].mean(),
                    df['cpu_usage_percent'].mean(),
                    df['api_calls'].mean()
                )
            )
            self.summaries[method_name] = summary
        
        return self.summaries
    
    def create_cost_comparison_chart(self, save_path: str = "cost_comparison.png") -> None:
        """Create comprehensive cost comparison chart."""
        if not self.summaries:
            self.generate_summaries()
        
        # Prepare data for plotting
        methods = list(self.summaries.keys())
        execution_times = [s.avg_execution_time for s in self.summaries.values()]
        memory_usage = [s.avg_memory_usage for s in self.summaries.values()]
        cpu_usage = [s.avg_cpu_usage for s in self.summaries.values()]
        efficiency_scores = [s.efficiency_score for s in self.summaries.values()]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RAG Methods: Computational Cost Analysis', fontsize=16, fontweight='bold')
        
        # Execution Time
        axes[0, 0].bar(methods, execution_times, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Average Execution Time (seconds)', fontweight='bold')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory Usage
        axes[0, 1].bar(methods, memory_usage, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Average Memory Usage (MB)', fontweight='bold')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # CPU Usage
        axes[1, 0].bar(methods, cpu_usage, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Average CPU Usage (%)', fontweight='bold')
        axes[1, 0].set_ylabel('CPU (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Efficiency Score
        axes[1, 1].bar(methods, efficiency_scores, color='gold', alpha=0.7)
        axes[1, 1].set_title('Efficiency Score (Higher = Better)', fontweight='bold')
        axes[1, 1].set_ylabel('Efficiency Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Cost comparison chart saved to {save_path}")
    
    def create_performance_efficiency_scatter(self, evaluation_results: Dict[str, pd.DataFrame], 
                                            save_path: str = "performance_efficiency.png") -> None:
        """Create scatter plot showing performance vs efficiency trade-offs."""
        if not self.summaries:
            self.generate_summaries()
        
        methods = list(self.summaries.keys())
        efficiency_scores = [self.summaries[m].efficiency_score for m in methods]
        
        # Calculate average performance scores (using BERT score as proxy)
        performance_scores = []
        for method in methods:
            if method in evaluation_results:
                avg_bert_score = evaluation_results[method]['bert_score'].mean()
                performance_scores.append(avg_bert_score)
            else:
                performance_scores.append(0.0)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, method in enumerate(methods):
            plt.scatter(efficiency_scores[i], performance_scores[i], 
                       s=200, alpha=0.7, color=colors[i % len(colors)], 
                       label=method, edgecolors='black', linewidth=1)
        
        plt.xlabel('Efficiency Score (Higher = Better)', fontweight='bold')
        plt.ylabel('Performance Score (BERT Score)', fontweight='bold')
        plt.title('RAG Methods: Performance vs Efficiency Trade-offs', fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add quadrant labels
        plt.axhline(y=np.mean(performance_scores), color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=np.mean(efficiency_scores), color='gray', linestyle='--', alpha=0.5)
        
        plt.text(0.05, 0.95, 'High Performance\nLow Efficiency', transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.text(0.05, 0.05, 'Low Performance\nLow Efficiency', transform=plt.gca().transAxes,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        plt.text(0.95, 0.95, 'High Performance\nHigh Efficiency', transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        plt.text(0.95, 0.05, 'Low Performance\nHigh Efficiency', transform=plt.gca().transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Performance vs efficiency chart saved to {save_path}")
    
    def generate_cost_report(self, output_file: str = "cost_analysis_report.txt") -> None:
        """Generate detailed cost analysis report."""
        if not self.summaries:
            self.generate_summaries()
        
        with open(output_file, 'w') as f:
            f.write("RAG Methods: Computational Cost Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall ranking by efficiency
            sorted_methods = sorted(self.summaries.items(), 
                                 key=lambda x: x[1].efficiency_score, reverse=True)
            
            f.write("Efficiency Ranking (Higher Score = Better):\n")
            f.write("-" * 40 + "\n")
            for i, (method, summary) in enumerate(sorted_methods, 1):
                f.write(f"{i}. {method}: {summary.efficiency_score:.4f}\n")
            
            f.write("\nDetailed Metrics:\n")
            f.write("-" * 40 + "\n")
            
            for method, summary in self.summaries.items():
                f.write(f"\n{method}:\n")
                f.write(f"  Average Execution Time: {summary.avg_execution_time:.3f} seconds\n")
                f.write(f"  Average Memory Usage: {summary.avg_memory_usage:.2f} MB\n")
                f.write(f"  Average CPU Usage: {summary.avg_cpu_usage:.2f}%\n")
                f.write(f"  Total API Calls: {summary.total_api_calls}\n")
                f.write(f"  Average Retrieval Time: {summary.avg_retrieval_time:.3f} seconds\n")
                f.write(f"  Average Generation Time: {summary.avg_generation_time:.3f} seconds\n")
                f.write(f"  Efficiency Score: {summary.efficiency_score:.4f}\n")
            
            # Recommendations
            f.write("\nRecommendations:\n")
            f.write("-" * 40 + "\n")
            
            best_efficiency = sorted_methods[0][0]
            f.write(f"• Most Efficient: {best_efficiency} (Best for resource-constrained environments)\n")
            
            fastest_method = min(self.summaries.items(), key=lambda x: x[1].avg_execution_time)[0]
            f.write(f"• Fastest: {fastest_method} (Best for real-time applications)\n")
            
            lowest_memory = min(self.summaries.items(), key=lambda x: x[1].avg_memory_usage)[0]
            f.write(f"• Lowest Memory: {lowest_memory} (Best for memory-constrained systems)\n")
            
            lowest_api_calls = min(self.summaries.items(), key=lambda x: x[1].total_api_calls)[0]
            f.write(f"• Fewest API Calls: {lowest_api_calls} (Best for cost-sensitive applications)\n")
        
        print(f"Cost analysis report saved to {output_file}")
    
    def get_cost_summary_table(self) -> pd.DataFrame:
        """Get cost summary as a pandas DataFrame."""
        if not self.summaries:
            self.generate_summaries()
        
        data = []
        for method, summary in self.summaries.items():
            data.append({
                'Method': method,
                'Avg_Execution_Time': summary.avg_execution_time,
                'Avg_Memory_MB': summary.avg_memory_usage,
                'Avg_CPU_Percent': summary.avg_cpu_usage,
                'Total_API_Calls': summary.total_api_calls,
                'Avg_Retrieval_Time': summary.avg_retrieval_time,
                'Avg_Generation_Time': summary.avg_generation_time,
                'Efficiency_Score': summary.efficiency_score
            })
        
        return pd.DataFrame(data)
