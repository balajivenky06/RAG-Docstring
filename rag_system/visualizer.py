"""
Visualization module for RAG evaluation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams.update({'font.size': 14}) # Increase default font size

class RAGVisualizer:
    """Creates comprehensive visualizations for RAG evaluation results."""
    
    def __init__(self):
        self.evaluation_data = {}
        self.cost_data = {}
    
    def load_evaluation_data(self, evaluation_files: Dict[str, str]) -> None:
        """Load evaluation results from pickle files."""
        for method_name, file_path in evaluation_files.items():
            try:
                df = pd.read_pickle(file_path)
                self.evaluation_data[method_name] = df
                print(f"Loaded evaluation data for {method_name}: {len(df)} samples")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def load_cost_data(self, cost_files: Dict[str, str]) -> None:
        """Load cost data from pickle files."""
        for method_name, file_path in cost_files.items():
            try:
                df = pd.read_pickle(file_path)
                self.cost_data[method_name] = df
                print(f"Loaded cost data for {method_name}: {len(df)} samples")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def create_performance_comparison_chart(self, save_path: str = "performance_comparison.png") -> None:
        """Create comprehensive performance comparison chart."""
        if not self.evaluation_data:
            print("No evaluation data loaded")
            return
        
        # Prepare data
        metrics = ['rouge_1_f1', 'bleu_score', 'bert_score', 'parameter_coverage', 
                  'return_coverage', 'exception_coverage', 'faithfulness_score', 'pydocstyle_adherence']
        
        methods = list(self.evaluation_data.keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        fig.suptitle('RAG Methods: Performance Comparison Across Metrics', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row = i // 4
            col = i % 4
            
            # Calculate means for each method
            means = []
            stds = []
            for method in methods:
                if metric in self.evaluation_data[method].columns:
                    metric_data = self.evaluation_data[method][metric].dropna()
                    means.append(metric_data.mean())
                    stds.append(metric_data.std())
                else:
                    means.append(0)
                    stds.append(0)
            
            # Create bar plot with error bars
            bars = axes[row, col].bar(methods, means, yerr=stds, capsize=5, alpha=0.7)
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[row, col].set_ylabel('Score')
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                  f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Performance comparison chart saved to {save_path}")
    
    def create_radar_chart(self, save_path: str = "radar_comparison.png") -> None:
        """Create radar chart comparing all methods."""
        if not self.evaluation_data:
            print("No evaluation data loaded")
            return
        
        # Select key metrics for radar chart
        radar_metrics = ['rouge_1_f1', 'bert_score', 'parameter_coverage', 
                        'return_coverage', 'faithfulness_score', 'pydocstyle_adherence']
        
        methods = list(self.evaluation_data.keys())
        
        # Calculate means for each method
        data_for_radar = {}
        for method in methods:
            values = []
            for metric in radar_metrics:
                if metric in self.evaluation_data[method].columns:
                    mean_val = self.evaluation_data[method][metric].dropna().mean()
                    values.append(mean_val)
                else:
                    values.append(0)
            data_for_radar[method] = values
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (method, values) in enumerate(data_for_radar.items()):
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in radar_metrics])
        ax.set_ylim(0, 1)
        ax.set_title('RAG Methods: Comprehensive Performance Radar Chart', 
                    size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Radar chart saved to {save_path}")
    
    def create_box_plots(self, save_path: str = "box_plots.png") -> None:
        """Create box plots showing distribution of scores."""
        if not self.evaluation_data:
            print("No evaluation data loaded")
            return
        
        # Select key metrics
        key_metrics = ['rouge_1_f1', 'bert_score', 'parameter_coverage', 'faithfulness_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RAG Methods: Score Distributions (Box Plots)', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(key_metrics):
            row = i // 2
            col = i % 2
            
            # Prepare data for box plot
            plot_data = []
            labels = []
            
            for method in self.evaluation_data.keys():
                if metric in self.evaluation_data[method].columns:
                    metric_data = self.evaluation_data[method][metric].dropna()
                    plot_data.append(metric_data)
                    labels.append(method)
            
            if plot_data:
                axes[row, col].boxplot(plot_data, labels=labels)
                axes[row, col].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                axes[row, col].set_ylabel('Score')
                axes[row, col].tick_params(axis='x', rotation=45)
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Box plots saved to {save_path}")
    
    def create_heatmap(self, save_path: str = "performance_heatmap.png") -> None:
        """Create heatmap showing performance across methods and metrics."""
        if not self.evaluation_data:
            print("No evaluation data loaded")
            return
        
        # Prepare data for heatmap
        metrics = ['rouge_1_f1', 'bleu_score', 'bert_score', 'parameter_coverage', 
                  'return_coverage', 'exception_coverage', 'faithfulness_score', 'pydocstyle_adherence']
        methods = list(self.evaluation_data.keys())
        
        heatmap_data = []
        for method in methods:
            row = []
            for metric in metrics:
                if metric in self.evaluation_data[method].columns:
                    mean_val = self.evaluation_data[method][metric].dropna().mean()
                    row.append(mean_val)
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=methods,
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Performance Score'})
        
        plt.title('RAG Methods: Performance Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Evaluation Metrics', fontweight='bold')
        plt.ylabel('RAG Methods', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Performance heatmap saved to {save_path}")
    
    def create_cost_performance_scatter(self, save_path: str = "cost_performance_scatter.png") -> None:
        """Create scatter plot showing cost vs performance trade-offs."""
        if not self.evaluation_data or not self.cost_data:
            print("Both evaluation and cost data required")
            return
        
        methods = list(self.evaluation_data.keys())
        
        # Calculate performance scores (average of key metrics)
        performance_scores = []
        cost_scores = []
        
        for method in methods:
            if method in self.evaluation_data and method in self.cost_data:
                # Calculate average performance
                key_metrics = ['rouge_1_f1', 'bert_score', 'parameter_coverage', 'faithfulness_score']
                perf_values = []
                for metric in key_metrics:
                    if metric in self.evaluation_data[method].columns:
                        mean_val = self.evaluation_data[method][metric].dropna().mean()
                        perf_values.append(mean_val)
                
                if perf_values:
                    avg_performance = np.mean(perf_values)
                    performance_scores.append(avg_performance)
                    
                    # Calculate cost score (lower is better)
                    avg_cost = self.cost_data[method]['execution_time'].mean()
                    cost_scores.append(avg_cost)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, method in enumerate(methods):
            if i < len(performance_scores):
                plt.scatter(cost_scores[i], performance_scores[i], 
                           s=200, alpha=0.7, color=colors[i % len(colors)], 
                           label=method, edgecolors='black', linewidth=1)
        
        plt.xlabel('Average Execution Time (seconds)', fontweight='bold')
        plt.ylabel('Average Performance Score', fontweight='bold')
        plt.title('RAG Methods: Cost vs Performance Trade-offs', fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add quadrant lines
        plt.axhline(y=np.mean(performance_scores), color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=np.mean(cost_scores), color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Cost vs performance scatter plot saved to {save_path}")
    
    def create_summary_table(self, save_path: str = "summary_table.png") -> None:
        """Create summary table visualization."""
        if not self.evaluation_data:
            print("No evaluation data loaded")
            return
        
        # Prepare summary data
        summary_data = []
        methods = list(self.evaluation_data.keys())
        
        for method in methods:
            row = {'Method': method}
            
            # Calculate key metrics
            key_metrics = ['rouge_1_f1', 'bert_score', 'parameter_coverage', 'faithfulness_score']
            for metric in key_metrics:
                if metric in self.evaluation_data[method].columns:
                    mean_val = self.evaluation_data[method][metric].dropna().mean()
                    row[metric.replace('_', ' ').title()] = f"{mean_val:.3f}"
                else:
                    row[metric.replace('_', ' ').title()] = "N/A"
            
            summary_data.append(row)
        
        # Create DataFrame and display
        summary_df = pd.DataFrame(summary_data)
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_df.values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('RAG Methods: Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Summary table saved to {save_path}")
    
    def generate_all_visualizations(self, output_dir: str = "visualizations") -> None:
        """Generate all visualizations and save to output directory."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating all visualizations...")
        
        # Generate all charts
        self.create_performance_comparison_chart(f"{output_dir}/performance_comparison.png")
        self.create_radar_chart(f"{output_dir}/radar_comparison.png")
        self.create_box_plots(f"{output_dir}/box_plots.png")
        self.create_heatmap(f"{output_dir}/performance_heatmap.png")
        
        if self.cost_data:
            self.create_cost_performance_scatter(f"{output_dir}/cost_performance_scatter.png")
        
        self.create_summary_table(f"{output_dir}/summary_table.png")
        
        print(f"All visualizations saved to {output_dir}/")
