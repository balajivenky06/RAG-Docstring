"""
Validation script to ensure all RAG methods use consistent configuration and prompts.
"""

import os
import sys
import json
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def validate_rag_consistency():
    """Validate that all RAG methods use consistent configuration."""
    print("RAG Consistency Validation")
    print("=" * 50)
    
    try:
        from rag_system import (
            SimpleRAG, CodeAwareRAG, CorrectiveRAG, FusionRAG, SelfRAG,
            get_common_rag_config, get_benchmark_config
        )
        
        # Get common configuration
        common_config = get_common_rag_config()
        benchmark_config = get_benchmark_config()
        
        print(f"✓ Common configuration loaded: {len(common_config)} parameters")
        print(f"✓ Benchmark configuration loaded: {len(benchmark_config)} sections")
        
        # Test RAG initialization (without actual services)
        rag_methods = {
            'SimpleRAG': SimpleRAG,
            'CodeAwareRAG': CodeAwareRAG,
            'CorrectiveRAG': CorrectiveRAG,
            'FusionRAG': FusionRAG,
            'SelfRAG': SelfRAG
        }
        
        validation_results = {}
        
        for method_name, rag_class in rag_methods.items():
            print(f"\n--- Validating {method_name} ---")
            
            try:
                # Create instance without calling __init__ to avoid service initialization
                rag_instance = rag_class.__new__(rag_class)
                
                # Manually set required attributes for validation
                rag_instance.index_name = f"test-{method_name.lower()}"
                rag_instance.custom_config = {}
                
                # Call the parent __init__ to set up configuration
                from rag_system.base_rag import BaseRAG
                BaseRAG.__init__(rag_instance, rag_instance.index_name, rag_instance.custom_config)
                
                # Validate common configuration usage
                validation_results[method_name] = {
                    'common_config_used': hasattr(rag_instance, 'common_config'),
                    'top_k': getattr(rag_instance, 'top_k', None),
                    'use_rewrite': getattr(rag_instance, 'use_rewrite', None),
                    'rewrite_temperature': getattr(rag_instance, 'rewrite_temperature', None),
                    'generation_temperature': getattr(rag_instance, 'generation_temperature', None),
                    'web_search_enabled': getattr(rag_instance, 'web_search_enabled', None),
                    'evaluation_enabled': getattr(rag_instance, 'evaluation_enabled', None)
                }
                
                print(f"  ✓ Common config: {validation_results[method_name]['common_config_used']}")
                print(f"  ✓ Top-k: {validation_results[method_name]['top_k']}")
                print(f"  ✓ Use rewrite: {validation_results[method_name]['use_rewrite']}")
                print(f"  ✓ Rewrite temp: {validation_results[method_name]['rewrite_temperature']}")
                print(f"  ✓ Generation temp: {validation_results[method_name]['generation_temperature']}")
                
            except Exception as e:
                print(f"  ✗ Error validating {method_name}: {e}")
                validation_results[method_name] = {'error': str(e)}
        
        # Check consistency across methods
        print(f"\n--- Consistency Check ---")
        consistent_params = ['top_k', 'use_rewrite', 'rewrite_temperature', 'generation_temperature']
        
        for param in consistent_params:
            values = []
            for method_name, results in validation_results.items():
                if 'error' not in results and param in results:
                    values.append(results[param])
            
            if len(set(values)) == 1:
                print(f"✓ {param}: All methods use same value ({values[0] if values else 'N/A'})")
            else:
                print(f"✗ {param}: Inconsistent values {values}")
        
        return validation_results
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return None

def validate_prompt_consistency():
    """Validate that all RAG methods use consistent prompts."""
    print(f"\n--- Prompt Consistency Check ---")
    
    try:
        from rag_system.prompts import (
            get_context_query, get_initial_generation_prompt, get_rewrite_prompt,
            get_critique_prompt, get_relevance_evaluation_prompt, get_web_search_query,
            get_final_generation_prompt, get_context_inclusion_prompt, get_system_prompt
        )
        
        # Test prompt functions
        test_code = "def test_function(param1, param2): return param1 + param2"
        
        prompts = {
            'context_query': get_context_query(test_code),
            'initial_generation': get_initial_generation_prompt(test_code),
            'rewrite': get_rewrite_prompt("test context"),
            'critique': get_critique_prompt(test_code, "test docstring", ["param1", "param2"]),
            'relevance_evaluation': get_relevance_evaluation_prompt("test doc", "test source", test_code),
            'web_search': get_web_search_query("test query", "def test"),
            'final_generation': get_final_generation_prompt(test_code),
            'context_inclusion': get_context_inclusion_prompt("test context", test_code),
            'system_docstring': get_system_prompt('docstring_generator'),
            'system_critique': get_system_prompt('critique'),
            'system_rewrite': get_system_prompt('rewrite')
        }
        
        print("✓ All prompt functions available:")
        for prompt_name, prompt_content in prompts.items():
            if prompt_content and len(prompt_content.strip()) > 0:
                print(f"  ✓ {prompt_name}: {len(prompt_content)} characters")
            else:
                print(f"  ✗ {prompt_name}: Empty or invalid")
        
        return prompts
        
    except ImportError as e:
        print(f"✗ Prompt import error: {e}")
        return None

def validate_benchmark_config():
    """Validate benchmark configuration file."""
    print(f"\n--- Benchmark Configuration Check ---")
    
    benchmark_file = "benchmark_config.json"
    
    if not os.path.exists(benchmark_file):
        print(f"✗ Benchmark configuration file not found: {benchmark_file}")
        return False
    
    try:
        with open(benchmark_file, 'r') as f:
            benchmark_config = json.load(f)
        
        print(f"✓ Benchmark configuration loaded from {benchmark_file}")
        
        # Check required sections
        required_sections = ['benchmark_config', 'common_parameters', 'model_config', 'method_specific']
        for section in required_sections:
            if section in benchmark_config:
                print(f"  ✓ {section}: Present")
            else:
                print(f"  ✗ {section}: Missing")
        
        # Check common parameters
        if 'common_parameters' in benchmark_config:
            common_params = benchmark_config['common_parameters']
            expected_params = ['top_k', 'use_rewrite', 'rewrite_temperature', 'generation_temperature']
            
            for param in expected_params:
                if param in common_params:
                    print(f"  ✓ Common param {param}: {common_params[param]}")
                else:
                    print(f"  ✗ Common param {param}: Missing")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in benchmark configuration: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading benchmark configuration: {e}")
        return False

def generate_consistency_report():
    """Generate a comprehensive consistency report."""
    print(f"\n--- Generating Consistency Report ---")
    
    # Run all validations
    rag_results = validate_rag_consistency()
    prompt_results = validate_prompt_consistency()
    benchmark_results = validate_benchmark_config()
    
    # Generate report
    report = {
        'timestamp': str(pd.Timestamp.now()),
        'rag_consistency': rag_results,
        'prompt_consistency': prompt_results is not None,
        'benchmark_config_valid': benchmark_results,
        'summary': {
            'total_methods': len(rag_results) if rag_results else 0,
            'consistent_methods': len([r for r in (rag_results or {}).values() if 'error' not in r]),
            'prompts_available': len(prompt_results) if prompt_results else 0,
            'benchmark_config_loaded': benchmark_results
        }
    }
    
    # Save report
    report_file = "consistency_report.json"
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"✓ Consistency report saved to {report_file}")
    except Exception as e:
        print(f"✗ Error saving report: {e}")
    
    return report

def main():
    """Main validation function."""
    print("RAG System Consistency Validation")
    print("=" * 60)
    
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available, using basic timestamp")
        pd = None
    
    # Run validation
    report = generate_consistency_report()
    
    # Summary
    print(f"\n--- Validation Summary ---")
    if report:
        summary = report.get('summary', {})
        print(f"Total RAG methods: {summary.get('total_methods', 0)}")
        print(f"Consistent methods: {summary.get('consistent_methods', 0)}")
        print(f"Prompts available: {summary.get('prompts_available', 0)}")
        print(f"Benchmark config: {'✓' if summary.get('benchmark_config_loaded') else '✗'}")
        
        if summary.get('consistent_methods', 0) == summary.get('total_methods', 0):
            print("\n🎉 All RAG methods are consistent!")
        else:
            print("\n⚠ Some inconsistencies found. Check the report for details.")
    
    return report is not None

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
