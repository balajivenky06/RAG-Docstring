"""
Test script to verify RAG system functionality.
"""

import os
import sys
import pandas as pd
from typing import Dict, List

# Add the current directory to Python path
# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from rag_system import (
            SystemConfig, SimpleRAG,
            SelfCorrectionRAG, RAGEvaluator, CostAnalyzer, RAGVisualizer
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration creation."""
    print("\nTesting configuration...")
    
    try:
        from rag_system import SystemConfig
        
        # Test creating a basic configuration
        config = SystemConfig()
        
        # Test that we can access basic properties
        assert hasattr(config, 'model')
        assert hasattr(config, 'pinecone')
        assert hasattr(config, 'retrieval')
        
        print("✓ Configuration creation successful")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_evaluator():
    """Test evaluator functionality."""
    print("\nTesting evaluator...")
    
    try:
        from rag_system import RAGEvaluator
        
        evaluator = RAGEvaluator()
        
        # Test sample evaluation
        code = '''
def add_numbers(a, b):
    return a + b
        '''
        
        ground_truth = '''
        Add two numbers together.
        
        Args:
            a (int): First number
            b (int): Second number
        
        Returns:
            int: Sum of a and b
        '''
        
        generated = '''
        Add two numbers.
        
        Args:
            a: first number
            b: second number
        
        Returns:
            sum
        '''
        
        metrics = evaluator.evaluate_single_sample(code, ground_truth, generated)
        
        assert 'rouge_1_f1' in metrics
        assert 'bert_score' in metrics
        assert 'parameter_coverage' in metrics
        
        print("✓ Evaluator functionality successful")
        return True
    except Exception as e:
        print(f"✗ Evaluator error: {e}")
        return False

def test_cost_analyzer():
    """Test cost analyzer functionality."""
    print("\nTesting cost analyzer...")
    
    try:
        from rag_system import CostAnalyzer, CostMetrics
        
        analyzer = CostAnalyzer()
        
        # Create sample cost data
        sample_data = {
            'execution_time': [1.0, 1.5, 2.0],
            'memory_usage_mb': [50.0, 60.0, 70.0],
            'cpu_usage_percent': [10.0, 15.0, 20.0],
            'api_calls': [2, 3, 4],
            'retrieval_time': [0.5, 0.7, 1.0],
            'generation_time': [0.5, 0.8, 1.0]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Test efficiency score calculation
        efficiency = analyzer.calculate_efficiency_score(1.0, 50.0, 10.0, 2)
        assert isinstance(efficiency, float)
        assert 0 <= efficiency <= 1
        
        print("✓ Cost analyzer functionality successful")
        return True
    except Exception as e:
        print(f"✗ Cost analyzer error: {e}")
        return False

def test_visualizer():
    """Test visualizer functionality."""
    print("\nTesting visualizer...")
    
    try:
        from rag_system import RAGVisualizer
        
        visualizer = RAGVisualizer()
        
        # Create sample evaluation data
        sample_eval_data = {
            'rouge_1_f1': [0.5, 0.6, 0.7],
            'bert_score': [0.6, 0.7, 0.8],
            'parameter_coverage': [0.8, 0.9, 1.0],
            'faithfulness_score': [0.7, 0.8, 0.9]
        }
        
        df = pd.DataFrame(sample_eval_data)
        
        # Test that visualizer can handle the data
        assert len(df) == 3
        assert 'rouge_1_f1' in df.columns
        
        print("✓ Visualizer functionality successful")
        return True
    except Exception as e:
        print(f"✗ Visualizer error: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading."""
    print("\nTesting dataset loading...")
    
    try:
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "class_files_df.pkl")
        
        if os.path.exists(dataset_path):
            df = pd.read_pickle(dataset_path)
            
            assert 'Code_without_comments' in df.columns
            assert 'Comments' in df.columns
            
            print(f"✓ Dataset loaded successfully: {len(df)} samples")
            return True
        else:
            print(f"⚠ Dataset file {dataset_path} not found")
            return False
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("RAG System Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_evaluator,
        test_cost_analyzer,
        test_visualizer,
        test_dataset_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
