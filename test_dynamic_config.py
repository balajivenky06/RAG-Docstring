"""
Test script to verify the dynamic configuration system.
"""

import os
import sys
import json
import tempfile

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_imports():
    """Test that configuration modules can be imported."""
    print("Testing configuration imports...")
    
    try:
        from rag_system import config, SystemConfig, ModelConfig, PineconeConfig
        from rag_system.config_loader import load_config_from_file, save_config_to_file
        print("✓ Configuration imports successful")
        return True
    except ImportError as e:
        print(f"✗ Configuration import error: {e}")
        return False

def test_prompt_imports():
    """Test that prompt modules can be imported."""
    print("\nTesting prompt imports...")
    
    try:
        from rag_system import prompt_manager, PromptTemplates, PromptManager
        from rag_system.prompts import get_context_query, get_system_prompt
        print("✓ Prompt imports successful")
        return True
    except ImportError as e:
        print(f"✗ Prompt import error: {e}")
        return False

def test_config_creation():
    """Test configuration object creation."""
    print("\nTesting configuration creation...")
    
    try:
        from rag_system import config, ModelConfig, PineconeConfig
        
        # Test accessing configuration
        assert hasattr(config, 'model')
        assert hasattr(config, 'pinecone')
        assert hasattr(config.model, 'generator_model')
        assert hasattr(config.pinecone, 'api_key')
        
        print("✓ Configuration creation successful")
        return True
    except Exception as e:
        print(f"✗ Configuration creation error: {e}")
        return False

def test_config_modification():
    """Test configuration modification."""
    print("\nTesting configuration modification...")
    
    try:
        from rag_system import config
        
        # Test modifying configuration
        original_temp = config.model.temperature
        config.model.temperature = 0.8
        assert config.model.temperature == 0.8
        
        # Restore original value
        config.model.temperature = original_temp
        
        print("✓ Configuration modification successful")
        return True
    except Exception as e:
        print(f"✗ Configuration modification error: {e}")
        return False

def test_prompt_functions():
    """Test prompt function calls."""
    print("\nTesting prompt functions...")
    
    try:
        from rag_system.prompts import get_context_query, get_system_prompt
        
        # Test context query generation
        code = "def test(): pass"
        query = get_context_query(code)
        assert isinstance(query, str)
        assert len(query) > 0
        
        # Test system prompt retrieval
        system_prompt = get_system_prompt('docstring_generator')
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        
        print("✓ Prompt functions successful")
        return True
    except Exception as e:
        print(f"✗ Prompt functions error: {e}")
        return False

def test_config_file_operations():
    """Test configuration file operations."""
    print("\nTesting configuration file operations...")
    
    try:
        from rag_system.config_loader import save_config_to_file, load_config_from_file
        from rag_system import config
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Test saving configuration
            save_config_to_file(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Test loading configuration
            loaded_config = load_config_from_file(tmp_path)
            assert loaded_config is not None
            
            print("✓ Configuration file operations successful")
            return True
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        print(f"✗ Configuration file operations error: {e}")
        return False

def test_rag_initialization():
    """Test RAG initialization with dynamic configuration."""
    print("\nTesting RAG initialization...")
    
    try:
        from rag_system import SimpleRAG, config
        
        # Set test configuration
        config.pinecone.api_key = "test_key"
        config.model.generator_model = "test-model"
        config.model.helper_model = "test-helper"
        
        # Test RAG initialization (without actually connecting to services)
        # This will test the configuration loading
        rag = SimpleRAG.__new__(SimpleRAG)  # Create without calling __init__
        
        # Test that configuration is accessible
        assert hasattr(rag, 'model_config')
        assert hasattr(rag, 'pinecone_config')
        
        print("✓ RAG initialization successful")
        return True
    except Exception as e:
        print(f"✗ RAG initialization error: {e}")
        return False

def test_template_loading():
    """Test configuration template loading."""
    print("\nTesting configuration template...")
    
    try:
        template_path = "config_template.json"
        
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                template = json.load(f)
            
            # Check required sections
            required_sections = ['model', 'pinecone', 'retrieval', 'evaluation']
            for section in required_sections:
                assert section in template, f"Missing section: {section}"
            
            print("✓ Configuration template loading successful")
            return True
        else:
            print("⚠ Configuration template not found")
            return False
    except Exception as e:
        print(f"✗ Configuration template error: {e}")
        return False

def run_all_tests():
    """Run all configuration tests."""
    print("Dynamic Configuration System Test Suite")
    print("=" * 50)
    
    tests = [
        test_config_imports,
        test_prompt_imports,
        test_config_creation,
        test_config_modification,
        test_prompt_functions,
        test_config_file_operations,
        test_rag_initialization,
        test_template_loading
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
        print("🎉 All configuration tests passed! Dynamic system is ready.")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
