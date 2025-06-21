#!/usr/bin/env python3
"""
Test script to check if all modules can be imported successfully
"""

def test_core_dependencies():
    """Test if core dependencies are available"""
    try:
        import pdfplumber
        import pandas
        import spacy
        import torch
        print("✓ Core dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Core dependency missing: {e}")
        return False

def test_dora_module():
    """Test if dora.py can be imported"""
    try:
        import dora
        print("✓ dora.py imports successfully")
        return True
    except Exception as e:
        print(f"✗ Import error in dora.py: {e}")
        return False

def test_dora_domains_module():
    """Test if dora_domains.py can be imported"""
    try:
        import dora_domains
        print("✓ dora_domains.py imports successfully")
        return True
    except Exception as e:
        print(f"✗ Import error in dora_domains.py: {e}")
        return False

def test_dora_workbook_module():
    """Test if dora_workbook_integration.py can be imported"""
    try:
        import dora_workbook_integration
        print("✓ dora_workbook_integration.py imports successfully")
        return True
    except Exception as e:
        print(f"✗ Import error in dora_workbook_integration.py: {e}")
        return False

if __name__ == "__main__":
    print("Testing DORA Controls Analyzer imports...")
    print("=" * 50)
    
    results = []
    results.append(test_core_dependencies())
    results.append(test_dora_module())
    results.append(test_dora_domains_module())
    results.append(test_dora_workbook_module())
    
    print("=" * 50)
    if all(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
