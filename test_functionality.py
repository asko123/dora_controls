#!/usr/bin/env python3
"""
Test script to check basic functionality of DORA Controls Analyzer
"""

def test_dora_analyzer_creation():
    """Test if DORAComplianceAnalyzer can be instantiated"""
    try:
        from dora import DORAComplianceAnalyzer
        analyzer = DORAComplianceAnalyzer("CELEX_32022R2554_EN_TXT.pdf")
        print("✓ DORAComplianceAnalyzer can be instantiated")
        return True
    except Exception as e:
        print(f"✗ Error creating DORAComplianceAnalyzer: {e}")
        return False

def test_dora_domains():
    """Test if DORA domains can be loaded"""
    try:
        from dora_domains import DORA_DOMAINS, create_dora_workbook
        print(f"✓ DORA domains loaded ({len(DORA_DOMAINS)} domains)")
        workbook = create_dora_workbook()
        print("✓ DORA workbook can be created")
        return True
    except Exception as e:
        print(f"✗ Error with DORA domains: {e}")
        return False

def test_file_structure():
    """Test if required files exist"""
    import os
    required_files = [
        "CELEX_32022R2554_EN_TXT.pdf",
        "dora.py",
        "dora_domains.py", 
        "dora_workbook_integration.py",
        "setup_and_run.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
        return True

if __name__ == "__main__":
    print("Testing DORA Controls Analyzer functionality...")
    print("=" * 50)
    
    results = []
    results.append(test_file_structure())
    results.append(test_dora_domains())
    results.append(test_dora_analyzer_creation())
    
    print("=" * 50)
    if all(results):
        print("✓ All functionality tests passed!")
    else:
        print("✗ Some functionality tests failed!")
