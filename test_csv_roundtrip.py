#!/usr/bin/env python3
"""
Test script to verify CSV export/import compatibility
"""

import os
import tempfile
from dora_domains import DORAWorkbook, create_dora_workbook, load_domains_from_csv

def test_csv_export_import_compatibility():
    """Test that export_to_csv and load_domains_from_csv are compatible"""
    print("Testing CSV export/import compatibility...")
    
    try:
        original_workbook = DORAWorkbook("nonexistent.csv")  # Force fallback to hardcoded
        original_domains = original_workbook.domains
        print(f"✓ Original workbook created with {len(original_domains)} domains")
        
        temp_csv = "test_export_temp.csv"
        exported_file = original_workbook.export_to_csv(temp_csv)
        print(f"✓ Domains exported to {exported_file}")
        
        imported_domains = load_domains_from_csv(temp_csv)
        print(f"✓ Domains loaded from CSV: {len(imported_domains)} domains")
        
        if len(original_domains) != len(imported_domains):
            print(f"✗ Domain count mismatch: {len(original_domains)} vs {len(imported_domains)}")
            return False
        
        for i, (orig, imported) in enumerate(zip(original_domains, imported_domains)):
            for key in ['code', 'article', 'domain', 'requirement']:
                if orig[key] != imported[key]:
                    print(f"✗ Domain {i} mismatch in '{key}': '{orig[key]}' vs '{imported[key]}'")
                    return False
        
        print("✓ All domains match perfectly after export/import")
        
        new_workbook = create_dora_workbook(temp_csv)
        new_domains = new_workbook.domains
        print(f"✓ New workbook created from exported CSV with {len(new_domains)} domains")
        
        if len(original_domains) != len(new_domains):
            print(f"✗ New workbook domain count mismatch: {len(original_domains)} vs {len(new_domains)}")
            return False
        
        for i, (orig, new) in enumerate(zip(original_domains, new_domains)):
            for key in ['code', 'article', 'domain', 'requirement']:
                if orig[key] != new[key]:
                    print(f"✗ New workbook domain {i} mismatch in '{key}': '{orig[key]}' vs '{new[key]}'")
                    return False
        
        print("✓ New workbook domains match original perfectly")
        
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            print("✓ Temporary CSV file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during export/import test: {e}")
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        return False

def test_csv_format_consistency():
    """Test that the CSV format is consistent with expectations"""
    print("\nTesting CSV format consistency...")
    
    try:
        workbook = DORAWorkbook("nonexistent.csv")  # Force hardcoded
        temp_csv = "test_format_temp.csv"
        workbook.export_to_csv(temp_csv)
        
        with open(temp_csv, 'r') as f:
            lines = f.readlines()
        
        header = lines[0].strip()
        expected_header = "code,article,domain,requirement"
        if header != expected_header:
            print(f"✗ Header mismatch: '{header}' vs '{expected_header}'")
            return False
        
        print("✓ CSV header format is correct")
        
        if len(lines) < 2:
            print("✗ CSV file has no data rows")
            return False
        
        print(f"✓ CSV file has {len(lines)-1} data rows")
        
        for i, line in enumerate(lines[1:6]):  # Check first 5 data rows
            parts = line.strip().split(',')
            if len(parts) < 4:
                print(f"✗ Row {i+1} has insufficient columns: {len(parts)}")
                return False
        
        print("✓ Sample rows have correct number of columns")
        
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        
        return True
        
    except Exception as e:
        print(f"✗ Error during format test: {e}")
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        return False

if __name__ == "__main__":
    print("Testing CSV Export/Import Compatibility")
    print("=" * 50)
    
    results = []
    results.append(test_csv_export_import_compatibility())
    results.append(test_csv_format_consistency())
    
    print("=" * 50)
    if all(results):
        print("✓ All CSV compatibility tests passed!")
    else:
        print("✗ Some CSV compatibility tests failed!")
