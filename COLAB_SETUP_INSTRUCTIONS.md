# 🚀 Running DORA Controls Analyzer in Google Colab

## Complete Setup Guide for [https://github.com/asko123/dora_controls.git](https://github.com/asko123/dora_controls.git)

### 📋 Prerequisites
- Google account for Colab access
- Your organization's policy PDF files
- 15-30 minutes for complete setup and analysis

---

## 🔧 Step-by-Step Setup

### Step 1: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Click **"New notebook"**

### Step 2: Enable GPU (Highly Recommended)
```python
# Check current runtime
import platform
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
```

**To enable GPU:**
1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **"GPU"**
3. Click **Save**
4. The notebook will restart automatically

### Step 3: Clone Repository and Install Dependencies

**Cell 1: Clone the repository**
```python
!git clone https://github.com/asko123/dora_controls.git
%cd dora_controls

# List repository contents
!ls -la
```

**Cell 2: Install required packages**
```python
# Install dependencies
!pip install pdfplumber pandas spacy torch transformers sentence-transformers tqdm xlsxwriter openpyxl filelock psutil

# Download spaCy language model
!python -m spacy download en_core_web_lg

print("✅ All dependencies installed successfully!")
```

**Cell 3: Verify installation**
```python
import torch
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
    print("✅ spaCy model loaded successfully")
except:
    print("❌ spaCy model failed to load")

print("✅ Setup verification complete!")
```

### Step 4: Upload Your Policy Documents

**Cell 4: Upload policy files**
```python
from google.colab import files
import os
import shutil

# Check if policies folder exists (it should from the repo)
if not os.path.exists('policies'):
    os.makedirs('policies')

print("📁 Please upload your organization's policy PDF files")
print("These will be analyzed against DORA requirements")
print("\nSupported formats: PDF only")
print("Recommended: Upload 1-10 policy documents for best performance")

# Upload files
uploaded = files.upload()

# Move uploaded files to policies folder
for filename in uploaded.keys():
    if filename.endswith('.pdf'):
        # Move to policies folder
        shutil.move(filename, f'policies/{filename}')
        print(f"✅ Moved {filename} to policies folder")
    else:
        print(f"⚠️  Skipped {filename} (not a PDF)")

# Show what's in policies folder
policy_files = os.listdir('policies')
print(f"\n📊 Total policy files ready for analysis: {len(policy_files)}")
for file in policy_files:
    print(f"  - {file}")
```

### Step 5: Verify Repository Structure

**Cell 5: Check all required files**
```python
import os

# Check for essential files
essential_files = {
    'CELEX_32022R2554_EN_TXT.pdf': 'DORA Legislation',
    'setup_and_run.py': 'Main Setup Script',
    'dora.py': 'Core Analyzer (Alternative)',
    'dora_domains.py': 'Domain Definitions',
    'dora_workbook_integration.py': 'Workbook Integration',
    'requirements-gpu.txt': 'GPU Requirements',
    'requirements-cpu.txt': 'CPU Requirements'
}

print("🔍 Checking repository structure:")
all_good = True

for file, description in essential_files.items():
    if os.path.exists(file):
        print(f"✅ {file} - {description}")
    else:
        print(f"❌ {file} - {description} (MISSING)")
        all_good = False

# Check policies folder
policies_count = len(os.listdir('policies')) if os.path.exists('policies') else 0
print(f"📁 Policies folder: {policies_count} files")

if all_good and policies_count > 0:
    print("\n🎉 Ready to run analysis!")
else:
    print("\n⚠️  Some files may be missing. Analysis may fail.")
```

### Step 6: Run the Analysis

**Cell 6: Execute DORA analysis**
```python
# Run the automated setup and analysis
print("🚀 Starting DORA Compliance Analysis...")
print("This may take 10-30 minutes depending on:")
print("- Number and size of policy documents")
print("- GPU availability")
print("- Model download requirements")
print()

# Execute the main setup script
!python setup_and_run.py

print("\n✅ Analysis execution completed!")
```

### Step 7: Alternative Analysis Methods

**Cell 7a: Run main analysis only (if setup script fails)**
```python
# Alternative: Run main analysis directly
print("🔄 Running main DORA analysis (alternative method)...")
!python dora.py
```

**Cell 7b: Run domain analysis only**
```python
# Alternative: Run domain-specific analysis
print("📊 Running domain-specific analysis...")
!python -m dora_workbook_integration
```

### Step 8: Check Results and Download Reports

**Cell 8: List and download generated reports**
```python
import os
from google.colab import files

print("📊 Looking for generated reports...")

# Check for different types of reports
reports_found = []

# Check analysis_output folder
if os.path.exists('analysis_output'):
    analysis_files = os.listdir('analysis_output')
    for file in analysis_files:
        if file.endswith('.txt'):
            reports_found.append(f"analysis_output/{file}")
            print(f"📄 Text Report: {file}")

# Check for Excel reports in root directory
for file in os.listdir('.'):
    if file.startswith('dora_domain_compliance') and file.endswith('.xlsx'):
        reports_found.append(file)
        print(f"📈 Excel Report: {file}")

# Check for log files
if os.path.exists('setup_and_run.log'):
    reports_found.append('setup_and_run.log')
    print(f"📝 Setup Log: setup_and_run.log")

if os.path.exists('dora_analyzer.log'):
    reports_found.append('dora_analyzer.log')
    print(f"📝 Analysis Log: dora_analyzer.log")

print(f"\n📦 Total reports found: {len(reports_found)}")

# Download all reports
if reports_found:
    print("\n⬇️  Downloading reports to your computer...")
    for report in reports_found:
        try:
            files.download(report)
            print(f"✅ Downloaded: {report}")
        except Exception as e:
            print(f"❌ Failed to download {report}: {e}")
else:
    print("⚠️  No reports found. Check the execution logs above for errors.")
```

### Step 9: View Report Summary (Optional)

**Cell 9: Display report summary in Colab**
```python
# Display a summary of results directly in Colab
import os

# Try to read and display text report summary
text_reports = []
if os.path.exists('analysis_output'):
    text_reports = [f for f in os.listdir('analysis_output') if f.endswith('.txt')]

if text_reports:
    report_path = f"analysis_output/{text_reports[0]}"
    print(f"📊 Summary from: {text_reports[0]}")
    print("=" * 60)
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract and display key sections
        lines = content.split('\n')
        in_summary = False
        summary_lines = []
        
        for line in lines[:100]:  # First 100 lines only
            if 'Executive Summary' in line or 'Overall compliance coverage' in line:
                in_summary = True
            if in_summary:
                summary_lines.append(line)
                if len(summary_lines) > 20:  # Limit output
                    break
        
        for line in summary_lines:
            print(line)
            
        print("\n📄 Full report downloaded to your computer")
        
    except Exception as e:
        print(f"Error reading report: {e}")
else:
    print("📊 No text reports found to preview")

# Try to display Excel report info
excel_reports = [f for f in os.listdir('.') if f.startswith('dora_domain_compliance') and f.endswith('.xlsx')]

if excel_reports:
    print(f"\n📈 Excel Report Available: {excel_reports[0]}")
    print("Contains:")
    print("  - Domain-specific compliance analysis")
    print("  - Interactive charts and pivot tables")
    print("  - Coverage statistics by domain")
    print("  - Policy mapping results")
```

---

## 🛠️ Troubleshooting Guide

### Common Issues and Solutions

**1. Out of Memory Error**
```python
# If you get memory errors, restart runtime and try with fewer files
# Runtime → Restart and run all (upload fewer policy files)
```

**2. GPU Not Available**
```python
# Check if GPU is enabled
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# If False, go to Runtime → Change runtime type → GPU
```

**3. Package Installation Failures**
```python
# Reinstall specific packages if needed
!pip install --upgrade torch transformers
!pip install --force-reinstall spacy
!python -m spacy download en_core_web_lg --force
```

**4. Repository Clone Issues**
```python
# If git clone fails, try again or use ZIP download
!rm -rf dora_controls  # Remove if partially cloned
!git clone https://github.com/asko123/dora_controls.git
```

**5. No Reports Generated**
```python
# Check for error messages in the logs
!cat setup_and_run.log | tail -50

# Or try running components individually
!python dora.py
```

---

## 📚 Understanding the Output

### Generated Reports:

1. **Text Report** (`analysis_output/dora_gap_analysis_*.txt`)
   - Comprehensive gap analysis
   - Article-by-article breakdown
   - Specific recommendations
   - Coverage percentages

2. **Excel Report** (`dora_domain_compliance_*.xlsx`)
   - Domain-structured analysis
   - Visual charts and graphs
   - Pivot tables for data analysis
   - SOC2-style organization

3. **Log Files** (`*.log`)
   - Detailed execution logs
   - Error messages and debugging info
   - Performance statistics

### Key Metrics to Look For:
- **Overall Compliance Coverage**: Percentage of DORA requirements covered
- **RTS Coverage**: Regulatory Technical Standards compliance
- **ITS Coverage**: Implementing Technical Standards compliance
- **Domain-Specific Scores**: Coverage by compliance domain

---

## 🔄 Re-running Analysis

To run analysis on different policy documents:

```python
# Clear previous policies
!rm -rf policies/*

# Upload new files (run upload cell again)
# Then re-run the analysis
!python setup_and_run.py
```

---

## 💡 Performance Tips

1. **Use GPU**: Always enable GPU for 5-10x faster processing
2. **Batch Size**: Start with 2-5 policy files for testing
3. **File Size**: Optimize large PDFs (>50MB) before upload
4. **Memory**: Restart runtime if you encounter memory issues
5. **Timeout**: Large analyses may take 20-30 minutes

---

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the log files for error messages
3. Visit the [GitHub repository](https://github.com/asko123/dora_controls.git) for updates
4. Ensure all policy files are valid PDFs with extractable text

---

**🎉 You're all set! Follow these steps to get comprehensive DORA compliance analysis in Google Colab.** 
