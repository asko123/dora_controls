# DORA Compliance Analyzer

A comprehensive tool for analyzing compliance with the Digital Operational Resilience Act (DORA - EU Regulation 2022/2554) by identifying gaps between your organization's policies and DORA requirements.

## Overview

The DORA Compliance Analyzer processes the official DORA legislation (CELEX_32022R2554_EN_TXT.pdf) and compares it against your organization's policy documents to:

1. Extract Regulatory Technical Standards (RTS) and Implementing Technical Standards (ITS) requirements from DORA
2. Analyze your organization's policy documents for compliance coverage
3. Identify gaps in your compliance posture
4. Generate detailed gap analysis reports with recommendations

## What are RTS and ITS?

**Regulatory Technical Standards (RTS)** and **Implementing Technical Standards (ITS)** are detailed technical requirements specified within the DORA legislation:

- **RTS**: Define the technical standards and specifications that regulated entities must adhere to. These are more detailed than the main legislation and specify "what" needs to be implemented.

- **ITS**: Describe specific procedures, forms, templates and methods for implementing the requirements. These focus on "how" the implementation should be done.

Both types of standards are crucial for compliance with DORA and are developed by European Supervisory Authorities (ESAs) to ensure consistent implementation across EU member states.

## Features

- **Automated Extraction**: Extracts Regulatory Technical Standards (RTS) and Implementing Technical Standards (ITS) requirements from the DORA legislation
- **Semantic Analysis**: Uses advanced NLP (spaCy and transformers) to analyze policy coverage
- **Multi-Policy Support**: Processes multiple policy documents in parallel
- **Comprehensive Reporting**: Generates detailed compliance gap analysis reports
- **Performance Optimized**: Implements memory management and caching for efficient processing
- **Flexible Configuration**: Customizable thresholds and processing parameters

## Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)
- Required Python packages (see installation section)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/dora-compliance-analyzer.git
   cd dora-compliance-analyzer
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_lg
   ```

## Required Packages

```
pdfplumber>=0.7.4
pandas>=1.3.5
spacy>=3.4.0
torch>=1.13.0
transformers>=4.22.0
sentence-transformers>=2.2.2
tqdm>=4.64.0
filelock>=3.8.0
psutil>=5.9.0
```

## Directory Structure

Before running the analyzer, ensure your files are organized as follows:

```
├── dora.py                      # Main analyzer code
├── CELEX_32022R2554_EN_TXT.pdf  # DORA legislation PDF
└── policies/                    # Your organization's policy documents
    ├── policy1.pdf
    ├── policy2.pdf
    └── ...
```

## Running the Analyzer

1. Place the DORA legislation PDF (CELEX_32022R2554_EN_TXT.pdf) in the root directory
2. Place your organization's policy documents (PDFs) in the `policies` folder
3. Run the analyzer using one of the following methods:

### Method 1: Direct Python Execution

```bash
# From the root directory
python -m WorkShop.dora
```

### Method 2: Import as Module

If you want to run the analyzer from your own script:

```python
from WorkShop.dora import DORAComplianceAnalyzer

# Initialize the analyzer
analyzer = DORAComplianceAnalyzer("CELEX_32022R2554_EN_TXT.pdf")

# Extract requirements
analyzer.extract_technical_standards()

# Analyze a policy document
analyzer.analyze_policy_document("Policy Name", policy_text)

# Generate report
analyzer.generate_gap_analysis_report()
```

### Method 3: Running the Main Function

```bash
# Navigate to the WorkShop directory
cd WorkShop

# Run the dora.py script
python dora.py
```

## Output Files

The analyzer generates the following outputs:

1. **Log File**: `dora_analyzer.log` - Contains detailed logs of the analysis process

2. **Gap Analysis Report**: Located in the `analysis_output` directory with filename format `dora_gap_analysis_YYYYMMDD_HHMMSS.txt`

3. **Cache Files**: Located in the `.cache` directory (for improved performance on subsequent runs)

## Interpreting Results

The gap analysis report includes:

1. **Executive Summary**: Overall compliance coverage percentage
2. **Policy Coverage Summary**: Per-policy coverage of Regulatory Technical Standards (RTS) and Implementing Technical Standards (ITS) requirements
3. **Gap Analysis**: Specific requirements not covered by your policies
4. **Recommendations**: Suggested actions to improve compliance

A high coverage score (>80%) indicates good alignment with DORA requirements, while lower scores highlight areas needing attention.

## Customizing Analysis

You can adjust the analysis behavior by modifying the `DORAConfig` class in the code. Key settings include:

- **Similarity Thresholds**:
  - `STRONG_MATCH_THRESHOLD = 0.6` - Threshold for strong semantic matches
  - `COVERAGE_MATCH_THRESHOLD = 0.7` - Threshold for section matches
  - `FINAL_COVERAGE_THRESHOLD = 0.8` - Threshold for considering a requirement covered

- **Model Configuration**:
  - Uses `microsoft/deberta-v3-base-mnli` for policy area classification (optimized for regulatory text)

## Troubleshooting

- **Memory Errors**: Reduce the `MAX_TEXT_CHUNK_SIZE` in DORAConfig
- **Slow Processing**: Use a GPU or increase `BATCH_SIZE` for parallel processing
- **Low Coverage**: Adjust similarity thresholds in DORAConfig
- **Missing DORA File**: Ensure the DORA legislation PDF is in the root directory with the exact name "CELEX_32022R2554_EN_TXT.pdf"
- **No Policies Found**: Make sure your policy PDFs are in the "policies" folder

## Example Command Sequence

```bash
# Clone repository
git clone https://github.com/yourusername/dora-compliance-analyzer.git
cd dora-compliance-analyzer

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Download DORA legislation if needed
# wget https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022R2554 -O CELEX_32022R2554_EN_TXT.pdf

# Create policies directory and add your PDFs
mkdir -p policies
cp /path/to/your/policies/*.pdf policies/

# Run the analyzer
python -m WorkShop.dora

# View the generated report (contains RTS and ITS compliance analysis)
cat analysis_output/dora_gap_analysis_*.txt
```

## License

[Specify your license here]

## Contributing

[Instructions for contributing to the project]
