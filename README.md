# DORA Controls Analyzer

A comprehensive tool for analyzing compliance with the Digital Operational Resilience Act (DORA - EU Regulation 2022/2554) by identifying gaps between your organization's policies and DORA requirements.

## How to Run the DORA Controls Analyzer

Choose the deployment method that best fits your needs:

### 🚀 Option 1: Kubernetes Deployment (Recommended for Production)

**Best for:** Production environments, scalable processing, cloud deployments

```bash
# Prerequisites: Kubernetes cluster with kubectl configured
git clone https://github.com/asko123/dora_controls.git
cd dora_controls

# Quick deployment
chmod +x k8s/deploy.sh
./k8s/deploy.sh

# Add your policy files
kubectl cp policies/ dora-analyzer/$(kubectl get pods -n dora-analyzer -l app=dora-analyzer -o jsonpath='{.items[0].metadata.name}'):/app/policies/

# Run analysis (choose CPU or GPU)
kubectl apply -f k8s/job-cpu.yaml     # For CPU processing
kubectl apply -f k8s/job-gpu.yaml     # For GPU processing (faster)

# Monitor progress and get results
kubectl logs -f job/dora-analyzer-cpu -n dora-analyzer
kubectl cp dora-analyzer/$(kubectl get pods -n dora-analyzer -l app=dora-analyzer -o jsonpath='{.items[0].metadata.name}'):/app/analysis_output/ ./results/
```

### 🖥️ Option 2: Local Automated Setup (Easiest for Testing)

**Best for:** Quick testing, development, single-use analysis

```bash
git clone https://github.com/asko123/dora_controls.git
cd dora_controls

# Add your policy documents
mkdir -p policies
cp /path/to/your/policies/*.pdf policies/

# Run automated setup (handles everything automatically)
python setup_and_run.py  # or ./setup_and_run.sh on Linux/macOS
```

### ⚙️ Option 3: Manual Local Setup (Advanced Users)

**Best for:** Custom configurations, development, troubleshooting

```bash
git clone https://github.com/asko123/dora_controls.git
cd dora_controls

# Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies (GPU recommended for 5-10x faster processing)
pip install -r requirements-gpu.txt  # For GPU support
# OR
pip install -r requirements-cpu.txt  # For CPU-only

# Download required models
python -m spacy download en_core_web_lg

# Add your policies and run
mkdir -p policies
cp /path/to/your/policies/*.pdf policies/
python dora.py
```

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

## Advanced Analysis Capabilities

The tool includes sophisticated analysis features to provide more accurate and actionable results:

1. **DORA Relevance Screening**: Automatically identifies which policies are relevant to DORA compliance based on content analysis, filtering out irrelevant documents to focus analysis on meaningful targets.

2. **Policy Focus Detection**: Intelligently determines each policy's intended focus areas (e.g., authentication, risk management, incident response) based on document title and content analysis.

3. **Context-Aware Gap Analysis**: Distinguishes between:
   - **Relevant Gaps**: Missing requirements that fall within a policy's intended scope
   - **Informational Gaps**: Requirements that are outside a policy's scope and should be covered elsewhere

4. **Article-Specific Analysis**: Provides detailed information about each gap including:
   - The specific DORA article and requirement text
   - The policy that should cover it
   - Similarity scores showing how close existing policies come to meeting the requirement
   - Detailed reasons why the requirement is considered not covered

5. **Tailored Recommendations**: Generates policy-specific recommendations based on:
   - Each policy's focus areas
   - The nature and severity of identified gaps
   - Whether new policies are needed or existing ones should be enhanced

These advanced capabilities ensure you receive a realistic, context-sensitive gap analysis that respects the specialized nature of different policy documents.

## Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)
- Required Python packages (see installation section)

## Automated Setup (Recommended)

**🚀 One-Click Setup and Analysis**

For the easiest experience, use our automated setup script that handles everything:

### Windows Users
```cmd
setup_and_run.bat
```

### macOS/Linux Users
```bash
./setup_and_run.sh
```

**What the automated setup does:**
1. ✅ Detects if you have a CUDA-compatible GPU
2. ✅ Automatically installs the correct requirements (GPU or CPU)
3. ✅ Downloads required spaCy language models
4. ✅ Verifies the installation
5. ✅ Runs the main DORA analysis (dora.py)
6. ✅ Runs the workbook analysis (domain-specific Excel reports)
7. ✅ Provides real-time progress updates and error handling

**Requirements for automated setup:**
- Python 3.7+ installed and accessible via command line
- Internet connection for downloading models and dependencies
- Your policy PDFs in the `policies` folder
- DORA legislation PDF in the root directory

**Example output:**
```
╔══════════════════════════════════════════════════════════════╗
║                    DORA Controls Analyzer                    ║
║                  Automated Setup & Runner                   ║
╚══════════════════════════════════════════════════════════════╝

✓ Python version: 3.8.10
✓ Project structure verified
✓ DORA legislation file found
✓ Found 5 policy files for analysis
✓ NVIDIA GPU detected
✓ Requirements installed successfully
✓ spaCy model downloaded successfully
✓ All required packages imported successfully
✓ CUDA available: 1 GPU(s)
🚀 Setup complete! Starting analysis...
```

If you prefer manual control over the installation, see the manual installation section below.

## Manual Installation

## System Requirements

The DORA Controls Analyzer performs intensive natural language processing and **requires significant computational resources**. GPU acceleration is **strongly recommended** for practical use.

### Hardware Requirements:

#### GPU Setup (Recommended)
- **GPU**: NVIDIA GPU with CUDA support
  - **Minimum**: GTX 1060 6GB or RTX 3050 (4GB VRAM)
  - **Recommended**: RTX 3070/4060 or better (8GB+ VRAM)
  - **Professional**: RTX 4080/4090 or A100 (for large document sets)
- **CPU**: Quad-core processor (2.5GHz or higher) 
- **RAM**: 16GB RAM (32GB recommended for large document sets)
- **Storage**: 5GB for application and models + space for documents

#### CPU-Only Setup (Not Recommended)
- **CPU**: 8+ core processor (3.0GHz or higher)
- **RAM**: 32GB RAM minimum (64GB recommended)
- **Storage**: 5GB for application and models + space for documents
- **Note**: Processing will be **5-10x slower** than GPU setup

### Software Requirements:

- **Operating System**: 
  - Windows 10/11 with CUDA support, macOS 10.15+, or Linux (Ubuntu 18.04+ recommended)
- **Python Environment**: 
  - Python 3.8 or 3.9 (3.8 recommended for optimal compatibility)
  - Virtual environment strongly recommended
- **CUDA** (for GPU setup):
  - CUDA 11.8 or 12.1 (check compatibility with your GPU)
  - cuDNN libraries (usually installed with PyTorch)

### Network Requirements:

- Internet connection for initial model download (approximately 3-4GB of downloads)
- No ongoing internet connection required after initial setup

### Performance Expectations:

| Document Size | GPU (RTX 3070) | CPU (8-core i7) |
|---------------|----------------|------------------|
| Small policy (5-10 pages) | 30-60 seconds | 3-5 minutes |
| Medium policy (20-30 pages) | 1-2 minutes | 8-15 minutes |
| Large policy (50+ pages) | 3-5 minutes | 20-40 minutes |
| Multiple policies (10 docs) | 15-30 minutes | 2-4 hours |

**⚠️ Important**: CPU-only processing of large document sets may take several hours and is not practical for regular use.

## Requirements

The DORA Controls Analyzer requires the following:

1. **Python Environment**: Python 3.7 or newer
2. **Dependencies**: The following Python packages (automatically installed via requirements.txt):
   - `numpy==1.23.5`: Mathematical operations (specific version to avoid conflicts)
   - `pdfplumber>=0.7.4`: PDF text extraction
   - `pandas>=1.3.5`: Data manipulation
   - `spacy>=3.4.0`: Natural language processing
   - `torch>=1.13.0`: Machine learning backend
   - `transformers>=4.22.0`: Hugging Face transformer models
   - `sentence-transformers>=2.2.2`: Semantic similarity
   - `tqdm>=4.64.0`: Progress bars
   - `filelock>=3.8.0`: Thread-safe file operations
   - `psutil>=5.9.0`: System monitoring
3. **SpaCy Model**: The English language model (`en_core_web_lg`)
4. **DORA Legislation**: The official DORA PDF file
5. **Policy Documents**: Your organization's PDF policy documents

## Installation

**⚠️ Important: GPU Support Highly Recommended**

The DORA Controls Analyzer uses intensive NLP models that process large amounts of text. **GPU acceleration provides 5-10x faster processing** compared to CPU-only mode.

### Option 1: GPU Installation (Recommended)

**Prerequisites:**
- NVIDIA GPU with CUDA support
- CUDA 11.8 or compatible version installed
- At least 4GB GPU memory

1. Clone this repository:
   ```bash
   git clone https://github.com/asko123/dora_controls.git
   cd dora_controls
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install GPU-accelerated dependencies:
   ```bash
   pip install -r requirements-gpu.txt
   python -m spacy download en_core_web_lg
   ```

4. Verify GPU installation:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Option 2: CPU-Only Installation (Slower)

**Use this only if you don't have a CUDA-compatible GPU.**

1. Clone this repository:
   ```bash
   git clone https://github.com/asko123/dora_controls.git
   cd dora_controls
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install CPU-only dependencies:
   ```bash
   pip install -r requirements-cpu.txt
   python -m spacy download en_core_web_lg
   ```

### Performance Comparison

| Setup | Small Policy (5 pages) | Large Policy (50 pages) | Multiple Policies (10 docs) |
|-------|------------------------|--------------------------|------------------------------|
| **GPU** | ~30 seconds | ~3 minutes | ~15 minutes |
| **CPU** | ~3 minutes | ~20 minutes | ~2 hours |

### Troubleshooting GPU Installation

**CUDA Version Issues:**
```bash
# Check your CUDA version
nvidia-smi

# For CUDA 11.7, use:
pip install torch>=1.13.0+cu117 torchvision>=0.14.0+cu117 --index-url https://download.pytorch.org/whl/cu117

# For CUDA 12.1, use:
pip install torch>=1.13.0+cu121 torchvision>=0.14.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**Memory Issues:**
```bash
# Reduce batch size in DORAConfig if you get GPU memory errors
# Edit dora.py and modify:
DORAConfig.BATCH_SIZE = 5  # Reduce from default 10
```

## Directory Structure

Before running the analyzer, ensure your files are organized as follows:

```
📁 Your Upload Package:
├── CELEX_32022R2554_EN_TXT.pdf (DORA legislation)
├── setup_and_run.py (main orchestrator)
│   dora.py (main analyzer)
│   dora_domains.py (domain definitions)
│   dora_workbook_integration.py (domain analysis)
└── policies/
    ├── your_policy_1.pdf
    ├── your_policy_2.pdf
    └── ...
```

**Note:** The `policies` folder will be created automatically if it doesn't exist. This is where you should place all your organization's policy documents in PDF format for analysis.

## Running the Analyzer

## Two Analysis Approaches

The DORA Controls Analyzer provides **two different analysis approaches** depending on your needs:

### 1. Main Analysis Tool: `dora.py` (Recommended for most users)

**Use this for comprehensive gap analysis:**

```bash
python dora.py
```

**What it does:**
- Extracts RTS and ITS requirements directly from DORA legislation
- Analyzes your policy documents for compliance coverage using semantic similarity
- Generates detailed text-based gap analysis reports with article-specific findings
- Handles multiple policies in parallel with intelligent relevance screening
- Provides context-aware recommendations based on policy focus areas

**Use this when:**
- You want a complete DORA compliance assessment
- You need detailed gap analysis with specific recommendations
- You prefer comprehensive text-based reports
- You're doing an initial compliance assessment

### 2. Domain-Specific Analysis: `dora_workbook_integration.py` (For structured reporting)

**Use this for SOC2-style domain organization:**

```bash
python -m WorkShop.dora_workbook_integration
```

**What it does:**
- Organizes DORA into 8 structured domains (similar to SOC2 approach)
- Maps your policies to specific DORA domains and controls
- Generates Excel reports with charts, pivot tables, and visual dashboards
- Provides domain-by-domain compliance scoring and tracking

**Use this when:**
- You need Excel-based reports for management presentations
- You want SOC2-style domain organization and tracking
- You're monitoring ongoing compliance across structured domains
- You need visual dashboards and charts for compliance teams

### Quick Decision Guide

```bash
# For comprehensive text-based gap analysis (most users):
python dora.py

# For domain-specific Excel reports (compliance teams):
python -m WorkShop.dora_workbook_integration

# You can run both tools - they complement each other!
```

### Step-by-step Instructions

1. **Prepare the DORA legislation file**:
   - Place the DORA legislation PDF (CELEX_32022R2554_EN_TXT.pdf) in the root directory
   - You can download it from the [Official EU Law website](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022R2554)

2. **Prepare your policy documents**:
   - Place your PDF policy documents in the `policies` folder
   - If the folder doesn't exist, it will be created automatically on first run
   - Each PDF should be a separate policy or standard document

3. **Run the analyzer**:
   ```bash
   # From the project root directory
   python dora.py
   ```

4. **View the results**:
   - The analysis logs will be displayed in the console
   - A detailed gap analysis report will be generated in the `analysis_output` folder
   - Additional logs are stored in `dora_analyzer.log`

### Advanced Usage

You can customize the analyzer's behavior by modifying settings in the `DORAConfig` class:

```python
# Example: Adjust similarity thresholds for more lenient matching
DORAConfig.STRONG_MATCH_THRESHOLD = 0.5  # Lower from default 0.6
DORAConfig.COVERAGE_MATCH_THRESHOLD = 0.6  # Lower from default 0.7
DORAConfig.FINAL_COVERAGE_THRESHOLD = 0.7  # Lower from default 0.8

# Example: Increase text chunk size for processing larger documents
DORAConfig.MAX_TEXT_CHUNK_SIZE = 50000  # Default is 25000
```

To run with modified settings, you can create a custom script:

```python
# custom_run.py
from dora import DORAComplianceAnalyzer, DORAConfig

# Modify configuration
DORAConfig.STRONG_MATCH_THRESHOLD = 0.5
DORAConfig.COVERAGE_MATCH_THRESHOLD = 0.6
DORAConfig.FINAL_COVERAGE_THRESHOLD = 0.7

# Run analysis
analyzer = DORAComplianceAnalyzer("CELEX_32022R2554_EN_TXT.pdf")
analyzer.extract_technical_standards()
# Process each policy in the policies directory
# ...
analyzer.generate_gap_analysis_report()
```

## Detailed Setup Instructions

### Local Installation Requirements

**⚠️ Important: GPU Support Highly Recommended**

The DORA Controls Analyzer uses intensive NLP models that process large amounts of text. **GPU acceleration provides 5-10x faster processing** compared to CPU-only mode.

#### Prerequisites for Local Setup
- Python 3.8+ (Python 3.11 recommended)
- For GPU support: NVIDIA GPU with CUDA 11.8+ and at least 4GB GPU memory
- At least 8GB system RAM
- 5GB free disk space for models and dependencies

#### Automated Local Setup (Recommended)

The easiest way to get started locally:

```bash
git clone https://github.com/asko123/dora_controls.git
cd dora_controls

# Add your policy documents
mkdir -p policies
cp /path/to/your/policies/*.pdf policies/

# Run automated setup - handles everything automatically
python setup_and_run.py
```

The automated setup script will:
- Create a virtual environment
- Install appropriate dependencies (GPU or CPU)
- Download required ML models
- Run the analysis automatically

#### Manual Local Setup (Advanced)

For users who need custom configurations:

**With GPU Support:**
```bash
git clone https://github.com/asko123/dora_controls.git
cd dora_controls

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements-gpu.txt
python -m spacy download en_core_web_lg

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

mkdir -p policies
cp /path/to/your/policies/*.pdf policies/
python dora.py
```

**CPU-Only Setup:**
```bash
git clone https://github.com/asko123/dora_controls.git
cd dora_controls

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements-cpu.txt
python -m spacy download en_core_web_lg

mkdir -p policies
cp /path/to/your/policies/*.pdf policies/
python dora.py

# Set up Python environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install CPU-only dependencies (no GPU required)
pip install -r requirements-cpu.txt
python -m spacy download en_core_web_lg

# Create policies folder and add your PDFs
mkdir -p policies
cp /path/to/your/policies/*.pdf policies/

# Download DORA legislation if needed
# wget https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022R2554 -O CELEX_32022R2554_EN_TXT.pdf

# Run the analyzer (will be slower without GPU)
python dora.py
```

## Example Command Sequence

```bash
# Clone repository
git clone https://github.com/asko123/dora_controls.git
cd dora_controls

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Download DORA legislation if needed
# wget https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022R2554 -O CELEX_32022R2554_EN_TXT.pdf

# The policies folder will be created automatically on first run
# You can check if it exists or create it manually
if [ ! -d "policies" ]; then
    mkdir -p policies
    echo "Created policies folder. Please add your policy PDFs here."
fi

# Add your policy documents to the policies folder
cp /path/to/your/policies/*.pdf policies/

# Run the analyzer
python dora.py

# View the generated report (contains RTS and ITS compliance analysis)
cat analysis_output/dora_gap_analysis_*.txt
```

## Output Files

The analyzer generates the following outputs:

1. **Log File**: `dora_analyzer.log` - Contains detailed logs of the analysis process

2. **Gap Analysis Report**: Located in the `analysis_output` directory with filename format `dora_gap_analysis_YYYYMMDD_HHMMSS.txt`

3. **Cache Files**: Located in the `.cache` directory (for improved performance on subsequent runs)

## Interpreting Results

The gap analysis report includes:

1. **Executive Summary**: Overall compliance coverage percentage
2. **Policy Coverage Summary**: Per-policy coverage of Regulatory Technical Standards (RTS) and Implementing Technical Standards (ITS) requirements with policy focus areas
3. **Detailed Gap Analysis**: Article-by-article breakdown of gaps with clear distinction between:
   - Relevant gaps that need addressing within existing policies
   - Informational gaps that are outside a policy's scope
4. **Recommendations**: Policy-specific actions to improve compliance

A high coverage score (>80%) indicates good alignment with DORA requirements, while lower scores highlight areas needing attention.

## Customizing Analysis

You can adjust the analysis behavior by modifying the `DORAConfig` class in the code. Key settings include:

- **Similarity Thresholds**:
  - `STRONG_MATCH_THRESHOLD = 0.6` - Threshold for strong semantic matches
  - `COVERAGE_MATCH_THRESHOLD = 0.7` - Threshold for section matches
  - `FINAL_COVERAGE_THRESHOLD = 0.8` - Threshold for considering a requirement covered

- **Model Configuration**:
  - Uses `nlpaueb/legal-bert-base-uncased` for policy area classification (specifically trained on legal and regulatory texts)
  - Uses `sentence-transformers/all-mpnet-base-v2` for semantic similarity calculation (superior performance on complex regulatory text)

## Model Selection

The DORA Controls Analyzer uses specialized models optimized for legal and regulatory text analysis:

1. **LegalBERT (`nlpaueb/legal-bert-base-uncased`)**: 
   - Pre-trained specifically on legal and regulatory documents
   - Shows 5-10% improved performance on legal classification tasks compared to general-purpose models
   - Better recognition of legal terminology and regulatory concepts

2. **MPNet (`sentence-transformers/all-mpnet-base-v2`)**: 
   - Consistently outperforms other semantic similarity models by 2-5% on complex text benchmarks
   - Better matching for regulatory requirements with legal/technical terminology
   - More accurate similarity scores for compliance matching

These specialized models enable more accurate identification of policy areas and better matching between DORA requirements and policy documents, resulting in more reliable gap analysis results.

## Troubleshooting

- **Memory Errors**: Reduce the `MAX_TEXT_CHUNK_SIZE` in DORAConfig
- **Slow Processing**: Use a GPU or increase `BATCH_SIZE` for parallel processing
- **Low Coverage**: Adjust similarity thresholds in DORAConfig
- **Missing DORA File**: Ensure the DORA legislation PDF is in the root directory with the exact name "CELEX_32022R2554_EN_TXT.pdf"
- **No Policies Found**: Make sure your policy PDFs are in the "policies" folder. The folder will be created automatically on first run, but you need to manually add your PDF files to it before analysis
- **Irrelevant Results**: The tool automatically screens documents for DORA relevance, but you can adjust the screening logic in the `is_dora_relevant` function if needed

## Testing

The DORA Controls Analyzer includes a comprehensive test suite to verify functionality and ensure reliability. Run these tests to validate your installation and check system health.

### Available Test Scripts

#### 1. Basic Functionality Tests (`test_functionality.py`)
Tests core analyzer functionality and file structure:

```bash
python test_functionality.py
```

**What it tests:**
- DORAComplianceAnalyzer instantiation
- DORA domains loading (47 domains)
- DORA workbook creation
- Required file presence

**Expected output:**
```
✓ All required files present
✓ DORA domains loaded (47 domains)
✓ DORA workbook can be created
✓ DORAComplianceAnalyzer can be instantiated
✓ All functionality tests passed!
```

#### 2. Import Validation Tests (`test_imports.py`)
Validates all module imports work correctly:

```bash
python test_imports.py
```

**What it tests:**
- Core module imports (dora.py, dora_domains.py)
- Workbook integration imports
- External dependency imports (spaCy, transformers, etc.)

#### 3. CSV Compatibility Tests (`test_csv_roundtrip.py`)
Tests CSV export/import functionality for domain loading:

```bash
python test_csv_roundtrip.py
```

**What it tests:**
- CSV export/import round-trip compatibility
- CSV format consistency and validation
- Domain data integrity after export/import cycles

### Running All Tests

To run all tests sequentially:

```bash
python test_imports.py && python test_functionality.py && python test_csv_roundtrip.py
```

### Test Troubleshooting

- **Import Errors**: Run `pip install -r requirements-cpu.txt` or `requirements-gpu.txt`
- **Missing spaCy Model**: The setup script automatically downloads required models
- **File Not Found**: Ensure you're running tests from the project root directory
- **CSV Test Failures**: Check file permissions and available disk space

## DORA Compliance Workbook

The DORA Controls Analyzer includes a structured compliance workbook that categorizes DORA requirements into domains similar to SOC2. This feature supports both hardcoded domains and flexible CSV-based domain loading.

### Key Features

1. **Organizes DORA into Domains**: Breaks down DORA into 8 key domains with specific controls:
   - ICT Risk Management
   - ICT Incident Management
   - Digital Testing
   - Third-Party Risk
   - Information Sharing
   - Business Continuity
   - Awareness & Training
   - Governance & Oversight

2. **Domain-Specific Analysis**: Maps your policies to specific DORA domains and controls with precise matching.

3. **Comprehensive Excel Reports**: Generates detailed workbooks with:
   - Color-coded compliance status by domain
   - Pivot tables showing coverage across all domains
   - Charts visualizing domain compliance
   - Similarity scores for each requirement

4. **Flexible Domain Loading**: Load domains from CSV files or use built-in defaults with automatic fallback.

### CSV-Based Domain Loading

The workbook now supports loading DORA domains from CSV files, providing flexibility for customization while maintaining backward compatibility.

#### Default CSV Loading
```python
from dora_domains import create_dora_workbook

# Automatically tries to load from dora_domains.csv, falls back to hardcoded domains
workbook = create_dora_workbook()
```

#### Custom CSV File
```python
# Load from custom CSV file
workbook = create_dora_workbook("my_custom_domains.csv")

# Or load domains directly
from dora_domains import load_domains_from_csv
domains = load_domains_from_csv("custom_domains.csv")
```

#### CSV Format Requirements
Your CSV file must include these columns:
- `code`: Domain code (e.g., "RM 1.1", "IM 2.3")
- `article`: DORA article reference (e.g., "Art. 5", "Art. 17")
- `domain`: Domain category (e.g., "ICT Risk Management")
- `requirement`: Detailed requirement description

**Example CSV format:**
```csv
code,article,domain,requirement
RM 1.1,Art. 5,ICT Risk Management,Governance and strategy for ICT risk
IM 1.1,Art. 17,ICT Incident Management,ICT incident management process implementation
```

#### Error Handling and Fallback
The system gracefully handles CSV loading errors:
- **File Not Found**: Falls back to hardcoded domains (47 domains)
- **Invalid Format**: Validates CSV structure and provides clear error messages
- **Duplicate Codes**: Prevents duplicate domain codes
- **Missing Columns**: Ensures all required columns are present

### Using the DORA Workbook

#### Python API
```python
from WorkShop.dora_workbook_integration import run_workbook_analysis

# Run analysis and generate Excel report
report_path = run_workbook_analysis(
    dora_path="CELEX_32022R2554_EN_TXT.pdf",
    policies_folder="policies"
)
```

#### Command Line
```bash
# Generate a domain-specific compliance report
python -m WorkShop.dora_workbook_integration
```

#### Advanced Usage
```python
from dora_domains import DORAWorkbook

# Create workbook with custom CSV
workbook = DORAWorkbook("my_domains.csv")

# Export current domains to CSV for customization
workbook.export_to_csv("exported_domains.csv")

# Access domain data
print(f"Loaded {len(workbook.domains)} domains")
for domain in workbook.domains[:3]:  # Show first 3
    print(f"{domain['code']}: {domain['requirement']}")
```

### Workbook Output

The workbook analysis generates:
- **Excel Report**: Comprehensive compliance workbook with multiple sheets
- **Domain Coverage**: Color-coded status by domain
- **Pivot Tables**: Cross-domain analysis and coverage statistics
- **Charts**: Visual compliance dashboard
- **Detailed Mapping**: Policy-to-requirement mappings with similarity scores

The workbook functionality is ideal for compliance teams that need to map DORA requirements to specific controls and track compliance status across domains.

## Kubernetes Deployment

The DORA Controls Analyzer can be deployed in Kubernetes for scalable batch processing of compliance analysis.

### Prerequisites
- Kubernetes cluster with kubectl configured
- Docker for building images
- For GPU support: NVIDIA GPU Operator installed in cluster

### Quick Start
```bash
# Build and deploy
chmod +x k8s/deploy.sh
./k8s/deploy.sh

# Upload your policy files to the policies PVC
kubectl cp policies/ dora-analyzer/$(kubectl get pods -n dora-analyzer -l app=dora-analyzer -o jsonpath='{.items[0].metadata.name}'):/app/policies/

# Run CPU analysis
kubectl apply -f k8s/job-cpu.yaml

# Or run GPU analysis (if GPU nodes available)
kubectl apply -f k8s/job-gpu.yaml

# Monitor job progress
kubectl logs -f job/dora-analyzer-cpu -n dora-analyzer

# Retrieve results
kubectl cp dora-analyzer/$(kubectl get pods -n dora-analyzer -l app=dora-analyzer -o jsonpath='{.items[0].metadata.name}'):/app/analysis_output/ ./results/
```

### Architecture
- **CPU Variant**: Uses `Dockerfile.cpu` for CPU-only processing
- **GPU Variant**: Uses `Dockerfile.gpu` with CUDA support for faster processing
- **Storage**: Persistent volumes for policies, output, and model cache
- **Jobs**: Kubernetes Jobs for batch processing with automatic retry

### Resource Requirements
- **CPU Jobs**: 4Gi-8Gi memory, 2-4 CPU cores
- **GPU Jobs**: 6Gi-12Gi memory, 2-4 CPU cores, 1 GPU
- **Storage**: 1Gi policies, 5Gi output, 2Gi cache

### Container Images
The deployment includes two Docker variants:
- **CPU Image**: Built from `Dockerfile.cpu` using Python 3.11-slim base
- **GPU Image**: Built from `Dockerfile.gpu` using NVIDIA CUDA 11.8 runtime

### Deployment Files
- `k8s/namespace.yaml`: Creates the dora-analyzer namespace
- `k8s/configmap.yaml`: Environment variables configuration
- `k8s/storage.yaml`: Persistent volume claims for data storage
- `k8s/job-cpu.yaml`: CPU processing job definition
- `k8s/job-gpu.yaml`: GPU processing job definition
- `k8s/deploy.sh`: Automated deployment script

### Usage Examples
```bash
# Check job status
kubectl get jobs -n dora-analyzer

# View job logs
kubectl logs job/dora-analyzer-cpu -n dora-analyzer

# Clean up completed jobs
kubectl delete job dora-analyzer-cpu -n dora-analyzer
kubectl delete job dora-analyzer-gpu -n dora-analyzer

# Scale resources if needed
kubectl patch job dora-analyzer-cpu -n dora-analyzer -p '{"spec":{"parallelism":2}}'
```

The Kubernetes deployment provides a scalable, containerized solution for running DORA compliance analysis in cloud environments with proper resource management and persistent storage.
