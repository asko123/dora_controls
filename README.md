# DORA Controls Analyzer

A comprehensive tool for analyzing compliance with the Digital Operational Resilience Act (DORA - EU Regulation 2022/2554) by identifying gaps between your organization's policies and DORA requirements.

## Quick Start

Choose the deployment method that best fits your needs:

### Kubernetes Deployment (Recommended for Production)

**Best for:** Production environments, scalable processing, cloud deployments

```bash
# Prerequisites: Kubernetes cluster with kubectl configured
git clone https://github.com/asko123/dora_controls.git
cd dora_controls

# Quick deployment
chmod +x k8s/deploy.sh
./k8s/deploy.sh

# Create DORA legislation ConfigMap
kubectl create configmap dora-legislation-config \
  --from-file=CELEX_32022R2554_EN_TXT.pdf \
  --namespace=dora-analyzer

# Add your policy files
kubectl cp policies/ dora-analyzer/$(kubectl get pods -n dora-analyzer -l app=dora-analyzer -o jsonpath='{.items[0].metadata.name}'):/app/policies/

# Run workbook analysis (currently working)
kubectl apply -f k8s/job-workbook-only.yaml

# Monitor progress and get results
kubectl logs -f job/dora-workbook-analyzer -n dora-analyzer
kubectl cp dora-analyzer/$(kubectl get pods -n dora-analyzer -l app=dora-analyzer -o jsonpath='{.items[0].metadata.name}'):/app/analysis_output/ ./results/
```

### Local Automated Setup (For Testing)

**Best for:** Quick testing, development, single-use analysis

```bash
git clone https://github.com/asko123/dora_controls.git
cd dora_controls

# Add your policy documents
mkdir -p policies
cp /path/to/your/policies/*.pdf policies/

# Run automated setup (handles everything automatically)
python setup_and_run.py
```

### Manual Local Setup (Advanced Users)

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
python setup_and_run.py
```

## Current Status

**Important Note:** The full ML pipeline is currently limited due to PyTorch security vulnerability CVE-2025-32434. The following functionality is available:

**Working Components:**
- Workbook analysis with DORA domain mapping
- Excel report generation 
- PDF text extraction
- Basic compliance gap identification
- Kubernetes deployment infrastructure

**Pending Components (awaiting PyTorch 2.6+ release):**
- ML-based semantic analysis
- Zero-shot classification
- Automated policy-requirement matching
- Advanced compliance scoring

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

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Required Python packages (see requirements files)
- Kubernetes cluster (for k8s deployment)

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

**Important**: CPU-only processing of large document sets may take several hours and is not practical for regular use.

## Requirements

The DORA Controls Analyzer requires the following:

1. **Python Environment**: Python 3.8 or newer
2. **Dependencies**: The following Python packages (automatically installed via requirements.txt):
   - `numpy>=1.21.0,<2.0.0`: Mathematical operations (version pinned for compatibility)
   - `pdfplumber>=0.7.4`: PDF text extraction
   - `pandas>=1.3.5`: Data manipulation
   - `spacy>=3.7.0`: Natural language processing
   - `transformers>=4.22.0`: Hugging Face transformer models
   - `sentence-transformers>=2.2.2`: Semantic similarity
   - `safetensors>=0.4.0`: Security mitigation for model loading
   - `tqdm>=4.64.0`: Progress bars
   - `filelock>=3.8.0`: Thread-safe file operations
   - `psutil>=5.9.0`: System monitoring
   - `xlsxwriter>=3.0.3`: Excel file creation
   - `openpyxl>=3.1.0`: Excel file reading/writing
3. **SpaCy Model**: The English language model (`en_core_web_lg`)
4. **DORA Legislation**: The official DORA PDF file
5. **Policy Documents**: Your organization's PDF policy documents

## Installation

**Important: GPU Support Highly Recommended**

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
pip install torch>=2.2.0+cu117 torchvision>=0.17.0+cu117 --index-url https://download.pytorch.org/whl/cu117

# For CUDA 12.1, use:
pip install torch>=2.2.0+cu121 torchvision>=0.17.0+cu121 --index-url https://download.pytorch.org/whl/cu121
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
Your Project Directory:
├── CELEX_32022R2554_EN_TXT.pdf (DORA legislation)
├── setup_and_run.py (main orchestrator)
├── dora.py (main analyzer)
├── dora_domains.py (domain definitions)
├── dora_workbook_integration.py (domain analysis)
├── requirements-cpu.txt (CPU dependencies)
├── requirements-gpu.txt (GPU dependencies)
├── k8s/ (Kubernetes deployment files)
│   ├── deploy.sh
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── storage.yaml
│   ├── job-cpu.yaml
│   ├── job-gpu.yaml
│   ├── job-workbook-only.yaml
│   └── k8s-deployment-guide.md
└── policies/
    ├── your_policy_1.pdf
    ├── your_policy_2.pdf
    └── ...
```

**Note:** The `policies` folder will be created automatically if it doesn't exist. This is where you should place all your organization's policy documents in PDF format for analysis.

## Running the Analyzer

### Two Analysis Approaches

The DORA Controls Analyzer provides **two different analysis approaches** depending on your needs:

### 1. Automated Setup Script: `setup_and_run.py` (Recommended for most users)

**Use this for comprehensive automated analysis:**

```bash
python setup_and_run.py
```

**What it does:**
- Detects GPU availability automatically
- Installs appropriate dependencies (GPU or CPU)
- Downloads required ML models
- Runs both main analysis and workbook generation
- Provides real-time progress updates and error handling

**Use this when:**
- You want a complete automated setup and analysis
- You're doing an initial compliance assessment
- You prefer hands-off operation with progress tracking

### 2. Workbook Analysis: `dora_workbook_integration.py` (For structured reporting)

**Use this for SOC2-style domain organization:**

```bash
python dora_workbook_integration.py
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
# For comprehensive automated analysis (most users):
python setup_and_run.py

# For domain-specific Excel reports (compliance teams):
python dora_workbook_integration.py

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
   python setup_and_run.py
   ```

4. **View the results**:
   - The analysis logs will be displayed in the console
   - Excel workbooks will be generated in the root directory
   - Additional logs are stored in `setup_and_run.log`

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

## Output Files

The analyzer generates the following outputs:

1. **Log File**: `setup_and_run.log` - Contains detailed logs of the analysis process

2. **Excel Reports**: Located in the root directory with filename format `dora_domain_compliance_YYYYMMDD_HHMMSS.xlsx`

3. **Cache Files**: Located in the `.cache` directory (for improved performance on subsequent runs)

## Interpreting Results

The Excel reports include:

1. **All Results**: Complete listing of all DORA requirements with compliance status
2. **Domain Coverage**: Cross-domain analysis showing coverage by policy areas
3. **Coverage Summary**: High-level statistics and compliance percentages

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
- **ML Model Errors**: Currently limited by PyTorch security issue CVE-2025-32434; use workbook-only analysis until resolved

## Testing

The DORA Controls Analyzer includes a test script to verify functionality and ensure reliability.

### Available Test Script

#### Functionality Tests (`test_functionality.py`)
Tests core analyzer functionality and file structure:

```bash
python test_functionality.py
```

**What it tests:**
- DORA domains loading (47 domains)
- DORA workbook creation
- Required file presence
- Basic system functionality

**Expected output:**
```
Testing DORA Controls Analyzer functionality...
==================================================
All required files present
DORA domains loaded (47 domains)
DORA workbook can be created
==================================================
All tests passed!
```

### Test Troubleshooting

- **Import Errors**: Run `pip install -r requirements-cpu.txt` or `requirements-gpu.txt`
- **Missing spaCy Model**: Run `python -m spacy download en_core_web_lg`
- **File Not Found**: Ensure you're running tests from the project root directory

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

#### Command Line
```bash
# Generate a domain-specific compliance report
python dora_workbook_integration.py
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
# Build and deploy infrastructure
chmod +x k8s/deploy.sh
./k8s/deploy.sh

# Create DORA legislation ConfigMap
kubectl create configmap dora-legislation-config \
  --from-file=CELEX_32022R2554_EN_TXT.pdf \
  --namespace=dora-analyzer

# Upload your policy files to the policies PVC
kubectl cp policies/ dora-analyzer/$(kubectl get pods -n dora-analyzer -l app=dora-analyzer -o jsonpath='{.items[0].metadata.name}'):/app/policies/

# Run workbook analysis (currently working)
kubectl apply -f k8s/job-workbook-only.yaml

# Monitor job progress
kubectl logs -f job/dora-workbook-analyzer -n dora-analyzer

# Retrieve results
kubectl cp dora-analyzer/$(kubectl get pods -n dora-analyzer -l app=dora-analyzer -o jsonpath='{.items[0].metadata.name}'):/app/analysis_output/ ./results/
```

### Architecture
- **CPU Variant**: Uses `Dockerfile.cpu` for CPU-only processing
- **GPU Variant**: Uses `Dockerfile.gpu` with CUDA support for faster processing
- **Workbook Variant**: Uses `job-workbook-only.yaml` for current functionality
- **Storage**: Persistent volumes for policies, output, and model cache
- **Jobs**: Kubernetes Jobs for batch processing with automatic retry

### Resource Requirements
- **CPU Jobs**: 4Gi-8Gi memory, 2-4 CPU cores
- **GPU Jobs**: 6Gi-12Gi memory, 2-4 CPU cores, 1 GPU
- **Workbook Jobs**: 2Gi-4Gi memory, 1-2 CPU cores
- **Storage**: 1Gi policies, 5Gi output, 2Gi cache

### Container Images
The deployment includes Docker variants:
- **CPU Image**: Built from `Dockerfile.cpu` using Python 3.11-slim base
- **GPU Image**: Built from `Dockerfile.gpu` using NVIDIA CUDA 12.9.1 runtime

### Deployment Files
- `k8s/namespace.yaml`: Creates the dora-analyzer namespace
- `k8s/configmap.yaml`: Environment variables configuration
- `k8s/storage.yaml`: Persistent volume claims for data storage
- `k8s/job-cpu.yaml`: CPU processing job definition
- `k8s/job-gpu.yaml`: GPU processing job definition
- `k8s/job-workbook-only.yaml`: Workbook-only job (currently working)
- `k8s/dora-legislation-configmap.yaml`: DORA legislation configuration
- `k8s/deploy.sh`: Automated deployment script
- `k8s/k8s-deployment-guide.md`: Detailed deployment guide

### Current Limitations

Due to PyTorch security vulnerability CVE-2025-32434:
- **CPU and GPU jobs**: Limited by ML model initialization issues
- **Workbook-only jobs**: Fully functional for Excel report generation
- **Infrastructure**: All Kubernetes components fully operational

### Usage Examples
```bash
# Check job status
kubectl get jobs -n dora-analyzer

# View job logs
kubectl logs job/dora-workbook-analyzer -n dora-analyzer

# Clean up completed jobs
kubectl delete job dora-workbook-analyzer -n dora-analyzer

# Scale workbook analysis for multiple policy sets
sed 's/dora-workbook-analyzer/dora-workbook-analyzer-2/' k8s/job-workbook-only.yaml | kubectl apply -f -
```

The Kubernetes deployment provides a scalable, containerized solution for running DORA compliance analysis in cloud environments with proper resource management and persistent storage. Currently optimized for workbook analysis until ML pipeline limitations are resolved.

## Security Considerations

### Container Security
- Non-root user (UID 1000) in containers
- Health checks for container validation
- Resource limits enforced
- Network policies can be applied

### Data Security
- Persistent volumes for data isolation
- Namespace isolation in Kubernetes
- ConfigMap for environment variables
- Policy documents require secure upload process

## Support and Documentation

For detailed deployment instructions, troubleshooting, and advanced configuration, see:
- `k8s/k8s-deployment-guide.md` - Comprehensive Kubernetes deployment guide
- `test_functionality.py` - System validation and health checks
- GitHub issues for community support

## License

This project is available under the terms specified in the repository license.