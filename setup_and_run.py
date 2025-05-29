#!/usr/bin/env python3
"""
DORA Controls Analyzer - Automated Setup and Analysis Script

This script automatically:
1. Detects GPU availability
2. Installs appropriate requirements (GPU or CPU)
3. Downloads required models
4. Runs the main DORA analysis
5. Runs the workbook analysis
6. Provides comprehensive error handling and progress tracking
"""

import subprocess
import sys
import os
import platform
import time
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup_and_run.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DORASetupRunner:
    """Automated setup and runner for DORA Controls Analyzer"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.gpu_available = False
        self.requirements_installed = False
        self.models_downloaded = False
        self.dora_analysis_complete = False
        self.workbook_analysis_complete = False
        
    def print_banner(self):
        """Print welcome banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    DORA Controls Analyzer                    ║
║                  Automated Setup & Runner                    ║
║                                                              ║
║  This script will automatically set up and run the complete  ║
║  DORA compliance analysis workflow.                          ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        logger.info("Starting DORA Controls Analyzer automated setup and analysis")

    def check_prerequisites(self):
        """Check basic prerequisites"""
        logger.info("Checking prerequisites...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 7):
            logger.error(f"Python 3.7+ required. Current version: {python_version.major}.{python_version.minor}")
            return False
        logger.info(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check if we're in the right directory
        if not os.path.exists('dora.py'):
            logger.error("dora.py not found. Please run this script from the project root directory.")
            return False
        logger.info("✓ Project structure verified")
        
        # Check for DORA legislation file
        dora_file = "CELEX_32022R2554_EN_TXT.pdf"
        if not os.path.exists(dora_file):
            logger.warning(f"⚠️  DORA legislation file ({dora_file}) not found.")
            logger.warning("   Please download it from: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022R2554")
            logger.warning("   Continuing setup, but analysis will fail without this file.")
        else:
            logger.info(f"✓ DORA legislation file found: {dora_file}")
        
        # Check/create policies folder
        policies_folder = Path("policies")
        if not policies_folder.exists():
            policies_folder.mkdir()
            logger.info("✓ Created policies folder")
        
        policy_files = list(policies_folder.glob("*.pdf"))
        if not policy_files:
            logger.warning("⚠️  No policy PDF files found in policies folder.")
            logger.warning("   Please add your policy documents to the 'policies' folder before running analysis.")
        else:
            logger.info(f"✓ Found {len(policy_files)} policy files for analysis")
        
        return True

    def detect_gpu(self):
        """Detect if CUDA-compatible GPU is available"""
        logger.info("Detecting GPU availability...")
        
        try:
            # Try to detect NVIDIA GPU
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✓ NVIDIA GPU detected")
                self.gpu_available = True
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Try alternative detection methods
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("✓ CUDA support detected via PyTorch")
                self.gpu_available = True
                return True
        except ImportError:
            pass
        
        logger.info("ℹ️  No CUDA-compatible GPU detected. Will use CPU-only mode (slower).")
        self.gpu_available = False
        return False

    def install_requirements(self):
        """Install appropriate requirements based on GPU availability"""
        logger.info("Installing requirements...")
        
        # Determine which requirements file to use
        if self.gpu_available:
            requirements_file = "requirements-gpu.txt"
            logger.info("Installing GPU-accelerated requirements...")
        else:
            requirements_file = "requirements-cpu.txt"
            logger.info("Installing CPU-only requirements...")
        
        if not os.path.exists(requirements_file):
            logger.error(f"Requirements file not found: {requirements_file}")
            return False
        
        try:
            # Install requirements
            cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
            logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
            logger.info("✓ Requirements installed successfully")
            self.requirements_installed = True
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Requirements installation timed out (10 minutes)")
            return False

    def download_models(self):
        """Download required spaCy models"""
        logger.info("Downloading spaCy language model...")
        
        try:
            cmd = [sys.executable, "-m", "spacy", "download", "en_core_web_lg"]
            logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            logger.info("✓ spaCy model downloaded successfully")
            self.models_downloaded = True
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download spaCy model: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Model download timed out (5 minutes)")
            return False

    def verify_installation(self):
        """Verify that the installation was successful"""
        logger.info("Verifying installation...")
        
        try:
            # Test imports
            import torch
            import spacy
            import transformers
            import sentence_transformers
            
            logger.info("✓ All required packages imported successfully")
            
            # Test GPU if available
            if self.gpu_available:
                if torch.cuda.is_available():
                    logger.info(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
                else:
                    logger.warning("⚠️  CUDA not available despite GPU detection")
                    self.gpu_available = False
            
            # Test spaCy model
            try:
                nlp = spacy.load("en_core_web_lg")
                logger.info("✓ spaCy model loaded successfully")
            except OSError:
                logger.error("✗ spaCy model not found")
                return False
            
            return True
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
            return False

    def run_dora_analysis(self):
        """Run the main DORA analysis"""
        logger.info("Starting main DORA analysis...")
        logger.info("This may take several minutes depending on your hardware and document size...")
        
        try:
            cmd = [sys.executable, "dora.py"]
            logger.info(f"Running: {' '.join(cmd)}")
            
            # Run with real-time output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1, 
                universal_newlines=True
            )
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(f"DORA: {output.strip()}")
            
            return_code = process.poll()
            
            if return_code == 0:
                logger.info("✓ Main DORA analysis completed successfully")
                self.dora_analysis_complete = True
                return True
            else:
                logger.error(f"DORA analysis failed with return code: {return_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error running DORA analysis: {e}")
            return False

    def run_workbook_analysis(self):
        """Run the workbook analysis"""
        logger.info("Starting DORA workbook analysis...")
        
        try:
            cmd = [sys.executable, "-m", "WorkShop.dora_workbook_integration"]
            logger.info(f"Running: {' '.join(cmd)}")
            
            # Run with real-time output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1, 
                universal_newlines=True
            )
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(f"Workbook: {output.strip()}")
            
            return_code = process.poll()
            
            if return_code == 0:
                logger.info("✓ Workbook analysis completed successfully")
                self.workbook_analysis_complete = True
                return True
            else:
                logger.error(f"Workbook analysis failed with return code: {return_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error running workbook analysis: {e}")
            return False

    def print_summary(self):
        """Print final summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                        ANALYSIS SUMMARY                     ║
╠══════════════════════════════════════════════════════════════╣
║ Setup Phase:                                                 ║
║   GPU Detection:      {'✓ Passed' if self.gpu_available else '- CPU Only'}                            ║
║   Requirements:       {'✓ Installed' if self.requirements_installed else '✗ Failed'}                          ║
║   Models:             {'✓ Downloaded' if self.models_downloaded else '✗ Failed'}                        ║
║                                                              ║
║ Analysis Phase:                                              ║
║   Main Analysis:      {'✓ Completed' if self.dora_analysis_complete else '✗ Failed'}                        ║
║   Workbook Analysis:  {'✓ Completed' if self.workbook_analysis_complete else '✗ Failed'}                        ║
║                                                              ║
║ Total Duration: {str(duration).split('.')[0]:<42} ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        print(summary)
        logger.info(f"Total execution time: {duration}")
        
        # Print output locations
        if self.dora_analysis_complete:
            logger.info("📄 Main analysis report: analysis_output/dora_gap_analysis_*.txt")
        
        if self.workbook_analysis_complete:
            logger.info("📊 Workbook analysis report: dora_domain_compliance_*.xlsx")
        
        # Print next steps
        if self.dora_analysis_complete or self.workbook_analysis_complete:
            logger.info("\n🎉 Analysis complete! Check the output files for your DORA compliance results.")
        else:
            logger.error("\n❌ Analysis failed. Check the logs above for details.")

    def run(self):
        """Run the complete setup and analysis workflow"""
        try:
            self.print_banner()
            
            # Setup phase
            if not self.check_prerequisites():
                logger.error("Prerequisites check failed. Exiting.")
                return False
            
            self.detect_gpu()
            
            if not self.install_requirements():
                logger.error("Requirements installation failed. Exiting.")
                return False
            
            if not self.download_models():
                logger.error("Model download failed. Exiting.")
                return False
            
            if not self.verify_installation():
                logger.error("Installation verification failed. Exiting.")
                return False
            
            logger.info("🚀 Setup complete! Starting analysis...")
            
            # Analysis phase
            if not self.run_dora_analysis():
                logger.error("Main DORA analysis failed.")
                # Continue to try workbook analysis anyway
            
            if not self.run_workbook_analysis():
                logger.error("Workbook analysis failed.")
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("Setup interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return False
        finally:
            self.print_summary()


def main():
    """Main entry point"""
    runner = DORASetupRunner()
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 
