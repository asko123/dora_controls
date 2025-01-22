"""
Module containing the process_policy function for DORA analysis
"""

import logging
from pathlib import Path
import pdfplumber
from tqdm import tqdm
import psutil
import gc
from .dora import DORAComplianceAnalyzer, DORAConfig

def process_policy(pdf_path, dora_path):
    """Process a single policy document.
    
    Args:
        pdf_path: Path to the policy PDF file
        dora_path: Path to the DORA regulation PDF file
    """
    try:
        policy_name = pdf_path.stem.replace("_", " ").title()
        logger = logging.getLogger('DORAAnalyzer')
        logger.info(f"Analyzing: {policy_name}")
        
        # Validate PDF before processing
        if not pdf_path.exists():
            raise FileNotFoundError(f"Policy file not found: {pdf_path}")
        if pdf_path.stat().st_size == 0:
            raise ValueError(f"Policy file is empty: {pdf_path}")
        
        # Extract text from PDF with proper resource management
        policy_text = ""
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                total_pages = len(pdf.pages)
                if total_pages == 0:
                    raise ValueError(f"No pages found in PDF: {pdf_path}")
                
                for page in tqdm(pdf.pages, 
                              desc=f"Reading {policy_name}", 
                              total=total_pages,
                              disable=DORAConfig.PROGRESS_BAR_DISABLE):
                    page_text = page.extract_text() or ""
                    policy_text += page_text + "\n\n"
                    
                    # Clear page object to free memory
                    del page
                    
                    # Check memory usage
                    memory_percent = psutil.Process().memory_percent()
                    if memory_percent > 80:  # 80% threshold
                        logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
                        gc.collect()
        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {str(e)}")
        
        # Validate extracted text
        if not policy_text.strip():
            raise ValueError(f"No text extracted from PDF: {pdf_path}")
        
        # Initialize analyzer with context manager
        with DORAComplianceAnalyzer(dora_path) as analyzer:
            # Analyze policy with retries
            result = analyzer.analyze_policy_document(policy_name, policy_text)
            
            # Validate result
            if not result or not isinstance(result, dict):
                raise ValueError(f"Invalid analysis result for {policy_name}")
            
            return {
                'policy_name': policy_name,
                'file_path': str(pdf_path),
                'result': result,
                'success': True
            }
            
    except Exception as e:
        logger = logging.getLogger('DORAAnalyzer')
        logger.error(f"Error analyzing {pdf_path.name}: {str(e)}", exc_info=True)
        return {
            'policy_name': pdf_path.stem.replace("_", " ").title(),
            'file_path': str(pdf_path),
            'error': str(e),
            'success': False
        } 
