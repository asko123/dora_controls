import pdfplumber
import pandas as pd
import spacy
import re
import torch
import torch.nn.functional as F  # Add this import
from typing import Tuple, List, Dict
from collections import defaultdict
from pathlib import Path
import difflib
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
from collections import Counter
import logging
from tqdm import tqdm
import functools
import time
from logging.handlers import RotatingFileHandler
import json
from datetime import timedelta
import os
from multiprocessing import Pool, cpu_count
import sys
import threading
from filelock import FileLock
import psutil
import gc
from sentence_transformers import SentenceTransformer
import hashlib
from functools import partial

# Remove import from dora_processor
# from .dora_processor import process_policy

# Remove the fallback implementation since we're adding the real implementation


def setup_logging(log_file: str = "dora_analyzer.log"):
    """Set up logging configuration."""
    logger = logging.getLogger('DORAAnalyzer')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_execution_time(func):
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('DORAAnalyzer')
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Finished {func.__name__} in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


class DORAConfig:
    """Configuration settings for DORA compliance analysis."""
    
    # Similarity thresholds
    STRONG_MATCH_THRESHOLD = 0.6
    COVERAGE_MATCH_THRESHOLD = 0.7
    FINAL_COVERAGE_THRESHOLD = 0.8
    
    # Text processing limits
    MAX_TEXT_CHUNK_SIZE = 25000
    MIN_SENTENCE_WORDS = 5
    
    # LLM settings
    ZERO_SHOT_MODEL = "nlpaueb/legal-bert-base-uncased"  # Legal domain-specific model for better recognition of legal and regulatory terminology
    LLM_THRESHOLD = 0.5  # Confidence threshold for predictions
    
    # File patterns
    PDF_GLOB_PATTERN = "**/*.pdf"
    
    # Output settings
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    OUTPUT_FOLDER_NAME = "analysis_output"
    REPORT_FILENAME_TEMPLATE = "dora_gap_analysis_{}.txt"
    
    # Regular expressions
    ARTICLE_PATTERN = r"Article\s+(\d+[a-z]?)\s*[–-]?\s*([^\n]+)(?:\n|\r\n?)(.*?)(?=Article\s+\d+[a-z]?|$)"
    SENTENCE_SPLIT_PATTERN = r'[.!?]+'

    # RTS and ITS patterns
    RTS_PATTERNS = [
        r"(?i)regulatory\s+technical\s+standards?\s+(?:shall|should|must|may|will)\s+[^.]+",
        r"(?i)RTS\s+(?:shall|should|must|may|will)\s+[^.]+",
        r"(?i)(?:develop|specify|adopt|implement)\s+regulatory\s+technical\s+standards?\s+[^.]+",
        r"(?i)regulatory\s+technical\s+standards?\s+(?:for|on|regarding|concerning)\s+[^.]+"
    ]
    
    ITS_PATTERNS = [
        r"(?i)implementing\s+technical\s+standards?\s+(?:shall|should|must|may|will)\s+[^.]+",
        r"(?i)ITS\s+(?:shall|should|must|may|will)\s+[^.]+",
        r"(?i)(?:develop|specify|adopt|implement)\s+implementing\s+technical\s+standards?\s+[^.]+",
        r"(?i)implementing\s+technical\s+standards?\s+(?:for|on|regarding|concerning)\s+[^.]+"
    ]

    # Logging settings
    LOG_FILE = "dora_analyzer.log"
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL = logging.INFO
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5

    # Caching settings
    CACHE_DIR = ".cache"
    POLICY_AREA_CACHE_FILE = "policy_area_cache.json"
    SIMILARITY_CACHE_FILE = "similarity_cache.json"
    CACHE_EXPIRY_DAYS = 7

    # Progress bar settings
    PROGRESS_BAR_DISABLE = False
    PROGRESS_BAR_COLOUR = "green"
    PROGRESS_BAR_DESC_LEN = 30

    # Memory management settings
    MEMORY_THRESHOLD_MB = 500  # Trigger GC when memory increase exceeds this value
    MEMORY_CHECK_INTERVAL = 10  # Check memory usage every N pages
    MEMORY_WARNING_THRESHOLD = 0.8  # Warn when memory usage exceeds 80%
    
    # File paths and directories
    OUTPUT_DIR = "reports"
    REPORT_FILENAME_TEMPLATE = "gap_analysis_{timestamp}.md"
    
    # Text processing settings
    MIN_SIMILARITY_SCORE = 0.75
    MAX_TEXT_LENGTH = 1000
    BATCH_SIZE = 10
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings."""
        logger = logging.getLogger('DORAAnalyzer')
        
        # Validate similarity thresholds
        if not (0 <= cls.STRONG_MATCH_THRESHOLD <= 1):
            raise ValueError("STRONG_MATCH_THRESHOLD must be between 0 and 1")
        if not (0 <= cls.COVERAGE_MATCH_THRESHOLD <= 1):
            raise ValueError("COVERAGE_MATCH_THRESHOLD must be between 0 and 1")
        if not (0 <= cls.FINAL_COVERAGE_THRESHOLD <= 1):
            raise ValueError("FINAL_COVERAGE_THRESHOLD must be between 0 and 1")
        
        # Validate text processing limits
        if cls.MAX_TEXT_CHUNK_SIZE <= 0:
            raise ValueError("MAX_TEXT_CHUNK_SIZE must be positive")
        if cls.MIN_SENTENCE_WORDS <= 0:
            raise ValueError("MIN_SENTENCE_WORDS must be positive")
        
        # Validate LLM settings
        if not cls.ZERO_SHOT_MODEL:
            raise ValueError("ZERO_SHOT_MODEL cannot be empty")
        
        # Validate regular expressions
        try:
            re.compile(cls.ARTICLE_PATTERN)
            re.compile(cls.SENTENCE_SPLIT_PATTERN)
            for pattern in cls.RTS_PATTERNS:
                re.compile(pattern)
            for pattern in cls.ITS_PATTERNS:
                re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regular expression pattern: {str(e)}")
        
        # Validate logging settings
        if cls.LOG_MAX_BYTES <= 0:
            raise ValueError("LOG_MAX_BYTES must be positive")
        if cls.LOG_BACKUP_COUNT < 0:
            raise ValueError("LOG_BACKUP_COUNT cannot be negative")
        
        # Validate caching settings
        if cls.CACHE_EXPIRY_DAYS <= 0:
            raise ValueError("CACHE_EXPIRY_DAYS must be positive")
        
        # Validate progress bar settings
        if not isinstance(cls.PROGRESS_BAR_DISABLE, bool):
            raise ValueError("PROGRESS_BAR_DISABLE must be a boolean")
        if cls.PROGRESS_BAR_DESC_LEN <= 0:
            raise ValueError("PROGRESS_BAR_DESC_LEN must be positive")
        
        logger.info("Configuration validation successful")


class CacheManager:
    """Manages caching operations for the analyzer."""
    
    def __init__(self, cache_dir: str = DORAConfig.CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('DORAAnalyzer')
        
        # Initialize cache files
        self.policy_area_cache_file = self.cache_dir / DORAConfig.POLICY_AREA_CACHE_FILE
        self.similarity_cache_file = self.cache_dir / DORAConfig.SIMILARITY_CACHE_FILE
        
        # Initialize locks for thread safety
        self._policy_area_lock = threading.Lock()
        self._similarity_lock = threading.Lock()
        
        # Load existing caches
        self.policy_area_cache = self._load_cache(self.policy_area_cache_file)
        self.similarity_cache = self._load_cache(self.similarity_cache_file)
    
    def _load_cache(self, cache_file: Path) -> Dict:
        """Load cache from file with thread safety."""
        try:
            if cache_file.exists():
                with FileLock(str(cache_file) + ".lock"):
                    with open(cache_file, 'r') as f:
                        cache = json.loads(f.read())
                    self.logger.info(f"Loaded cache from {cache_file}")
                    return self._clean_expired_entries(cache)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading cache {cache_file}: {str(e)}")
            return {}
    
    def _save_cache(self, cache: Dict, cache_file: Path) -> None:
        """Save cache to file with thread safety."""
        try:
            with FileLock(str(cache_file) + ".lock"):
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)
                self.logger.info(f"Saved cache to {cache_file}")
        except Exception as e:
            self.logger.error(f"Error saving cache {cache_file}: {str(e)}")
    
    def _clean_expired_entries(self, cache: Dict) -> Dict:
        """Remove expired cache entries."""
        now = datetime.now()
        expiry = now - timedelta(days=DORAConfig.CACHE_EXPIRY_DAYS)
        
        cleaned_cache = {
            k: v for k, v in cache.items()
            if datetime.fromisoformat(v.get('timestamp', '2000-01-01')).replace(tzinfo=None) > expiry
        }
        
        removed = len(cache) - len(cleaned_cache)
        if removed > 0:
            self.logger.info(f"Removed {removed} expired cache entries")
        
        return cleaned_cache
    
    def get_policy_area(self, text_hash: str) -> str:
        """Get cached policy area for text with thread safety."""
        with self._policy_area_lock:
            cache_entry = self.policy_area_cache.get(text_hash)
            if cache_entry:
                self.logger.debug(f"Cache hit for policy area: {text_hash}")
                return cache_entry['value']
        return None
    
    def set_policy_area(self, text_hash: str, policy_area: str) -> None:
        """Cache policy area for text with thread safety."""
        with self._policy_area_lock:
            self.policy_area_cache[text_hash] = {
                'value': policy_area,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache(self.policy_area_cache, self.policy_area_cache_file)
    
    def get_similarity(self, text_pair_hash: str) -> float:
        """Get cached similarity score for text pair with thread safety."""
        with self._similarity_lock:
            cache_entry = self.similarity_cache.get(text_pair_hash)
            if cache_entry:
                self.logger.debug(f"Cache hit for similarity: {text_pair_hash}")
                return cache_entry['value']
        return None
    
    def set_similarity(self, text_pair_hash: str, similarity: float) -> None:
        """Cache similarity score for text pair with thread safety."""
        with self._similarity_lock:
            self.similarity_cache[text_pair_hash] = {
                'value': similarity,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache(self.similarity_cache, self.similarity_cache_file)
    
    def clear_caches(self) -> None:
        """Clear all caches with thread safety."""
        with self._policy_area_lock, self._similarity_lock:
            self.policy_area_cache = {}
            self.similarity_cache = {}
            self._save_cache(self.policy_area_cache, self.policy_area_cache_file)
            self._save_cache(self.similarity_cache, self.similarity_cache_file)
            self.logger.info("Cleared all caches")


class TextProcessor:
    """Utility class for text processing operations."""
    
    # Class-level cache for processed text
    _clean_text_cache = {}
    _sentence_cache = {}
    _cache_lock = threading.Lock()
    _max_cache_size = 1000
    
    @classmethod
    def _get_cache_key(cls, text: str, operation: str) -> str:
        """Generate a cache key for text operations."""
        return f"{operation}_{hash(text)}"
    
    @classmethod
    def _add_to_cache(cls, key: str, value: any, cache_dict: Dict) -> None:
        """Add value to cache with size limit enforcement."""
        with cls._cache_lock:
            if len(cache_dict) >= cls._max_cache_size:
                # Remove oldest 10% of entries when cache is full
                remove_count = cls._max_cache_size // 10
                for _ in range(remove_count):
                    cache_dict.pop(next(iter(cache_dict)))
            cache_dict[key] = value
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content with caching."""
        if not text:
            return ""
        
        # Check cache first
        cache_key = TextProcessor._get_cache_key(text, 'clean')
        with TextProcessor._cache_lock:
            if cache_key in TextProcessor._clean_text_cache:
                return TextProcessor._clean_text_cache[cache_key]
        
        try:
            # Convert to string if not already
            text = str(text)
            
            # Basic cleaning
            cleaned = text.strip()
            
            # Process in smaller chunks for large texts
            if len(cleaned) > DORAConfig.MAX_TEXT_CHUNK_SIZE:
                chunks = [cleaned[i:i+DORAConfig.MAX_TEXT_CHUNK_SIZE] 
                         for i in range(0, len(cleaned), DORAConfig.MAX_TEXT_CHUNK_SIZE)]
                processed_chunks = []
                
                for chunk in chunks:
                    # Remove extra whitespace
                    chunk = ' '.join(chunk.split())
                    
                    # Remove special characters but keep essential punctuation
                    chunk = re.sub(r'[^\w\s.,;:?!()\-\'\"]+', ' ', chunk)
                    
                    # Normalize whitespace around punctuation
                    chunk = re.sub(r'\s*([.,;:?!()\-])\s*', r'\1 ', chunk)
                    
                    # Remove multiple spaces
                    chunk = re.sub(r'\s+', ' ', chunk)
                    
                    processed_chunks.append(chunk)
                
                cleaned = ' '.join(processed_chunks)
            else:
                # Process small text directly
                cleaned = ' '.join(cleaned.split())
                cleaned = re.sub(r'[^\w\s.,;:?!()\-\'\"]+', ' ', cleaned)
                cleaned = re.sub(r'\s*([.,;:?!()\-])\s*', r'\1 ', cleaned)
                cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Remove empty lines
            cleaned = '\n'.join(line.strip() for line in cleaned.split('\n') if line.strip())
            
            # Cache the result
            TextProcessor._add_to_cache(cache_key, cleaned, TextProcessor._clean_text_cache)
            
            return cleaned.strip()
            
        except Exception as e:
            logger = logging.getLogger('DORAAnalyzer')
            logger.error(f"Error cleaning text: {str(e)}", exc_info=True)
            return text.strip() if text else ""
    
    @staticmethod
    def extract_sentences(text: str, min_words: int = DORAConfig.MIN_SENTENCE_WORDS) -> List[str]:
        """Extract meaningful sentences from text with caching."""
        if not text:
            return []
        
        # Check cache first
        cache_key = TextProcessor._get_cache_key(f"{text}_{min_words}", 'sentences')
        with TextProcessor._cache_lock:
            if cache_key in TextProcessor._sentence_cache:
                return TextProcessor._sentence_cache[cache_key]
        
        try:
            # Split text into sentences
            sentences = [s.strip() for s in re.split(DORAConfig.SENTENCE_SPLIT_PATTERN, text) if s.strip()]
            
            # Filter by minimum word count
            filtered_sentences = [s for s in sentences if len(s.split()) >= min_words]
            
            # Cache the result
            TextProcessor._add_to_cache(cache_key, filtered_sentences, TextProcessor._sentence_cache)
            
            return filtered_sentences
            
        except Exception as e:
            logger = logging.getLogger('DORAAnalyzer')
            logger.error(f"Error extracting sentences: {str(e)}", exc_info=True)
            return []
    
    @staticmethod
    def clean_cell_content(cell: str) -> str:
        """Clean individual table cell content."""
        if cell is None:
            return ""
        return TextProcessor.clean_text(str(cell))
    
    @staticmethod
    def _extract_full_requirement(text: str, start_pos: int, context_window: int = 200) -> str:
        """Extract the full requirement context from the text with improved accuracy.
        
        Args:
            text (str): Source text to extract from
            start_pos (int): Starting position of the requirement
            context_window (int, optional): Size of context window around match. Defaults to 200.
            
        Returns:
            str: Extracted requirement text with context
        """
        logger = logging.getLogger('DORAAnalyzer')
        try:
            if not text or start_pos < 0 or start_pos >= len(text):
                logger.warning(f"Invalid text or position: text_len={len(text) if text else 0}, pos={start_pos}")
                return ""
                
            # Find sentence boundaries
            sentence_start = start_pos
            sentence_end = start_pos
            
            # Look for sentence start (period + space or newline)
            for i in range(start_pos - 1, max(-1, start_pos - context_window), -1):
                if i < 0:
                    sentence_start = 0
                    break
                if text[i] == '.' and (i + 1 >= len(text) or text[i + 1].isspace()):
                    sentence_start = i + 1
                    while sentence_start < len(text) and text[sentence_start].isspace():
                        sentence_start += 1
                    break
                if text[i] == '\n':
                    sentence_start = i + 1
                    break
            
            # Look for sentence end (period + space or newline)
            for i in range(start_pos, min(len(text), start_pos + context_window)):
                if i >= len(text):
                    sentence_end = len(text)
                    break
                if text[i] == '.' and (i + 1 >= len(text) or text[i + 1].isspace()):
                    sentence_end = i + 1
                    break
                if text[i] == '\n':
                    sentence_end = i
                    break
            
            # Extract and clean the requirement text
            requirement = text[sentence_start:sentence_end].strip()
            
            # If no clear sentence boundaries found, use context window
            if not requirement:
                context_start = max(0, start_pos - context_window // 2)
                context_end = min(len(text), start_pos + context_window // 2)
                requirement = text[context_start:context_end].strip()
                logger.debug(f"No clear sentence boundaries found, using context window: {context_window} chars")
            
            # Log extraction details
            logger.debug(f"Extracted requirement: start={sentence_start}, end={sentence_end}, length={len(requirement)}")
            
            return requirement
            
        except Exception as e:
            logger.error(f"Error extracting requirement: {str(e)}", exc_info=True)
            # Fallback to simple context window extraction
            try:
                context_start = max(0, start_pos - context_window // 2)
                context_end = min(len(text), start_pos + context_window // 2)
                return text[context_start:context_end].strip()
            except Exception as e2:
                logger.error(f"Fallback extraction failed: {str(e2)}", exc_info=True)
                return ""


class RequirementAnalyzer:
    """Class for analyzing and processing requirements."""
    
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.logger = logging.getLogger('DORAAnalyzer')
    
    def analyze_requirement_coverage(self, requirement: Dict, policy_text: str) -> Dict:
        """Analyze how well a requirement is covered in the policy text with enhanced accuracy.
        
        Args:
            requirement (Dict): Requirement dictionary containing text and metadata
            policy_text (str): Policy text to analyze against
            
        Returns:
            Dict: Coverage analysis results including similarity scores and matching sections
        """
        try:
            # Input validation and normalization
            if not requirement or not isinstance(requirement, dict):
                self.logger.error("Invalid requirement format")
                return self._create_empty_coverage_result()
                
            if 'requirement_text' not in requirement:
                self.logger.error("Missing requirement_text in requirement")
                return self._create_empty_coverage_result()
            
            # Process requirement text
            req_text = ' '.join(requirement['requirement_text']) if isinstance(requirement['requirement_text'], list) else str(requirement['requirement_text'])
            req_text = req_text.strip().lower()
            
            # Process policy text
            policy_text = ' '.join(policy_text) if isinstance(policy_text, list) else str(policy_text)
            policy_text = policy_text.strip().lower()
            
            # Skip if either text is empty
            if not req_text or not policy_text:
                self.logger.warning("Empty requirement or policy text")
                return self._create_empty_coverage_result()
            
            # Log analysis start
            self.logger.info(f"Analyzing coverage for requirement: {req_text[:200]}...")
            
            # Get sentences and calculate similarity
            policy_sentences = TextProcessor.extract_sentences(policy_text)
            if not policy_sentences:
                self.logger.warning("No valid sentences found in policy text")
                return self._create_empty_coverage_result()
                
            # Process requirement with length limit
            req_doc = self.nlp(req_text[:DORAConfig.MAX_TEXT_CHUNK_SIZE])
            
            # Initialize tracking variables
            max_similarity = 0.0
            matching_sections = []
            processed_sentences = 0
            skipped_sentences = 0
            
            # Process each sentence
            for sentence in policy_sentences:
                try:
                    similarity = self._calculate_sentence_similarity(req_doc, sentence)
                    processed_sentences += 1
                    
                    if similarity is None:
                        skipped_sentences += 1
                        continue
                        
                    max_similarity = max(max_similarity, similarity)
                    if similarity > DORAConfig.COVERAGE_MATCH_THRESHOLD:
                        matching_sections.append({
                            'text': sentence,
                            'similarity': similarity
                        })
                except Exception as e:
                    self.logger.error(f"Error processing sentence: {str(e)}", exc_info=True)
                    skipped_sentences += 1
                    continue
            
            # Determine coverage based on similarity
            is_covered = max_similarity > DORAConfig.FINAL_COVERAGE_THRESHOLD
            
            # Log analysis results
            self.logger.info(f"Coverage analysis complete:")
            self.logger.info(f"- Processed sentences: {processed_sentences}")
            self.logger.info(f"- Skipped sentences: {skipped_sentences}")
            self.logger.info(f"- Max similarity: {max_similarity:.3f}")
            self.logger.info(f"- Coverage status: {'Covered' if is_covered else 'Not covered'}")
            if matching_sections:
                self.logger.info(f"- Found {len(matching_sections)} matching sections")
            
            # Return results
            return {
                'covered': is_covered,
                'similarity_score': max_similarity,
                'matching_sections': sorted(matching_sections, 
                                         key=lambda x: x['similarity'], 
                                         reverse=True)[:3],  # Top 3 matching sections
                'stats': {
                    'processed_sentences': processed_sentences,
                    'skipped_sentences': skipped_sentences,
                    'matching_sections_count': len(matching_sections)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing requirement coverage: {str(e)}", exc_info=True)
            return self._create_empty_coverage_result()
            
    def _create_empty_coverage_result(self) -> Dict:
        """Create an empty coverage result with default values."""
        return {
            'covered': False,
            'similarity_score': 0.0,
            'matching_sections': [],
            'stats': {
                'processed_sentences': 0,
                'skipped_sentences': 0,
                'matching_sections_count': 0
            }
        }
    
    def _calculate_sentence_similarity(self, req_doc, sentence: str) -> float:
        """Calculate semantic similarity between requirement and sentence with caching.
        
        Args:
            req_doc: spaCy Doc object of the requirement text
            sentence (str): Sentence to compare against
            
        Returns:
            float: Similarity score between 0 and 1, or None if calculation fails
        """
        try:
            # Generate cache key
            cache_key = f"{req_doc.text}_{sentence}"
            
            # Check cache first
            with TextProcessor._cache_lock:
                if cache_key in TextProcessor._similarity_cache:
                    self.logger.debug("Using cached similarity score")
                    return TextProcessor._similarity_cache[cache_key]
            
            # Clean and normalize sentence
            sentence = sentence.strip().lower()
            if not sentence:
                return None
                
            # Process sentence with length limit
            sentence_doc = self.nlp(sentence[:DORAConfig.MAX_TEXT_CHUNK_SIZE])
            
            # Check vector norms
            if not sentence_doc.vector_norm or not req_doc.vector_norm:
                self.logger.debug("Zero vector norm detected, skipping similarity calculation")
                return None
                
            # Calculate similarity
            similarity = req_doc.similarity(sentence_doc)
            
            # Cache the result
            with TextProcessor._cache_lock:
                TextProcessor._similarity_cache[cache_key] = similarity
                
            # Log high similarity matches
            if similarity > DORAConfig.STRONG_MATCH_THRESHOLD:
                self.logger.debug(f"High similarity match ({similarity:.3f}): {sentence[:100]}...")
                
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating sentence similarity: {str(e)}", exc_info=True)
            return None
    
    def process_requirements(self, requirements: Dict[str, List], policy_text: str, requirement_type: str) -> List[Dict]:
        """Process a set of requirements against policy text with enhanced tracking and error handling.
        
        Args:
            requirements (Dict[str, List]): Requirements organized by article number
            policy_text (str): Policy text to analyze against
            requirement_type (str): Type of requirements (RTS or ITS)
            
        Returns:
            List[Dict]: List of processed requirements with coverage analysis
        """
        try:
            self.logger.info(f"Processing {requirement_type} requirements...")
            
            # Input validation
            if not isinstance(requirements, dict):
                self.logger.error(f"Invalid requirements format: {type(requirements)}")
                return []
                
            if not policy_text:
                self.logger.error("Empty policy text provided")
                return []
            
            # Initialize tracking variables
            results = []
            total_requirements = sum(len(reqs) for reqs in requirements.values())
            processed_count = 0
            covered_count = 0
            error_count = 0
            
            self.logger.info(f"Found {total_requirements} {requirement_type} requirements to process")
            
            # Process requirements with progress bar
            with tqdm(total=total_requirements, 
                     desc=f"Processing {requirement_type} Requirements", 
                     disable=DORAConfig.PROGRESS_BAR_DISABLE) as pbar:
                
                for article_num, reqs in requirements.items():
                    self.logger.debug(f"Processing Article {article_num} ({len(reqs)} requirements)")
                    
                    for req in reqs:
                        try:
                            # Analyze requirement coverage
                            coverage = self.analyze_requirement_coverage(req, policy_text)
                            
                            # Update requirement with coverage details
                            req_result = req.copy()
                            req_result.update({
                                'covered': coverage['covered'],
                                'similarity_score': coverage['similarity_score'],
                                'matching_sections': coverage['matching_sections'],
                                'requirement_type': requirement_type,
                                'article_num': article_num,
                                'processing_stats': coverage.get('stats', {})
                            })
                            
                            results.append(req_result)
                            processed_count += 1
                            
                            if coverage['covered']:
                                covered_count += 1
                                self.logger.info(f"\nFound coverage for {requirement_type} requirement in Article {article_num}:")
                                self.logger.info(f"Requirement: {req['requirement_text'][:200]}...")
                                self.logger.info(f"Similarity score: {coverage['similarity_score']:.2f}")
                                
                                if coverage['matching_sections']:
                                    self.logger.debug("Top matching sections:")
                                    for idx, section in enumerate(coverage['matching_sections'], 1):
                                        self.logger.debug(f"{idx}. Score: {section['similarity']:.2f}")
                                        self.logger.debug(f"   Text: {section['text'][:200]}...")
                            
                        except Exception as e:
                            self.logger.error(f"Error processing requirement in Article {article_num}: {str(e)}", exc_info=True)
                            error_count += 1
                            continue
                            
                        finally:
                            pbar.update(1)
            
            # Log summary statistics
            coverage_rate = (covered_count / total_requirements * 100) if total_requirements > 0 else 0
            self.logger.info(f"\nProcessing Summary for {requirement_type}:")
            self.logger.info(f"- Total requirements: {total_requirements}")
            self.logger.info(f"- Successfully processed: {processed_count}")
            self.logger.info(f"- Requirements covered: {covered_count} ({coverage_rate:.1f}%)")
            self.logger.info(f"- Processing errors: {error_count}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing {requirement_type} requirements: {str(e)}", exc_info=True)
            return []




class DORAComplianceAnalyzer:
    """DORA compliance analyzer class."""
    
    def __init__(self, dora_path: str):
        """Initialize the analyzer with DORA PDF path."""
        self.logger = logging.getLogger('DORAAnalyzer')
        self.dora_path = dora_path
        self.nlp = spacy.load("en_core_web_lg")
        self.rts_requirements = {}
        self.its_requirements = {}
        self.policy_coverage = {}
        self.cache_manager = CacheManager()
        
        # Add policy areas dictionary for classification
        self.policy_areas = {
            "authentication_security": {
                "primary_keywords": [
                    "authentication", "identity", "access control", "MFA", "password",
                    "biometric", "credentials", "login", "authorization"
                ],
                "secondary_keywords": [
                    "security", "policy", "verification", "token", "session",
                    "timeout", "account", "protection"
                ],
                "context_phrases": [
                    "multi-factor authentication", "access control system",
                    "identity management", "password requirements", "user authentication"
                ]
            },
            "data_protection": {
                "primary_keywords": [
                    "data protection", "encryption", "confidentiality", "integrity",
                    "privacy", "GDPR", "data breach", "data security"
                ],
                "secondary_keywords": [
                    "information", "storage", "processing", "data access", "retention",
                    "classification", "anonymization", "pseudonymization"
                ],
                "context_phrases": [
                    "personal data protection", "data protection impact assessment",
                    "data minimization", "data access controls", "encryption standards"
                ]
            },
            "incident_response": {
                "primary_keywords": [
                    "incident response", "breach", "incident", "SIRT", "CSIRT",
                    "detection", "containment", "eradication", "recovery"
                ],
                "secondary_keywords": [
                    "response plan", "notification", "investigation", "forensics",
                    "remediation", "lessons learned", "playbook", "disaster recovery"
                ],
                "context_phrases": [
                    "security incident response team", "incident response plan",
                    "breach notification procedure", "cyber incident management",
                    "post-incident review"
                ]
            },
            "risk_management": {
                "primary_keywords": [
                    "risk management", "risk assessment", "risk mitigation",
                    "risk register", "threat", "vulnerability", "impact"
                ],
                "secondary_keywords": [
                    "control", "likelihood", "mitigation", "acceptance", "transfer",
                    "avoidance", "monitoring", "review", "evaluation"
                ],
                "context_phrases": [
                    "enterprise risk management", "risk assessment methodology",
                    "risk treatment plan", "third-party risk management",
                    "continuous risk monitoring"
                ]
            },
            "business_continuity": {
                "primary_keywords": [
                    "business continuity", "disaster recovery", "resilience",
                    "BCP", "DRP", "contingency", "crisis management"
                ],
                "secondary_keywords": [
                    "recovery time objective", "recovery point objective", "RTO", "RPO",
                    "backup", "failover", "alternate site", "recovery strategy"
                ],
                "context_phrases": [
                    "business continuity plan", "disaster recovery testing",
                    "business impact analysis", "crisis communication plan",
                    "recovery procedures"
                ]
            },
            "supply_chain": {
                "primary_keywords": [
                    "supply chain", "vendor", "third party", "outsourcing",
                    "supplier", "contractor", "service provider"
                ],
                "secondary_keywords": [
                    "assessment", "due diligence", "contract", "SLA", "monitoring",
                    "oversight", "termination", "contingency"
                ],
                "context_phrases": [
                    "third-party risk management", "vendor assessment process",
                    "supply chain security", "vendor management program",
                    "critical service provider"
                ]
            },
            "governance": {
                "primary_keywords": [
                    "governance", "policy", "procedure", "standard", "guideline",
                    "compliance", "oversight", "accountability", "responsibility"
                ],
                "secondary_keywords": [
                    "board", "committee", "framework", "reporting", "review",
                    "audit", "management", "leadership", "structure"
                ],
                "context_phrases": [
                    "governance framework", "policy management", "compliance program",
                    "oversight committee", "management accountability"
                ]
            },
            "general": {
                "primary_keywords": [
                    "requirement", "standard", "regulation", "control", "measure",
                    "program", "system", "documentation", "report"
                ],
                "secondary_keywords": [
                    "implement", "maintain", "develop", "establish", "ensure",
                    "review", "test", "assess", "monitor"
                ],
                "context_phrases": [
                    "general requirements", "applicable standards", "regulatory compliance",
                    "documented procedures", "operational controls"
                ]
            }
        }
        
        # Compile regex patterns
        self.rts_patterns = [re.compile(pattern) for pattern in DORAConfig.RTS_PATTERNS]
        self.its_patterns = [re.compile(pattern) for pattern in DORAConfig.ITS_PATTERNS]

        # Initialize zero-shot classifier for policy area classification
        self.zero_shot_classifier = pipeline("zero-shot-classification", 
                                          model=DORAConfig.ZERO_SHOT_MODEL)
        
        # Initialize sentence transformer for similarity calculations
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            # Clear caches
            if hasattr(self, 'cache_manager'):
                self.cache_manager.clear()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}", exc_info=True)

    def extract_technical_standards(self) -> bool:
        """Extract RTS and ITS requirements from DORA text."""
        try:
            # Extract and clean text from DORA PDF
            dora_text = self._extract_and_clean_text()
            if not dora_text:
                self.logger.error("Failed to extract text from DORA PDF")
                return False
            
            # Initialize counters
            rts_count = 0
            its_count = 0
            
            # Process articles with progress bar
            with tqdm(total=len(self.articles), desc="Processing Articles", disable=DORAConfig.PROGRESS_BAR_DISABLE) as pbar:
                for article_num, article_data in self.articles.items():
                    try:
                        # Extract requirements from article
                        article_text = article_data['content']
                        article_title = article_data['title']
                        
                        # Process RTS requirements
                        for pattern in self.rts_patterns:
                            matches = pattern.finditer(article_text, re.IGNORECASE | re.MULTILINE)
                            for match in matches:
                                try:
                                    requirement_text = match.group(0).strip()
                                    cleaned_text = TextProcessor.clean_text(requirement_text)
                                    
                                    # Get context around the requirement
                                    context_start = max(0, match.start() - 200)
                                    context_end = min(len(article_text), match.end() + 200)
                                    full_context = article_text[context_start:context_end].strip()
                                    
                                    # Determine policy area using zero-shot classification
                                    policy_area = self._classify_policy_area(cleaned_text)
                                    
                                    # Create requirement dictionary
                                    requirement = {
                                        'article_num': article_num,
                                        'requirement_text': requirement_text,
                                        'full_context': full_context,
                                        'policy_area': policy_area,
                                        'metadata': {
                                            'extraction_confidence': 'high' if len(cleaned_text) > 50 else 'medium'
                                        }
                                    }
                                    
                                    # Add to RTS requirements
                                    if article_num not in self.rts_requirements:
                                        self.rts_requirements[article_num] = []
                                    self.rts_requirements[article_num].append(requirement)
                                    rts_count += 1
                                    
                                    # Log the extracted requirement
                                    self.logger.debug(f"Extracted requirement from Article {article_num}: {requirement_text[:100]}...")
                                    
                                    # Clear variables to free memory
                                    del cleaned_text
                                    del requirement_text
                                    del requirement
                                    
                                    # Check memory usage periodically
                                    if (rts_count + its_count) % 100 == 0:
                                        memory_percent = psutil.Process().memory_percent()
                                        if memory_percent > 80:  # 80% threshold
                                            self.logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
                                            gc.collect()
                                    
                                    pbar.update(1)
                                
                                except Exception as e:
                                    self.logger.error(f"Error processing RTS match: {str(e)}", exc_info=True)
                                    continue
                        
                        # Process ITS requirements
                        for pattern in self.its_patterns:
                            matches = pattern.finditer(article_text, re.IGNORECASE | re.MULTILINE)
                            for match in matches:
                                try:
                                    requirement_text = match.group(0).strip()
                                    cleaned_text = TextProcessor.clean_text(requirement_text)
                                    
                                    # Get context around the requirement
                                    context_start = max(0, match.start() - 200)
                                    context_end = min(len(article_text), match.end() + 200)
                                    full_context = article_text[context_start:context_end].strip()
                                    
                                    # Determine policy area using zero-shot classification
                                    policy_area = self._classify_policy_area(cleaned_text)
                                    
                                    # Create requirement dictionary
                                    requirement = {
                                        'article_num': article_num,
                                        'requirement_text': requirement_text,
                                        'full_context': full_context,
                                        'policy_area': policy_area,
                                        'metadata': {
                                            'extraction_confidence': 'high' if len(cleaned_text) > 50 else 'medium'
                                        }
                                    }
                                    
                                    # Add to ITS requirements
                                    if article_num not in self.its_requirements:
                                        self.its_requirements[article_num] = []
                                    self.its_requirements[article_num].append(requirement)
                                    its_count += 1
                                    
                                    # Log the extracted requirement
                                    self.logger.debug(f"Extracted requirement from Article {article_num}: {requirement_text[:100]}...")
                                    
                                    # Clear variables to free memory
                                    del cleaned_text
                                    del requirement_text
                                    del requirement
                                    
                                    # Check memory usage periodically
                                    if (rts_count + its_count) % 100 == 0:
                                        memory_percent = psutil.Process().memory_percent()
                                        if memory_percent > 80:  # 80% threshold
                                            self.logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
                                            gc.collect()
                                    
                                    pbar.update(1)
                                
                                except Exception as e:
                                    self.logger.error(f"Error processing ITS match: {str(e)}", exc_info=True)
                                    continue
                        
                        pbar.update(1)
                            
                    except Exception as e:
                        self.logger.error(f"Error processing article {article_num}: {str(e)}", exc_info=True)
                        continue
                        
            # Log final counts
            self.logger.info(f"\nExtraction complete:")
            self.logger.info(f"Total RTS requirements: {rts_count}")
            self.logger.info(f"Total ITS requirements: {its_count}")
            
            if rts_count == 0 and its_count == 0:
                self.logger.warning("No requirements were extracted. Check the regex patterns and input text.")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting technical standards: {str(e)}", exc_info=True)
            return False

    def _extract_and_clean_text(self, pdf_path: str = None) -> str:
        """Extract and clean text from a PDF file with memory monitoring and article extraction."""
        try:
            import psutil
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
            
            # Use provided path or default to instance path
            pdf_path = pdf_path or self.dora_path
            
            # Extract text from PDF
            self.logger.info(f"Extracting text from PDF: {pdf_path}")
            with pdfplumber.open(pdf_path) as pdf:
                text_chunks = []
                total_pages = len(pdf.pages)
                
                with tqdm(total=total_pages, desc="Extracting PDF Text", disable=DORAConfig.PROGRESS_BAR_DISABLE) as pbar:
                    for i, page in enumerate(pdf.pages):
                        try:
                            # Extract text from current page
                            text = page.extract_text()
                            if text:
                                text_chunks.append(text)
                            
                            # Monitor memory usage
                            current_memory = process.memory_info().rss / 1024 / 1024
                            memory_increase = current_memory - initial_memory
                            
                            # Log memory usage every 10 pages
                            if i % 10 == 0:
                                self.logger.debug(f"Memory usage after page {i+1}: {current_memory:.2f} MB (Increase: {memory_increase:.2f} MB)")
                            
                            # Trigger garbage collection if memory usage increases significantly
                            if memory_increase > DORAConfig.MEMORY_THRESHOLD_MB:
                                self.logger.warning(f"Memory threshold exceeded ({memory_increase:.2f} MB). Triggering garbage collection...")
                                gc.collect()
                                current_memory = process.memory_info().rss / 1024 / 1024
                                self.logger.info(f"Memory usage after GC: {current_memory:.2f} MB")
                                initial_memory = current_memory  # Reset baseline
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            self.logger.error(f"Error extracting text from page {i+1}: {str(e)}", exc_info=True)
                            continue
                
                # Combine text chunks
                text = "\n".join(text_chunks)
                
                # Initialize articles dictionary
                self.articles = {}
                
                # Extract articles using the configured pattern
                article_matches = re.finditer(DORAConfig.ARTICLE_PATTERN, text, re.MULTILINE | re.DOTALL)
                
                for match in article_matches:
                    article_num = match.group(1)  # Article number
                    article_title = match.group(2)  # Article title
                    article_content = match.group(3)  # Article content
                    
                    # Clean and normalize the content
                    cleaned_content = TextProcessor.clean_text(article_content)
                    
                    # Store article with both number and title
                    self.articles[article_num] = {
                        'title': article_title.strip(),
                        'content': cleaned_content,
                        'raw_content': article_content  # Keep raw content for context
                    }
                
                self.logger.info(f"Extracted {len(self.articles)} articles from DORA text")
                
                # Log article extraction details
                for article_num, article_data in self.articles.items():
                    self.logger.debug(f"Article {article_num}: {article_data['title'][:100]}...")
                
                # Clean the text
                text = self._remove_table_content_from_text(text, [])  # No tables extracted yet
                text = TextProcessor.clean_text(text)
                
                # Final memory cleanup
                text_chunks.clear()
                gc.collect()
                final_memory = process.memory_info().rss / 1024 / 1024
                self.logger.info(f"Final memory usage: {final_memory:.2f} MB")
                
                return text
                
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}", exc_info=True)
            self.articles = {}  # Initialize empty articles on error
            return ""

    def _process_completed_table(
        self,
        table: List[List[str]],
        tables_data: List[Dict],
        start_page: int = None,
        end_page: int = None,
    ) -> None:
        """Process a completed table and add it to tables_data."""
        try:
            # Validate table
            if not table or not isinstance(table, list):
                return

            # Remove empty rows and clean cell content
            cleaned_table = []
            for row in table:
                if not row:
                    continue
                cleaned_row = [TextProcessor.clean_cell_content(cell) for cell in row]
                if any(cleaned_row):  # Only keep rows with at least one non-empty cell
                    cleaned_table.append(cleaned_row)

            if not cleaned_table:
                return

            # Create DataFrame
            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])

            # Store table data with metadata
            tables_data.append({
                "data": df,
                "header": cleaned_table[0],
                "num_rows": len(df),
                "num_cols": len(df.columns),
                "start_page": start_page,
                "end_page": end_page,
            })

        except Exception as e:
            self.logger.error(f"Error processing table: {str(e)}")

    def _remove_table_content_from_text(self, text: str, tables: List[List[List[str]]]) -> str:
        """Remove table content from extracted text to avoid duplication."""
        if not tables:
            return text

        # Create a set of table content for faster lookup
        table_content = set()
        for table in tables:
            for row in table:
                for cell in row:
                    if isinstance(cell, str):
                        # Add both exact content and normalized version
                        cell_text = TextProcessor.clean_cell_content(cell)
                        table_content.add(cell_text)
                        table_content.add(" ".join(cell_text.split()))

        # Split text into lines and process each
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue

            # Check if line contains table content
            should_keep = True
            normalized_line = " ".join(line.split())

            for content in table_content:
                if content in line or content in normalized_line:
                    should_keep = False
                    break

            if should_keep:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a given PDF file and clean it."""
        try:
            extracted_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    cleaned_text = TextProcessor.clean_text(page_text)
                    extracted_text += cleaned_text + "\n\n"
            return extracted_text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    @log_execution_time
    def analyze_policy_document(self, policy_name: str, policy_text: str, max_retries: int = 3) -> Dict:
        """Analyze a policy document for RTS and ITS coverage with progress tracking and retries."""
        retry_count = 0
        last_error = None
        backoff_time = 1  # Initial backoff time in seconds
        
        while retry_count < max_retries:
            try:
                self.logger.info(f"\nAnalyzing policy document: {policy_name} (Attempt {retry_count + 1}/{max_retries})")
                
                # Input validation
                if not policy_name or not isinstance(policy_name, str):
                    raise ValueError("Invalid policy name provided")
                if not policy_text or not isinstance(policy_text, str):
                    raise ValueError("Invalid policy text provided")
                if len(policy_text.strip()) < 100:  # Arbitrary minimum length
                    raise ValueError("Policy text too short - might be corrupted or empty")
                
                # Initialize the requirement analyzer
                requirement_analyzer = RequirementAnalyzer(self.nlp)
                
                # Process RTS requirements with progress bar
                self.logger.info("Processing RTS requirements...")
                total_rts = sum(len(reqs) for reqs in self.rts_requirements.values())
                rts_results = []
                rts_errors = []
                
                with tqdm(total=total_rts, desc="RTS Analysis", disable=DORAConfig.PROGRESS_BAR_DISABLE) as pbar:
                    for article_reqs in self.rts_requirements.values():
                        try:
                            results = requirement_analyzer.process_requirements(article_reqs, policy_text, "RTS")
                            rts_results.extend(results)
                            pbar.update(len(article_reqs))
                        except Exception as e:
                            error_msg = f"Error processing RTS requirements: {str(e)}"
                            self.logger.error(error_msg, exc_info=True)
                            rts_errors.append(error_msg)
                            if retry_count == max_retries - 1:
                                raise
                
                # Process ITS requirements with progress bar
                self.logger.info("Processing ITS requirements...")
                total_its = sum(len(reqs) for reqs in self.its_requirements.values())
                its_results = []
                its_errors = []
                
                with tqdm(total=total_its, desc="ITS Analysis", disable=DORAConfig.PROGRESS_BAR_DISABLE) as pbar:
                    for article_reqs in self.its_requirements.values():
                        try:
                            results = requirement_analyzer.process_requirements(article_reqs, policy_text, "ITS")
                            its_results.extend(results)
                            pbar.update(len(article_reqs))
                        except Exception as e:
                            error_msg = f"Error processing ITS requirements: {str(e)}"
                            self.logger.error(error_msg, exc_info=True)
                            its_errors.append(error_msg)
                            if retry_count == max_retries - 1:
                                raise
                
                # Validate results
                if not rts_results and not its_results:
                    raise ValueError("No results generated for either RTS or ITS requirements")
                
                # Update global coverage tracking
                self._update_global_coverage(policy_name, rts_results, its_results)
                
                # Store per-policy results
                try:
                    self.policy_coverage[policy_name] = {
                        'rts_results': rts_results,
                        'its_results': its_results,
                        'rts_errors': rts_errors,
                        'its_errors': its_errors,
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                except Exception as e:
                    self.logger.error(f"Error storing policy coverage: {str(e)}", exc_info=True)
                    if retry_count == max_retries - 1:
                        raise
                
                # Calculate and log coverage statistics
                try:
                    rts_covered = sum(1 for result in rts_results if result['covered'])
                    its_covered = sum(1 for result in its_results if result['covered'])
                    
                    coverage_stats = {
                        'total_rts': total_rts,
                        'covered_rts': rts_covered,
                        'rts_coverage': (rts_covered / total_rts * 100) if total_rts > 0 else 0,
                        'total_its': total_its,
                        'covered_its': its_covered,
                        'its_coverage': (its_covered / total_its * 100) if total_its > 0 else 0,
                        'rts_errors': len(rts_errors),
                        'its_errors': len(its_errors),
                        'analysis_complete': True,
                        'global_coverage': self._calculate_global_coverage()
                    }
                    
                    self.logger.info("\nCoverage Statistics:")
                    self.logger.info(f"Policy Coverage:")
                    self.logger.info(f"- RTS Coverage: {coverage_stats['rts_coverage']:.2f}% ({rts_covered}/{total_rts})")
                    self.logger.info(f"- ITS Coverage: {coverage_stats['its_coverage']:.2f}% ({its_covered}/{total_its})")
                    
                    self.logger.info("\nGlobal Coverage (across all policies):")
                    self.logger.info(f"- RTS Coverage: {coverage_stats['global_coverage']['rts_coverage']:.2f}%")
                    self.logger.info(f"- ITS Coverage: {coverage_stats['global_coverage']['its_coverage']:.2f}%")
                    self.logger.info(f"- Overall Coverage: {coverage_stats['global_coverage']['overall_coverage']:.2f}%")
                    
                    if rts_errors or its_errors:
                        self.logger.warning(f"Analysis completed with errors - RTS: {len(rts_errors)}, ITS: {len(its_errors)}")
                    
                    return coverage_stats
                    
                except Exception as e:
                    self.logger.error(f"Error calculating coverage statistics: {str(e)}", exc_info=True)
                    if retry_count == max_retries - 1:
                        raise
                    return {
                        'total_rts': 0,
                        'covered_rts': 0,
                        'rts_coverage': 0,
                        'total_its': 0,
                        'covered_its': 0,
                        'its_coverage': 0,
                        'analysis_complete': False,
                        'error': str(e)
                    }
                
            except Exception as e:
                retry_count += 1
                last_error = str(e)
                if retry_count < max_retries:
                    self.logger.warning(f"Attempt {retry_count} failed, retrying in {backoff_time}s... Error: {str(e)}")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                else:
                    self.logger.error(f"All retry attempts failed for {policy_name}: {str(e)}", exc_info=True)
                    return {
                        'total_rts': 0,
                        'covered_rts': 0,
                        'rts_coverage': 0,
                        'total_its': 0,
                        'covered_its': 0,
                        'its_coverage': 0,
                        'analysis_complete': False,
                        'error': last_error
                    }

    def _update_global_coverage(self, policy_name: str, rts_results: List[Dict], its_results: List[Dict]) -> None:
        """Update global coverage tracking across all policies."""
        try:
            # Initialize global coverage tracking if not exists
            if not hasattr(self, 'global_coverage'):
                self.global_coverage = {
                    'rts': defaultdict(lambda: {'policies': [], 'similarity_scores': {}}),
                    'its': defaultdict(lambda: {'policies': [], 'similarity_scores': {}})
                }
            
            # Initialize policy set if not exists
            if not hasattr(self, 'policy_set'):
                self.policy_set = set()
            self.policy_set.add(policy_name)
            
            # Update RTS coverage
            for result in rts_results:
                if result.get('covered', False):
                    req_id = f"{result['article_num']}_{result['requirement_text'][:50]}"
                    if policy_name not in self.global_coverage['rts'][req_id]['policies']:
                        self.global_coverage['rts'][req_id]['policies'].append(policy_name)
                        self.global_coverage['rts'][req_id]['similarity_scores'][policy_name] = result.get('similarity_score', 0.0)
            
            # Update ITS coverage
            for result in its_results:
                if result.get('covered', False):
                    req_id = f"{result['article_num']}_{result['requirement_text'][:50]}"
                    if policy_name not in self.global_coverage['its'][req_id]['policies']:
                        self.global_coverage['its'][req_id]['policies'].append(policy_name)
                        self.global_coverage['its'][req_id]['similarity_scores'][policy_name] = result.get('similarity_score', 0.0)
                        
        except Exception as e:
            self.logger.error(f"Error updating global coverage: {str(e)}", exc_info=True)

    def _calculate_global_coverage(self, coverage_type: str = 'union') -> Dict:
        """Calculate global coverage statistics across all policies."""
        try:
            if not hasattr(self, 'global_coverage'):
                return {
                    'rts_coverage': 0.0,
                    'its_coverage': 0.0,
                    'overall_coverage': 0.0
                }
            
            # Count total requirements
            total_rts = sum(len(reqs) for reqs in self.rts_requirements.values())
            total_its = sum(len(reqs) for reqs in self.its_requirements.values())
            
            # Count covered requirements based on coverage type
            if coverage_type == 'intersection':
                # Requirement is covered only if all policies cover it
                covered_rts = sum(
                    1 for req_data in self.global_coverage['rts'].values()
                    if len(req_data['policies']) == len(self.policy_set)
                )
                covered_its = sum(
                    1 for req_data in self.global_coverage['its'].values()
                    if len(req_data['policies']) == len(self.policy_set)
                )
            else:  # union coverage
                # Requirement is covered if at least one policy covers it
                covered_rts = len(self.global_coverage['rts'])
                covered_its = len(self.global_coverage['its'])
            
            # Calculate coverage percentages
            rts_coverage = (covered_rts / total_rts * 100) if total_rts > 0 else 0
            its_coverage = (covered_its / total_its * 100) if total_its > 0 else 0
            overall_coverage = (
                (covered_rts + covered_its) / (total_rts + total_its) * 100
                if (total_rts + total_its) > 0 else 0
            )
            
            # Calculate per-policy coverage statistics
            policy_stats = {}
            for policy_name in self.policy_set:
                rts_by_policy = sum(
                    1 for req_data in self.global_coverage['rts'].values()
                    if policy_name in req_data['policies']
                )
                its_by_policy = sum(
                    1 for req_data in self.global_coverage['its'].values()
                    if policy_name in req_data['policies']
                )
                policy_stats[policy_name] = {
                    'rts_coverage': (rts_by_policy / total_rts * 100) if total_rts > 0 else 0,
                    'its_coverage': (its_by_policy / total_its * 100) if total_its > 0 else 0,
                    'overall_coverage': (
                        (rts_by_policy + its_by_policy) / (total_rts + total_its) * 100
                        if (total_rts + total_its) > 0 else 0
                    )
                }
            
            # Calculate coverage overlap statistics
            overlap_stats = self._calculate_coverage_overlap()
            
            return {
                'rts_coverage': rts_coverage,
                'its_coverage': its_coverage,
                'overall_coverage': overall_coverage,
                'covered_rts': covered_rts,
                'covered_its': covered_its,
                'total_rts': total_rts,
                'total_its': total_its,
                'coverage_type': coverage_type,
                'policy_stats': policy_stats,
                'overlap_stats': overlap_stats,
                'coverage_details': {
                    'rts': dict(self.global_coverage['rts']),
                    'its': dict(self.global_coverage['its'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating global coverage: {str(e)}", exc_info=True)
            return {
                'rts_coverage': 0.0,
                'its_coverage': 0.0,
                'overall_coverage': 0.0
            }

    def _calculate_coverage_overlap(self) -> Dict:
        """Calculate coverage overlap statistics between policies."""
        try:
            overlap_stats = {
                'rts': defaultdict(int),  # Number of requirements covered by N policies
                'its': defaultdict(int),
                'policy_pairs': defaultdict(lambda: {'rts': 0, 'its': 0})  # Overlap between policy pairs
            }
            
            # Calculate requirements covered by N policies
            for req_data in self.global_coverage['rts'].values():
                overlap_stats['rts'][len(req_data['policies'])] += 1
            
            for req_data in self.global_coverage['its'].values():
                overlap_stats['its'][len(req_data['policies'])] += 1
            
            # Calculate policy pair overlaps
            policies = list(self.policy_set)
            for i, policy1 in enumerate(policies):
                for policy2 in policies[i+1:]:
                    pair_key = f"{policy1}_{policy2}"
                    
                    # RTS overlap
                    rts_overlap = sum(1 for req_data in self.global_coverage['rts'].values()
                                    if policy1 in req_data['policies'] and policy2 in req_data['policies'])
                    overlap_stats['policy_pairs'][pair_key]['rts'] = rts_overlap
                    
                    # ITS overlap
                    its_overlap = sum(1 for req_data in self.global_coverage['its'].values()
                                    if policy1 in req_data['policies'] and policy2 in req_data['policies'])
                    overlap_stats['policy_pairs'][pair_key]['its'] = its_overlap
            
            return dict(overlap_stats)
            
        except Exception as e:
            self.logger.error(f"Error calculating coverage overlap: {str(e)}", exc_info=True)
            return {}

    def _calculate_similarity(self, req_text: str, policy_text: str) -> float:
        """Calculate similarity between requirement and policy text using a hybrid approach with caching.
        
        Similarity threshold tuning guidance:
        - Increase DORAConfig.STRONG_MATCH_THRESHOLD (default: 0.6) for stricter matching
        - Decrease DORAConfig.COVERAGE_MATCH_THRESHOLD (default: 0.7) for more lenient section matching
        - Decrease DORAConfig.FINAL_COVERAGE_THRESHOLD (default: 0.8) for higher overall coverage
        
        Different document types may require different thresholds:
        - Technical documents: May need lower thresholds (0.5-0.7) due to technical jargon
        - Policy documents: Standard thresholds (0.6-0.8) work well
        - Legal documents: May need higher thresholds (0.7-0.9) for precise matching
        
        To find optimal thresholds:
        1. Start with defaults
        2. Run analysis on sample documents with known coverage
        3. Examine false positives and false negatives
        4. Adjust thresholds and re-run until optimal results achieved
        
        Args:
            req_text (str): The requirement text to match
            policy_text (str): The policy text to check for coverage
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Generate cache key using hash
            cache_key = hashlib.sha256(f"{req_text}_{policy_text}".encode()).hexdigest()
            
            # Check cache
            if hasattr(self, 'similarity_cache') and cache_key in self.similarity_cache:
                return self.similarity_cache[cache_key]
            
            # Initialize cache if not exists
            if not hasattr(self, 'similarity_cache'):
                self.similarity_cache = {}
            
            # First pass: Fast SpaCy similarity
            spacy_similarity = self.nlp(req_text).similarity(self.nlp(policy_text))
            
            # If similarity is in the borderline range, do a more thorough check
            if 0.5 <= spacy_similarity <= 0.8:
                # Check transformer cache
                if not hasattr(self, 'embedding_cache'):
                    self.embedding_cache = {}
                
                # Get or compute requirement embedding
                req_cache_key = hashlib.sha256(req_text.encode()).hexdigest()
                if req_cache_key in self.embedding_cache:
                    req_embedding = self.embedding_cache[req_cache_key]
                else:
                    req_embedding = self.sentence_transformer.encode(req_text, convert_to_tensor=True)
                    self.embedding_cache[req_cache_key] = req_embedding
                
                # Get or compute policy embedding
                policy_cache_key = hashlib.sha256(policy_text.encode()).hexdigest()
                if policy_cache_key in self.embedding_cache:
                    policy_embedding = self.embedding_cache[policy_cache_key]
                else:
                    policy_embedding = self.sentence_transformer.encode(policy_text, convert_to_tensor=True)
                    self.embedding_cache[policy_cache_key] = policy_embedding
                
                # Calculate cosine similarity using torch.nn.functional
                transformer_similarity = F.cosine_similarity(
                    req_embedding.unsqueeze(0),
                    policy_embedding.unsqueeze(0)
                ).item()
                
                # Weight both similarities (favor transformer for accuracy)
                final_similarity = (spacy_similarity * 0.3) + (transformer_similarity * 0.7)
                
                self.logger.debug(f"Hybrid similarity: SpaCy={spacy_similarity:.3f}, "
                                f"Transformer={transformer_similarity:.3f}, Final={final_similarity:.3f}")
                
                # Cache the result
                self.similarity_cache[cache_key] = final_similarity
                return final_similarity
            
            # Cache and return SpaCy similarity for non-borderline cases
            self.similarity_cache[cache_key] = spacy_similarity
            return spacy_similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}", exc_info=True)
            return 0.0

    def _cleanup_caches(self):
        """Clean up embedding and similarity caches to prevent memory bloat."""
        try:
            # Clear embedding cache if it exists
            if hasattr(self, 'embedding_cache'):
                self.embedding_cache.clear()
                del self.embedding_cache
            
            # Clear similarity cache if it exists
            if hasattr(self, 'similarity_cache'):
                self.similarity_cache.clear()
                del self.similarity_cache
            
            self.logger.info("Cleared embedding and similarity caches")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up caches: {str(e)}", exc_info=True)

    @log_execution_time
    def generate_gap_analysis_report(self) -> str:
        """Generate a comprehensive gap analysis report with progress tracking."""
        try:
            self.logger.info("Generating gap analysis report...")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.getcwd(), DORAConfig.OUTPUT_FOLDER_NAME)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp for the report
            timestamp = datetime.now().strftime(DORAConfig.TIMESTAMP_FORMAT)
            report_filename = DORAConfig.REPORT_FILENAME_TEMPLATE.format(timestamp=timestamp)
            report_path = os.path.join(output_dir, report_filename)
            
            # Initialize report sections
            sections = []
            sections.append("# DORA Compliance Gap Analysis Report")
            sections.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Executive Summary
            self.logger.info("Generating executive summary...")
            sections.append("## Executive Summary")
            total_requirements = sum(len(reqs) for reqs in self.rts_requirements.values()) + \
                               sum(len(reqs) for reqs in self.its_requirements.values())
            
            total_covered = 0
            policy_coverage_stats = {}
            policy_focus_areas = {}
            
            # Process policy coverage statistics and determine policy focus areas
            for policy_name, coverage in self.policy_coverage.items():
                # Extract policy text from first result if available
                policy_text = ""
                if coverage['rts_results']:
                    policy_text = coverage['rts_results'][0].get('full_context', '')
                elif coverage['its_results']:
                    policy_text = coverage['its_results'][0].get('full_context', '')
                
                # Determine policy focus areas
                focus_areas = self._determine_policy_focus_areas(policy_name, policy_text)
                policy_focus_areas[policy_name] = focus_areas
                
                rts_covered = sum(1 for r in coverage['rts_results'] if r.get('covered', False))
                its_covered = sum(1 for r in coverage['its_results'] if r.get('covered', False))
                total_covered = max(total_covered, rts_covered + its_covered)
                
                policy_coverage_stats[policy_name] = {
                    'rts_covered': rts_covered,
                    'its_covered': its_covered,
                    'total_covered': rts_covered + its_covered,
                    'rts_total': len(coverage['rts_results']),
                    'its_total': len(coverage['its_results']),
                    'focus_areas': focus_areas
                }
            
            # Calculate overall coverage
            overall_coverage = (total_covered / total_requirements * 100) if total_requirements > 0 else 0
            
            # Add summary statistics to report
            sections.append(f"Overall compliance coverage: {overall_coverage:.2f}%")
            sections.append(f"Total requirements analyzed: {total_requirements}")
            sections.append(f"Requirements covered: {total_covered}")
            sections.append(f"Requirements with gaps: {total_requirements - total_covered}\n")
            
            # Policy Coverage Summary
            sections.append("## Policy Coverage Summary")
            
            # Process each policy's coverage statistics
            for policy_name, stats in policy_coverage_stats.items():
                sections.append(f"### {policy_name}")
                
                # Add policy focus areas
                focus_areas_display = [area.replace('_', ' ').title() for area in stats['focus_areas']]
                sections.append(f"**Policy Focus Areas**: {', '.join(focus_areas_display)}")
                
                # Calculate coverage percentages
                rts_coverage = (stats['rts_covered']/stats['rts_total']*100) if stats['rts_total'] > 0 else 0
                its_coverage = (stats['its_covered']/stats['its_total']*100) if stats['its_total'] > 0 else 0
                total_coverage = (stats['total_covered']/(stats['rts_total'] + stats['its_total'])*100) \
                    if (stats['rts_total'] + stats['its_total']) > 0 else 0
                
                # Add coverage details to report
                sections.append(f"- RTS Requirements Covered: {stats['rts_covered']}/{stats['rts_total']} ({rts_coverage:.2f}%)")
                sections.append(f"- ITS Requirements Covered: {stats['its_covered']}/{stats['its_total']} ({its_coverage:.2f}%)")
                sections.append(f"- Total Requirements Covered: {stats['total_covered']}/{stats['rts_total'] + stats['its_total']} ({total_coverage:.2f}%)\n")
            
            # Generate detailed gap analysis by article
            sections.append("## Detailed Gap Analysis")
            sections.append("\nThis section identifies specific DORA articles and requirements that are not adequately covered by existing policies.\n")
            
            # Collect all gaps across policies
            rts_gaps = {}
            its_gaps = {}
            
            for policy_name, coverage in self.policy_coverage.items():
                # Get policy focus areas
                focus_areas = policy_focus_areas.get(policy_name, ['general'])
                
                # Process RTS gaps
                for result in coverage.get('rts_results', []):
                    if not result.get('covered', False):
                        article_num = result.get('article_num', 'Unknown')
                        requirement_area = result.get('policy_area', 'general')
                        
                        # Determine if this gap is relevant to this policy's focus
                        is_relevant_to_policy = requirement_area in focus_areas or 'general' in focus_areas
                        
                        if article_num not in rts_gaps:
                            rts_gaps[article_num] = []
                        
                        # Add gap details
                        rts_gaps[article_num].append({
                            'requirement_text': result.get('requirement_text', 'Unknown requirement'),
                            'similarity_score': result.get('similarity_score', 0.0),
                            'policy_name': policy_name,
                            'policy_area': requirement_area,
                            'is_relevant_to_policy': is_relevant_to_policy,
                            'policy_focus_areas': focus_areas
                        })
                
                # Process ITS gaps
                for result in coverage.get('its_results', []):
                    if not result.get('covered', False):
                        article_num = result.get('article_num', 'Unknown')
                        requirement_area = result.get('policy_area', 'general')
                        
                        # Determine if this gap is relevant to this policy's focus
                        is_relevant_to_policy = requirement_area in focus_areas or 'general' in focus_areas
                        
                        if article_num not in its_gaps:
                            its_gaps[article_num] = []
                        
                        # Add gap details
                        its_gaps[article_num].append({
                            'requirement_text': result.get('requirement_text', 'Unknown requirement'),
                            'similarity_score': result.get('similarity_score', 0.0),
                            'policy_name': policy_name,
                            'policy_area': requirement_area,
                            'is_relevant_to_policy': is_relevant_to_policy,
                            'policy_focus_areas': focus_areas
                        })
            
            # Report RTS gaps by article
            if rts_gaps:
                sections.append("### Regulatory Technical Standards (RTS) Gaps")
                for article_num in sorted(rts_gaps.keys()):
                    sections.append(f"\n#### Article {article_num}")
                    
                    # Separate relevant and non-relevant gaps
                    relevant_gaps = [g for g in rts_gaps[article_num] if g['is_relevant_to_policy']]
                    non_relevant_gaps = [g for g in rts_gaps[article_num] if not g['is_relevant_to_policy']]
                    
                    # Process relevant gaps first
                    if relevant_gaps:
                        sections.append("\n**Relevant Gaps (missing in policies that should cover this):**")
                        for i, gap in enumerate(relevant_gaps, 1):
                            sections.append(f"\n**Gap {i}:**")
                            sections.append(f"- **Requirement:** {gap['requirement_text']}")
                            sections.append(f"- **Policy:** {gap['policy_name']}")
                            sections.append(f"- **Policy Focus Areas:** {', '.join([a.replace('_', ' ').title() for a in gap['policy_focus_areas']])}")
                            sections.append(f"- **Requirement Area:** {gap['policy_area'].replace('_', ' ').title()}")
                            sections.append(f"- **Best Match Score:** {gap['similarity_score']:.2f}")
                            
                            # Add gap reason based on similarity score
                            if gap['similarity_score'] < 0.3:
                                reason = "No significant coverage found despite being within policy scope"
                            elif gap['similarity_score'] < 0.5:
                                reason = "Minimal coverage found, but falls well below the threshold for compliance"
                            else:
                                reason = "Partial coverage found, but insufficient to meet compliance requirements"
                            
                            sections.append(f"- **Gap Reason:** {reason}")
                            
                            # Add recommendation
                            if gap['similarity_score'] < 0.3:
                                recommendation = f"Add a section to {gap['policy_name']} specifically addressing this requirement"
                            elif gap['similarity_score'] < 0.5:
                                recommendation = f"Significantly enhance {gap['policy_name']} to better address this requirement"
                            else:
                                recommendation = f"Update {gap['policy_name']} to more explicitly address this requirement"
                            
                            sections.append(f"- **Recommendation:** {recommendation}")
                    
                    # Process non-relevant gaps (for information only)
                    if non_relevant_gaps:
                        sections.append("\n**Informational Gaps (outside policy scope, covered elsewhere):**")
                        for i, gap in enumerate(non_relevant_gaps, 1):
                            sections.append(f"\n**Info {i}:**")
                            sections.append(f"- **Requirement:** {gap['requirement_text']}")
                            sections.append(f"- **Policy:** {gap['policy_name']}")
                            sections.append(f"- **Policy Focus Areas:** {', '.join([a.replace('_', ' ').title() for a in gap['policy_focus_areas']])}")
                            sections.append(f"- **Requirement Area:** {gap['policy_area'].replace('_', ' ').title()}")
                            sections.append(f"- **Note:** This requirement is outside the scope of this policy and should be covered by a dedicated {gap['policy_area'].replace('_', ' ')} policy")
            else:
                sections.append("### No RTS Gaps Identified\n")
            
            # Report ITS gaps by article
            if its_gaps:
                sections.append("\n### Implementing Technical Standards (ITS) Gaps")
                for article_num in sorted(its_gaps.keys()):
                    sections.append(f"\n#### Article {article_num}")
                    
                    # Separate relevant and non-relevant gaps
                    relevant_gaps = [g for g in its_gaps[article_num] if g['is_relevant_to_policy']]
                    non_relevant_gaps = [g for g in its_gaps[article_num] if not g['is_relevant_to_policy']]
                    
                    # Process relevant gaps first
                    if relevant_gaps:
                        sections.append("\n**Relevant Gaps (missing in policies that should cover this):**")
                        for i, gap in enumerate(relevant_gaps, 1):
                            sections.append(f"\n**Gap {i}:**")
                            sections.append(f"- **Requirement:** {gap['requirement_text']}")
                            sections.append(f"- **Policy:** {gap['policy_name']}")
                            sections.append(f"- **Policy Focus Areas:** {', '.join([a.replace('_', ' ').title() for a in gap['policy_focus_areas']])}")
                            sections.append(f"- **Requirement Area:** {gap['policy_area'].replace('_', ' ').title()}")
                            sections.append(f"- **Best Match Score:** {gap['similarity_score']:.2f}")
                            
                            # Add gap reason based on similarity score
                            if gap['similarity_score'] < 0.3:
                                reason = "No implementation procedures found despite being within policy scope"
                            elif gap['similarity_score'] < 0.5:
                                reason = "Some implementation details found, but falls well below the threshold for compliance"
                            else:
                                reason = "Partial implementation details found, but insufficient to meet compliance requirements"
                            
                            sections.append(f"- **Gap Reason:** {reason}")
                            
                            # Add recommendation
                            if gap['similarity_score'] < 0.3:
                                recommendation = f"Add implementation procedures to {gap['policy_name']} for this requirement"
                            elif gap['similarity_score'] < 0.5:
                                recommendation = f"Enhance implementation details in {gap['policy_name']} for this requirement"
                            else:
                                recommendation = f"Expand existing implementation procedures in {gap['policy_name']} for this requirement"
                            
                            sections.append(f"- **Recommendation:** {recommendation}")
                    
                    # Process non-relevant gaps (for information only)
                    if non_relevant_gaps:
                        sections.append("\n**Informational Gaps (outside policy scope, covered elsewhere):**")
                        for i, gap in enumerate(non_relevant_gaps, 1):
                            sections.append(f"\n**Info {i}:**")
                            sections.append(f"- **Requirement:** {gap['requirement_text']}")
                            sections.append(f"- **Policy:** {gap['policy_name']}")
                            sections.append(f"- **Policy Focus Areas:** {', '.join([a.replace('_', ' ').title() for a in gap['policy_focus_areas']])}")
                            sections.append(f"- **Requirement Area:** {gap['policy_area'].replace('_', ' ').title()}")
                            sections.append(f"- **Note:** This requirement is outside the scope of this policy and should be covered by a dedicated {gap['policy_area'].replace('_', ' ')} policy")
            else:
                sections.append("### No ITS Gaps Identified\n")
            
            # Add recommendation summary
            sections.append("\n## Recommendations Summary")
            sections.append("\nBased on the identified gaps, the following high-level recommendations are provided:\n")
            
            # Generate high-level recommendations
            if overall_coverage < 50:
                sections.append("1. **Critical Action Required**: Develop comprehensive DORA compliance policies and procedures")
                sections.append("2. Prioritize addressing RTS requirements with zero or minimal coverage")
                sections.append("3. Establish a DORA compliance program with regular assessments")
            elif overall_coverage < 75:
                sections.append("1. **Significant Enhancements Needed**: Update existing policies to address identified gaps")
                sections.append("2. Develop implementation procedures for ITS requirements")
                sections.append("3. Conduct regular gap assessments as DORA requirements evolve")
            else:
                sections.append("1. **Refinement Recommended**: Fine-tune policies to address specific gaps")
                sections.append("2. Ensure implementation procedures are clearly documented")
                sections.append("3. Maintain ongoing compliance monitoring")
            
            # Add next steps
            sections.append("\n### Next Steps")
            sections.append("\nTo address the identified gaps, consider the following steps:\n")
            sections.append("1. Prioritize gaps based on risk and implementation effort")
            sections.append("2. Develop an action plan with clear ownership and timelines")
            sections.append("3. Consider engaging compliance specialists for complex requirements")
            sections.append("4. Re-assess compliance after implementing changes")
            sections.append("5. Establish a monitoring process to maintain compliance as regulations evolve")
            
            # Write report to file
            self.logger.info(f"Writing report to {report_path}...")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(sections))
            
            self.logger.info("Gap analysis report generated successfully.")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating gap analysis report: {str(e)}", exc_info=True)
            return "Error generating report"

    def _determine_policy_focus_areas(self, policy_name: str, policy_text: str) -> List[str]:
        """
        Determine the primary focus areas of a policy based on its title and content.
        
        Args:
            policy_name: Name of the policy document
            policy_text: Text content of the policy
            
        Returns:
            List of policy areas that are the focus of this document
        """
        focus_areas = []
        policy_name_lower = policy_name.lower()
        
        # Map of keywords to policy areas
        focus_area_keywords = {
            'authentication_security': ['authentication', 'access control', 'identity', 'password', 'credential', 'login', 'biometric'],
            'data_protection': ['data protection', 'data security', 'confidentiality', 'privacy', 'gdpr', 'data breach', 'encryption'],
            'incident_response': ['incident', 'breach', 'response', 'detection', 'sirt', 'csirt', 'security event'],
            'risk_management': ['risk', 'threat', 'vulnerability', 'assessment', 'control', 'mitigation', 'management'],
            'business_continuity': ['continuity', 'recovery', 'disaster', 'resilience', 'bcp', 'drp', 'backup', 'restore'],
            'supply_chain': ['vendor', 'third party', 'supplier', 'outsource', 'service provider', 'contractor', 'third-party'],
            'governance': ['governance', 'compliance', 'policy', 'procedure', 'standard', 'oversight', 'accountability']
        }
        
        # Check title for obvious focus areas
        for area, keywords in focus_area_keywords.items():
            if any(keyword in policy_name_lower for keyword in keywords):
                focus_areas.append(area)
        
        # If no clear areas from title, analyze content
        if not focus_areas:
            # Count keyword occurrences in text
            area_scores = {area: 0 for area in focus_area_keywords.keys()}
            
            for area, keywords in focus_area_keywords.items():
                for keyword in keywords:
                    area_scores[area] += policy_text.lower().count(keyword)
            
            # Find the top 2 areas by keyword occurrence
            top_areas = sorted(area_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # Only include areas with significant scores
            for area, score in top_areas:
                if score > 5:  # Threshold for significance
                    focus_areas.append(area)
        
        # Always include 'general' for broadly applicable policies
        if not focus_areas or any(term in policy_name_lower for term in ['general', 'framework', 'program', 'overall']):
            focus_areas.append('general')
        
        return focus_areas

    def _identify_policy_area(self, text: str) -> str:
        """Identify the policy area using zero-shot classification and keyword analysis."""
        try:
            self.logger.info("Analyzing policy area for text snippet...")
            text = text.lower()
            
            # Process text in batches
            BATCH_SIZE = DORAConfig.MAX_TEXT_CHUNK_SIZE
            text_batches = [text[i:i+BATCH_SIZE] for i in range(0, len(text), BATCH_SIZE)]
            
            # Initialize area scores
            area_scores = defaultdict(float)
            
            # Prepare candidate labels for zero-shot classification
            candidate_labels = list(self.policy_areas.keys())
            candidate_labels = [label.replace('_', ' ').title() for label in candidate_labels]
            
            # Process each batch with zero-shot classification
            for batch_idx, batch_text in enumerate(text_batches):
                self.logger.debug(f"Processing batch {batch_idx + 1}/{len(text_batches)}")
                
                try:
                    # Get zero-shot predictions
                    result = self.zero_shot_classifier(
                        batch_text,
                        candidate_labels=candidate_labels,
                        multi_label=True
                    )
                    
                    # Process predictions
                    for label, score in zip(result['labels'], result['scores']):
                        if score > DORAConfig.LLM_THRESHOLD:
                            area = label.lower().replace(' ', '_')
                            area_scores[area] += score
                            self.logger.debug(f"Found match for area '{area}' with score {score:.3f}")
                
                except Exception as e:
                    self.logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
                    continue
            
            # Get the best matching area from zero-shot classification
            if area_scores:
                best_area = max(area_scores.items(), key=lambda x: x[1])
                confidence = best_area[1] / len(text_batches)  # Normalize by number of batches
                
                if confidence > DORAConfig.STRONG_MATCH_THRESHOLD:
                    self.logger.info(f"Selected policy area via zero-shot: {best_area[0]} (confidence: {confidence:.3f})")
                    return best_area[0]
            
            # Fallback to keyword-based analysis if zero-shot confidence is low
            self.logger.warning("Zero-shot confidence too low, falling back to keyword matching")
            return self._identify_policy_area_by_keywords(text)
            
        except Exception as e:
            self.logger.error(f"Error in policy area identification: {str(e)}", exc_info=True)
            return "general"

    def _identify_policy_area_by_keywords(self, text: str) -> str:
        """Fallback method using keyword matching to identify policy area."""
        try:
            max_score = 0
            best_area = "general"
            
            for area, keywords in self.policy_areas.items():
                score = 0
                
                # Check primary keywords (higher weight)
                for keyword in keywords["primary_keywords"]:
                    if keyword.lower() in text:
                        score += 2
                
                # Check secondary keywords (lower weight)
                for keyword in keywords["secondary_keywords"]:
                    if keyword.lower() in text:
                        score += 1
                
                # Check context phrases (highest weight)
                for phrase in keywords["context_phrases"]:
                    if phrase.lower() in text:
                        score += 3
                
                if score > max_score:
                    max_score = score
                    best_area = area
            
            self.logger.info(f"Keyword analysis selected area: {best_area} (score: {max_score})")
            return best_area
            
        except Exception as e:
            self.logger.error(f"Error in keyword-based analysis: {str(e)}", exc_info=True)
            return "general"
    
    def _extract_llm_response(self, response: str) -> str:
        """Extract the policy area from the LLM response.
        
        Args:
            response (str): Raw response from the LLM
            
        Returns:
            str: Extracted policy area name, or 'general' if no clear area is found
        """
        try:
            # Clean and normalize the response
            cleaned = response.strip().lower()
            
            # Extract the first line or sentence that contains a policy area
            for line in cleaned.split('\n'):
                line = line.strip()
                if line:
                    # Check if the line contains any of our defined policy areas
                    for area in self.policy_areas.keys():
                        if area.lower() in line:
                            self.logger.debug(f"Found policy area in LLM response: {area}")
                            return area
                            
            # If no exact match found, try to find the closest match
            for line in cleaned.split('\n'):
                line = line.strip()
                if line:
                    max_similarity = 0
                    best_area = None
                    
                    for area in self.policy_areas.keys():
                        similarity = self.nlp(line).similarity(self.nlp(area))
                        if similarity > max_similarity and similarity > DORAConfig.STRONG_MATCH_THRESHOLD:
                            max_similarity = similarity
                            best_area = area
                    
                    if best_area:
                        self.logger.debug(f"Found closest matching policy area: {best_area} (similarity: {max_similarity:.3f})")
                        return best_area
            
            self.logger.warning("No clear policy area found in LLM response")
            return "general"
            
        except Exception as e:
            self.logger.error(f"Error extracting policy area from LLM response: {str(e)}", exc_info=True)
            return "general"

    def _extract_article_title(self, article_content: str) -> str:
        """Extract the title from an article's content with enhanced error handling.
        
        Args:
            article_content (str): The content of the article to extract the title from.
            
        Returns:
            str: The extracted title, or a default title if extraction fails.
        """
        try:
            # Try to find a title pattern at the start of the content
            title_match = re.search(r'^(?:Article\s+\d+\s*[-:])?\s*(.+?)(?:\n|$)', article_content.strip())
            if title_match:
                title = title_match.group(1).strip()
                self.logger.debug(f"Extracted title: {title}")
                return title
            
            # Fallback: Try to find any section-like header
            header_match = re.search(r'^#+\s*(.+?)(?:\n|$)', article_content.strip())
            if header_match:
                title = header_match.group(1).strip()
                self.logger.debug(f"Extracted title from header: {title}")
                return title
            
            # If no title found, return a default with article number if available
            article_num_match = re.search(r'Article\s+(\d+)', article_content)
            if article_num_match:
                default_title = f"Article {article_num_match.group(1)}"
                self.logger.warning(f"No title found, using default: {default_title}")
                return default_title
            
            self.logger.warning("Could not extract title, using default")
            return "Untitled Article"
            
        except Exception as e:
            self.logger.error(f"Error extracting article title: {str(e)}", exc_info=True)
            return "Untitled Article"

    def _generate_policy_recommendations(self, gaps_by_area: Dict) -> Dict:
        """Generate detailed policy recommendations based on coverage gaps."""
        try:
            recommendations = defaultdict(lambda: {
                'high_priority': [],
                'medium_priority': [],
                'low_priority': [],
                'implementation_steps': [],
                'best_practices': [],
                'compliance_metrics': []
            })
            
            # Process gaps and generate recommendations
            for area, gaps in gaps_by_area.items():
                # Process RTS gaps
                for req in gaps.get('rts', []):
                    context = req['requirement_text'][:100]
                    priority = self._determine_recommendation_priority(req)
                    
                    if priority == 'high':
                        recommendations[area]['high_priority'].append(
                            f"High priority action needed for RTS requirement: {context}"
                        )
                    elif priority == 'medium':
                        recommendations[area]['medium_priority'].append(
                            f"Review and enhance RTS requirement: {context}"
                        )
                    else:
                        recommendations[area]['low_priority'].append(
                            f"Monitor RTS requirement: {context}"
                        )
                
                # Process ITS gaps
                for req in gaps.get('its', []):
                    context = req['requirement_text'][:100]
                    priority = self._determine_recommendation_priority(req)
                    
                    if priority == 'high':
                        recommendations[area]['high_priority'].append(
                            f"High priority action needed for ITS requirement: {context}"
                        )
                    elif priority == 'medium':
                        recommendations[area]['medium_priority'].append(
                            f"Review and enhance ITS requirement: {context}"
                        )
                    else:
                        recommendations[area]['low_priority'].append(
                            f"Monitor ITS requirement: {context}"
                        )
                
                # Add implementation steps based on policy area
                recommendations[area]['implementation_steps'] = [
                    "1. Review current policies and procedures",
                    "2. Identify gaps and areas for improvement",
                    "3. Develop implementation plan",
                    "4. Execute changes",
                    "5. Monitor and evaluate effectiveness"
                ]
                
                # Add best practices based on policy area
                recommendations[area]['best_practices'] = [
                    "Regular policy reviews",
                    "Staff training and awareness",
                    "Documentation and change management",
                    "Continuous monitoring and improvement"
                ]
                
                # Generate compliance metrics
                recommendations[area]['compliance_metrics'] = self._generate_compliance_metrics(area, gaps)
            
            return dict(recommendations)
            
        except Exception as e:
            self.logger.error(f"Error generating policy recommendations: {str(e)}", exc_info=True)
            return {}

    def _determine_recommendation_priority(self, requirement: Dict) -> str:
        """Determine the priority of a recommendation based on the requirement."""
        try:
            # Initialize score
            priority_score = 0
            
            # Check requirement metadata
            if requirement.get('metadata', {}).get('extraction_confidence') == 'high':
                priority_score += 2
            
            # Check requirement text length and complexity
            text = requirement.get('requirement_text', '')
            if len(text) > 200:  # Complex requirement
                priority_score += 2
            elif len(text) > 100:  # Moderate requirement
                priority_score += 1
            
            # Check for critical keywords
            critical_keywords = {
                'shall', 'must', 'required', 'mandatory', 'essential',
                'critical', 'security', 'risk', 'compliance', 'protect'
            }
            if any(keyword in text.lower() for keyword in critical_keywords):
                priority_score += 2
            
            # Check policy area criticality
            critical_areas = {
                'authentication_security', 'data_protection',
                'incident_response', 'risk_management'
            }
            if requirement.get('policy_area') in critical_areas:
                priority_score += 2
            
            # Determine priority based on score
            if priority_score >= 4:
                return 'high'
            elif priority_score >= 2:
                return 'medium'
            return 'low'
            
        except Exception as e:
            self.logger.error(f"Error determining recommendation priority: {str(e)}", exc_info=True)
            return 'medium'  # Default to medium priority on error

    def _generate_compliance_metrics(self, area: str, gaps: Dict) -> List[str]:
        """Generate compliance metrics for a policy area."""
        try:
            metrics = []
            
            # Common metrics for all areas
            metrics.extend([
                "Percentage of requirements covered",
                "Number of high-priority gaps",
                "Time since last policy review"
            ])
            
            # Area-specific metrics
            area_metrics = {
                'authentication_security': [
                    "MFA adoption rate",
                    "Failed authentication attempts",
                    "Access control violations",
                    "Password policy compliance rate"
                ],
                'data_protection': [
                    "Data encryption coverage",
                    "Data access audit compliance",
                    "Data retention policy adherence",
                    "Data breach incident rate"
                ],
                'incident_response': [
                    "Average incident response time",
                    "Incident resolution rate",
                    "Training completion rate",
                    "Post-incident review completion rate"
                ],
                'risk_management': [
                    "Risk assessment completion rate",
                    "Control effectiveness metrics",
                    "Risk mitigation success rate",
                    "Risk register update frequency"
                ]
            }
            
            # Add area-specific metrics if available
            if area in area_metrics:
                metrics.extend(area_metrics[area])
            
            # Add gap-based metrics
            rts_count = len(gaps.get('rts', []))
            its_count = len(gaps.get('its', []))
            if rts_count > 0:
                metrics.append(f"RTS compliance gap closure rate ({rts_count} gaps)")
            if its_count > 0:
                metrics.append(f"ITS compliance gap closure rate ({its_count} gaps)")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error generating compliance metrics: {str(e)}", exc_info=True)
            return ["Percentage of requirements covered"]  # Return basic metric on error

    def _classify_policy_area(self, text: str) -> str:
        """Determine the policy area of a text using zero-shot classification.
        
        Args:
            text (str): Text to classify
        
        Returns:
            str: The identified policy area
        """
        try:
            # Generate hash for caching
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Check cache
            cached_area = self.cache_manager.get_policy_area(text_hash)
            if cached_area:
                return cached_area
            
            # Truncate text if too long
            if len(text) > 1000:
                text = text[:1000]  # Use first 1000 characters for classification
            
            # Get candidate labels from policy areas
            candidate_labels = list(self.policy_areas.keys())
            
            # Run zero-shot classification
            result = self.zero_shot_classifier(
                text,
                candidate_labels=candidate_labels,
                multi_label=False
            )
            
            # Get top prediction with confidence > threshold
            for label, score in zip(result['labels'], result['scores']):
                if score > DORAConfig.LLM_THRESHOLD:
                    # Store in cache
                    self.cache_manager.set_policy_area(text_hash, label)
                    return label
            
            # Fallback to keyword matching if no confident classification
            policy_area = self._identify_policy_area_by_keywords(text)
            
            # Store in cache
            self.cache_manager.set_policy_area(text_hash, policy_area)
            return policy_area
            
        except Exception as e:
            self.logger.error(f"Error classifying policy area: {str(e)}", exc_info=True)
            return "general"  # Default to general area on error


def retry_failed_policies(failed_policies: List[Dict], dora_path: str, retries: int = 1, adjust_thresholds: bool = True) -> List[Dict]:
    """Retry processing failed policies with adjusted settings.
    
    Args:
        failed_policies: List of failed policy results
        dora_path: Path to DORA legislation
        retries: Number of retry attempts
        adjust_thresholds: Whether to adjust similarity thresholds
        
    Returns:
        List of successfully reprocessed policies
    """
    logger = logging.getLogger('DORAAnalyzer')
    logger.info(f"Attempting to reprocess {len(failed_policies)} failed policies")
    
    successful_retries = []
    remaining_failures = []
    
    # Original thresholds to restore later
    original_thresholds = {
        'strong': DORAConfig.STRONG_MATCH_THRESHOLD,
        'coverage': DORAConfig.COVERAGE_MATCH_THRESHOLD,
        'final': DORAConfig.FINAL_COVERAGE_THRESHOLD
    }
    
    try:
        for retry in range(retries):
            # Adjust thresholds for more lenient matching on each retry
            if adjust_thresholds:
                adjustment_factor = 0.1 * (retry + 1)
                DORAConfig.STRONG_MATCH_THRESHOLD = max(0.4, original_thresholds['strong'] - adjustment_factor)
                DORAConfig.COVERAGE_MATCH_THRESHOLD = max(0.5, original_thresholds['coverage'] - adjustment_factor)
                DORAConfig.FINAL_COVERAGE_THRESHOLD = max(0.6, original_thresholds['final'] - adjustment_factor)
                
                logger.info(f"Retry {retry+1}: Adjusted thresholds - "
                           f"Strong={DORAConfig.STRONG_MATCH_THRESHOLD:.2f}, "
                           f"Coverage={DORAConfig.COVERAGE_MATCH_THRESHOLD:.2f}, "
                           f"Final={DORAConfig.FINAL_COVERAGE_THRESHOLD:.2f}")
            
            # Process each failed policy
            retry_candidates = failed_policies if retry == 0 else remaining_failures
            remaining_failures = []
            
            for policy in retry_candidates:
                policy_path = policy.get('policy_path')
                if not policy_path or not os.path.exists(policy_path):
                    logger.warning(f"Cannot retry policy: {policy.get('policy_name')} - invalid path")
                    remaining_failures.append(policy)
                    continue
                
                logger.info(f"Retrying policy: {policy.get('policy_name')}")
                
                # Process with current settings
                result = process_policy(policy_path, dora_path)
                
                if result.get('success'):
                    logger.info(f"Successfully reprocessed: {policy.get('policy_name')}")
                    successful_retries.append(result)
                else:
                    logger.warning(f"Failed to reprocess: {policy.get('policy_name')}")
                    remaining_failures.append(policy)
    
    except Exception as e:
        logger.error(f"Error during retry processing: {str(e)}", exc_info=True)
    
    finally:
        # Restore original thresholds
        if adjust_thresholds:
            DORAConfig.STRONG_MATCH_THRESHOLD = original_thresholds['strong']
            DORAConfig.COVERAGE_MATCH_THRESHOLD = original_thresholds['coverage']
            DORAConfig.FINAL_COVERAGE_THRESHOLD = original_thresholds['final']
            logger.info("Restored original threshold settings")
    
    logger.info(f"Retry processing summary: {len(successful_retries)} succeeded, {len(remaining_failures)} still failed")
    return successful_retries

def process_policy(policy_path: str, dora_path: str) -> Dict:
    """Process a policy document against DORA requirements.
    
    This function analyzes a policy document for compliance with DORA's RTS and ITS requirements.
    
    Args:
        policy_path: Path to the policy PDF to analyze
        dora_path: Path to the DORA legislation PDF
        
    Returns:
        Dict with analysis results including:
        - success (bool): Whether analysis was successful
        - policy_name (str): Name of the policy file
        - policy_path (str): Path to the policy file
        - rts_coverage (float): Percentage of RTS requirements covered
        - its_coverage (float): Percentage of ITS requirements covered
        - overall_coverage (float): Overall coverage percentage
        - analysis_timestamp (str): When the analysis was performed
    """
    logger = logging.getLogger('DORAAnalyzer')
    logger.info(f"Processing policy: {os.path.basename(policy_path)}")
    
    try:
        # Extract text from policy document
        with pdfplumber.open(policy_path) as pdf:
            policy_text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                policy_text += page_text + "\n\n"
        
        if not policy_text.strip():
            logger.error(f"Failed to extract text from {policy_path}")
            return {
                'success': False,
                'policy_name': os.path.basename(policy_path),
                'policy_path': policy_path,
                'error': "Failed to extract text from PDF"
            }
            
        # Clean and normalize the extracted text
        policy_text = TextProcessor.clean_text(policy_text)
        
        # Initialize the analyzer
        analyzer = DORAComplianceAnalyzer(dora_path)
        
        # Extract RTS and ITS requirements if not done already
        if not analyzer.extract_technical_standards():
            logger.error(f"Failed to extract technical standards from DORA")
            return {
                'success': False,
                'policy_name': os.path.basename(policy_path),
                'policy_path': policy_path,
                'error': "Failed to extract technical standards from DORA"
            }
        
        # Analyze the policy document against DORA requirements
        policy_name = os.path.basename(policy_path)
        analysis_results = analyzer.analyze_policy_document(policy_name, policy_text)
        
        if not analysis_results.get('analysis_complete', False):
            logger.error(f"Analysis failed for {policy_name}: {analysis_results.get('error', 'Unknown error')}")
            return {
                'success': False,
                'policy_name': policy_name,
                'policy_path': policy_path,
                'error': analysis_results.get('error', 'Analysis failed')
            }
            
        # Create success response
        return {
            'success': True,
            'policy_name': policy_name,
            'policy_path': policy_path,
            'rts_coverage': analysis_results.get('rts_coverage', 0.0),
            'its_coverage': analysis_results.get('its_coverage', 0.0),
            'overall_coverage': (analysis_results.get('rts_coverage', 0.0) + analysis_results.get('its_coverage', 0.0)) / 2,
            'total_rts': analysis_results.get('total_rts', 0),
            'covered_rts': analysis_results.get('covered_rts', 0),
            'total_its': analysis_results.get('total_its', 0),
            'covered_its': analysis_results.get('covered_its', 0),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(policy_path)}: {str(e)}", exc_info=True)
        return {
            'success': False,
            'policy_name': os.path.basename(policy_path),
            'policy_path': policy_path,
            'error': str(e)
        }

def is_dora_relevant(policy_path: str) -> Tuple[bool, str]:
    """
    Screen a policy document to determine if it's relevant to DORA compliance.
    
    Args:
        policy_path: Path to the policy PDF to screen
        
    Returns:
        Tuple containing:
        - Boolean indicating if the document is relevant
        - Reason for the decision
    """
    logger = logging.getLogger('DORAAnalyzer')
    logger.info(f"Screening document for DORA relevance: {os.path.basename(policy_path)}")
    
    try:
        # Extract text from policy document (first 10 pages or all if fewer)
        with pdfplumber.open(policy_path) as pdf:
            policy_text = ""
            max_pages = min(10, len(pdf.pages))  # Only check first 10 pages for efficiency
            for i in range(max_pages):
                page_text = pdf.pages[i].extract_text() or ""
                policy_text += page_text + "\n\n"
        
        if not policy_text.strip():
            return False, "Failed to extract text from PDF"
            
        # Clean the extracted text
        policy_text = TextProcessor.clean_text(policy_text)
        
        # DORA-specific keywords and phrases
        dora_keywords = [
            "digital operational resilience", "DORA", "operational resilience",
            "ict risk management", "cyber security", "cyber resilience",
            "ICT third party risk", "ICT incident management", "ICT business continuity",
            "financial entity", "critical ICT service provider", "digital testing",
            "regulatory technical standards", "implementing technical standards",
            "RTS", "ITS", "cyber attack", "operational incident", "information security"
        ]
        
        # Check for DORA keywords
        keyword_matches = []
        for keyword in dora_keywords:
            if keyword.lower() in policy_text.lower():
                keyword_matches.append(keyword)
        
        # If we have multiple keyword matches, it's likely relevant
        if len(keyword_matches) >= 3:
            return True, f"Found DORA-related keywords: {', '.join(keyword_matches[:5])}"
        
        # If fewer matches, do deeper analysis
        if len(keyword_matches) > 0:
            # Check for sections about compliance, risk management, or security
            compliance_patterns = [
                r"(?i)compliance\s+with\s+regulatory",
                r"(?i)regulatory\s+compliance",
                r"(?i)security\s+policies",
                r"(?i)risk\s+management\s+framework",
                r"(?i)incident\s+response",
                r"(?i)business\s+continuity",
                r"(?i)disaster\s+recovery",
                r"(?i)third.party\s+management"
            ]
            
            for pattern in compliance_patterns:
                if re.search(pattern, policy_text):
                    return True, f"Found DORA-related content with {len(keyword_matches)} keywords"
        
        # If document is very short, it's likely not a full policy document
        if len(policy_text.split()) < 500:  # Fewer than 500 words
            return False, "Document too short to be a comprehensive policy"
        
        # Default to not relevant if few matches
        return False, f"Insufficient DORA-related content ({len(keyword_matches)} keyword matches)"
            
    except Exception as e:
        logger.error(f"Error screening {os.path.basename(policy_path)}: {str(e)}", exc_info=True)
        return False, f"Error during screening: {str(e)}"


def main():
    """Main function to run the DORA compliance analyzer."""
    logger = logging.getLogger('DORAAnalyzer')
    setup_logging()  # Initialize logging
    
    try:
        # Set multiprocessing start method to 'spawn' if CUDA is available
        if torch.cuda.is_available():
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
            logger.info("Using 'spawn' start method for multiprocessing with CUDA")
        
        dora_path = "CELEX_32022R2554_EN_TXT.pdf"
        policy_folder = "policies"
        
        # Check if DORA file exists
        if not os.path.exists(dora_path):
            logger.error(f"DORA legislation file not found: {dora_path}")
            logger.info("Please ensure the DORA legislation PDF (CELEX_32022R2554_EN_TXT.pdf) is in the root directory")
            return
        
        # Check if policies folder exists
        if not os.path.exists(policy_folder):
            logger.warning(f"Policies folder not found: {policy_folder}")
            logger.info("Creating policies folder. Please add your policy documents there.")
            os.makedirs(policy_folder, exist_ok=True)
            return
        
        # Get all PDF files from the folder
        logger.info(f"Scanning for policy documents in: {policy_folder}")
        all_pdf_files = list(Path(policy_folder).glob("**/*.pdf"))
        
        if not all_pdf_files:
            logger.warning("No PDF files found in the specified folder!")
            return
        
        logger.info(f"Found {len(all_pdf_files)} policy documents, screening for DORA relevance...")
        
        # Screen documents for DORA relevance
        relevant_files = []
        irrelevant_files = []
        
        for pdf_file in tqdm(all_pdf_files, desc="Screening documents", disable=DORAConfig.PROGRESS_BAR_DISABLE):
            is_relevant, reason = is_dora_relevant(str(pdf_file))
            if is_relevant:
                relevant_files.append(pdf_file)
                logger.info(f"Relevant: {pdf_file.name} - {reason}")
            else:
                irrelevant_files.append((pdf_file, reason))
                logger.info(f"Not relevant: {pdf_file.name} - {reason}")
        
        # Log screening results
        if irrelevant_files:
            logger.info("\nThe following files were determined to be not relevant to DORA and will be skipped:")
            for file, reason in irrelevant_files:
                logger.info(f"- {file.name}: {reason}")
        
        pdf_files = relevant_files
        logger.info(f"\nProceeding with {len(pdf_files)} DORA-relevant documents")
        
        if not pdf_files:
            logger.warning("No DORA-relevant policy documents found!")
            return
        
        # Set up multiprocessing
        num_processes = min(cpu_count(), len(pdf_files))
        chunk_size = max(1, len(pdf_files) // (num_processes * 4))  # Dynamic chunk size
        
        # Process policies in parallel with progress tracking and error handling
        logger.info(f"Starting parallel analysis with {num_processes} processes (chunk size: {chunk_size})...")
        failed_policies = []
        successful_policies = []
        
        with Pool(processes=num_processes) as pool:
            try:
                # Create a partial function with dora_path
                process_policy_with_dora = partial(process_policy, dora_path=dora_path)
                
                results = list(tqdm(
                    pool.imap_unordered(process_policy_with_dora, pdf_files, chunksize=chunk_size),
                    total=len(pdf_files),
                    desc="Analyzing policies",
                    disable=DORAConfig.PROGRESS_BAR_DISABLE
                ))
                
                # Process results
                for result in results:
                    if result['success']:
                        successful_policies.append(result)
                    else:
                        failed_policies.append(result)
                    
            except KeyboardInterrupt:
                logger.warning("Analysis interrupted by user")
                pool.terminate()
                raise
            except Exception as e:
                logger.error(f"Error in parallel processing: {str(e)}", exc_info=True)
                pool.terminate()
                raise
        
        # Log analysis summary
        logger.info("\nAnalysis Summary:")
        logger.info(f"Total policies processed: {len(pdf_files)}")
        logger.info(f"Successful analyses: {len(successful_policies)}")
        logger.info(f"Failed analyses: {len(failed_policies)}")
        
        # Retry failed policies with adjusted settings
        if failed_policies:
            logger.info("\nAttempting to retry failed policies with adjusted settings...")
            retry_successes = retry_failed_policies(failed_policies, dora_path, retries=2)
            
            # Update counts with retry results
            successful_policies.extend(retry_successes)
            failed_policies = [p for p in failed_policies if p['policy_name'] not in [r['policy_name'] for r in retry_successes]]
            
            logger.info("\nUpdated Analysis Summary (after retries):")
            logger.info(f"Total policies processed: {len(pdf_files)}")
            logger.info(f"Successful analyses: {len(successful_policies)}")
            logger.info(f"Failed analyses: {len(failed_policies)}")
            
        if failed_policies:
            logger.warning("\nFailed Policies:")
            for policy in failed_policies:
                logger.warning(f"- {policy['policy_name']}: {policy['error']}")
                
        # Generate and save gap analysis report
        if successful_policies:
            logger.info("Generating gap analysis report...")
            
            # Initialize analyzer with DORA path
            analyzer = DORAComplianceAnalyzer(dora_path)
            
            # Extract RTS and ITS requirements
            if not analyzer.extract_technical_standards():
                logger.error("Failed to extract technical standards from DORA")
                return
                
            report_path = analyzer.generate_gap_analysis_report()
            
            if report_path == "Error generating report":
                logger.error("Failed to generate gap analysis report")
                return
            
            logger.info(f"Gap analysis report has been written to {report_path}")
        else:
            logger.error("No successful policy analyses to generate report")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Program interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up any temporary files or resources
        try:
            gc.collect()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
