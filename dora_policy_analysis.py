import pdfplumber
import pandas as pd
import spacy
import re
import torch
from typing import Tuple, List, Dict
from collections import defaultdict
from pathlib import Path
import difflib
from transformers import pipeline
from datetime import datetime
from collections import Counter
class DORAComplianceAnalyzer:
    def __init__(self, dora_pdf_path):
        """Initialize the analyzer with proper sequence."""
        try:
            print("Starting DORA Compliance Analyzer initialization...")
            # 1. Validate input
            if not Path(dora_pdf_path).exists():
                raise FileNotFoundError(f"DORA PDF not found at: {dora_pdf_path}")
            self.dora_pdf_path = dora_pdf_path
            # 2. Initialize NLP components first (required for later steps)
            print("Loading NLP models...")
            self.nlp = self._initialize_nlp()
            # 3. Initialize LLM (required for analysis)
            print("Loading LLM model...")
            self.llm = self._initialize_llm()
            # 4. Initialize policy areas (core reference data)
            print("Initializing policy areas...")
            self.policy_areas = self._initialize_policy_areas()
            # 5. Extract and clean DORA text
            print("Extracting DORA text...")
            self.dora_text = self._extract_and_clean_text()
            # 6. Initialize storage structures
            self.rts_requirements = {}
            self.its_requirements = {}
            self.policy_coverage = {}  # Initialize the policy coverage dictionary
            self.articles_processed = 0
            print("Initialization complete.")
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise
    def _initialize_nlp(self):
        """Initialize NLP with error handling."""
        try:
            return spacy.load("en_core_web_lg")
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download("en_core_web_lg")
            return spacy.load("en_core_web_lg")
    def _initialize_llm(self):
        """Initialize LLM with proper configuration."""
        return pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=2048,
            temperature=0.1,  # Lower temperature for more focused analysis
            pad_token_id=2,  # Explicitly set pad_token_id
            eos_token_id=2,  # Explicitly set eos_token_id
        )
    def _initialize_policy_areas(self):
        """Initialize enhanced policy areas with weighted keywords."""
        return {
            "authentication_security": {
                "primary_keywords": [
                    "authentication",
                    "login",
                    "credentials",
                    "access control",
                    "identity verification",
                    "multi-factor",
                    "MFA",
                    "2FA",
                    "authorization",
                    "identity management",
                    "privileged access",
                ],
                "secondary_keywords": [
                    "password",
                    "biometric",
                    "token",
                    "security code",
                    "authentication factor",
                    "identity provider",
                    "SSO",
                    "single sign-on",
                    "access rights",
                    "user permissions",
                ],
                "context_phrases": [
                    "user authentication process",
                    "secure access control",
                    "authentication protocol implementation",
                    "identity management system",
                    "access control mechanism",
                    "privileged access management",
                ],
            },
            "cryptography_security": {
                "primary_keywords": [
                    "encryption",
                    "cryptographic",
                    "cipher",
                    "key management",
                    "PKI",
                    "digital signature",
                    "hash",
                    "cryptographic controls",
                    "key rotation",
                    "cryptographic algorithm",
                ],
                "secondary_keywords": [
                    "AES",
                    "RSA",
                    "elliptic curve",
                    "ECDSA",
                    "key derivation",
                    "PKCS",
                    "certificate",
                    "TLS",
                    "SSL",
                    "HSM",
                    "symmetric encryption",
                    "asymmetric encryption",
                ],
                "context_phrases": [
                    "encryption implementation",
                    "key lifecycle management",
                    "cryptographic protocol requirements",
                    "secure key generation",
                    "cryptographic module validation",
                    "encryption at rest",
                ],
            },
            "data_protection": {
                "primary_keywords": [
                    "data protection",
                    "privacy",
                    "GDPR",
                    "data security",
                    "personal data",
                    "sensitive data",
                    "data processing",
                    "data classification",
                    "data governance",
                    "data sovereignty",
                ],
                "secondary_keywords": [
                    "data minimization",
                    "data retention",
                    "data disposal",
                    "data handling",
                    "data transfer",
                    "data storage",
                    "data anonymization",
                    "pseudonymization",
                    "data masking",
                ],
                "context_phrases": [
                    "protection of personal data",
                    "data privacy requirements",
                    "secure data processing",
                    "data protection measures",
                    "data lifecycle management",
                    "privacy by design",
                ],
            },
            "incident_response": {
                "primary_keywords": [
                    "incident response",
                    "security incident",
                    "breach response",
                    "incident management",
                    "incident handling",
                    "CSIRT",
                    "incident detection",
                    "incident investigation",
                    "SOC",
                ],
                "secondary_keywords": [
                    "incident reporting",
                    "incident analysis",
                    "forensics",
                    "incident recovery",
                    "incident containment",
                    "incident triage",
                    "incident escalation",
                    "incident documentation",
                ],
                "context_phrases": [
                    "incident response procedure",
                    "security incident handling",
                    "breach notification requirements",
                    "incident management process",
                    "security operations center",
                    "incident response team",
                ],
            },
            "risk_management": {
                "primary_keywords": [
                    "risk management",
                    "risk assessment",
                    "risk analysis",
                    "risk mitigation",
                    "risk treatment",
                    "risk framework",
                    "risk appetite",
                    "risk tolerance",
                    "risk governance",
                ],
                "secondary_keywords": [
                    "risk identification",
                    "risk evaluation",
                    "risk monitoring",
                    "risk control",
                    "risk register",
                    "risk matrix",
                    "risk acceptance",
                    "risk transfer",
                    "residual risk",
                ],
                "context_phrases": [
                    "risk management framework",
                    "risk assessment process",
                    "risk mitigation measures",
                    "risk control implementation",
                    "enterprise risk management",
                    "risk reporting",
                ],
            },
            "business_continuity": {
                "primary_keywords": [
                    "business continuity",
                    "disaster recovery",
                    "BCP",
                    "DRP",
                    "service continuity",
                    "operational resilience",
                    "business resilience",
                    "continuity planning",
                ],
                "secondary_keywords": [
                    "recovery time objective",
                    "RTO",
                    "recovery point objective",
                    "RPO",
                    "business impact analysis",
                    "BIA",
                    "contingency plan",
                    "failover",
                    "backup strategy",
                ],
                "context_phrases": [
                    "business continuity planning",
                    "disaster recovery procedures",
                    "service continuity requirements",
                    "resilience measures",
                    "recovery strategy implementation",
                    "continuity testing",
                ],
            },
            "change_management": {
                "primary_keywords": [
                    "change management",
                    "change control",
                    "release management",
                    "deployment management",
                    "configuration management",
                    "change process",
                    "version control",
                ],
                "secondary_keywords": [
                    "change request",
                    "change approval",
                    "change implementation",
                    "release process",
                    "deployment process",
                    "rollback procedure",
                    "configuration baseline",
                    "change documentation",
                ],
                "context_phrases": [
                    "change management procedure",
                    "change control process",
                    "release management requirements",
                    "deployment controls",
                    "configuration management system",
                    "change advisory board",
                ],
            },
            "vendor_management": {
                "primary_keywords": [
                    "vendor management",
                    "supplier management",
                    "third party",
                    "outsourcing",
                    "service provider",
                    "contractor",
                    "vendor assessment",
                    "supplier evaluation",
                ],
                "secondary_keywords": [
                    "vendor risk",
                    "contract management",
                    "SLA",
                    "service level agreement",
                    "vendor compliance",
                    "supplier audit",
                    "vendor performance",
                    "vendor security",
                ],
                "context_phrases": [
                    "vendor management process",
                    "supplier assessment requirements",
                    "third party risk management",
                    "outsourcing controls",
                    "vendor due diligence",
                    "supplier relationship management",
                ],
            },
            "asset_management": {
                "primary_keywords": [
                    "asset management",
                    "asset inventory",
                    "asset tracking",
                    "asset lifecycle",
                    "asset register",
                    "asset classification",
                    "asset ownership",
                    "critical assets",
                ],
                "secondary_keywords": [
                    "asset valuation",
                    "asset disposal",
                    "asset maintenance",
                    "asset monitoring",
                    "asset protection",
                    "asset controls",
                    "asset documentation",
                    "asset audit",
                ],
                "context_phrases": [
                    "asset management process",
                    "asset lifecycle management",
                    "asset protection requirements",
                    "asset control implementation",
                    "critical asset identification",
                    "asset inventory management",
                ],
            },
            "compliance_monitoring": {
                "primary_keywords": [
                    "compliance monitoring",
                    "regulatory compliance",
                    "audit",
                    "compliance assessment",
                    "compliance reporting",
                    "controls testing",
                    "compliance framework",
                    "compliance program",
                ],
                "secondary_keywords": [
                    "compliance review",
                    "compliance verification",
                    "control testing",
                    "compliance documentation",
                    "compliance metrics",
                    "audit trail",
                    "compliance evidence",
                    "control effectiveness",
                ],
                "context_phrases": [
                    "compliance monitoring process",
                    "regulatory reporting requirements",
                    "compliance assessment procedures",
                    "control testing methodology",
                    "compliance program management",
                    "audit documentation",
                ],
            },
        }
    def _validate_policy_areas(self):
        """Validate policy areas structure and keywords."""
        required_keys = {"primary_keywords", "secondary_keywords", "context_phrases"}
        for area, content in self.policy_areas.items():
            # Check structure
            if not all(key in content for key in required_keys):
                raise ValueError(f"Missing required keys in policy area: {area}")
            # Validate keyword lists
            for key in required_keys:
                if not isinstance(content[key], list):
                    raise TypeError(f"Keywords must be lists: {area}.{key}")
                if not all(isinstance(k, str) for k in content[key]):
                    raise TypeError(f"All keywords must be strings: {area}.{key}")
            # Check for duplicates
            all_keywords = (
                content["primary_keywords"]
                + content["secondary_keywords"]
                + content["context_phrases"]
            )
            duplicates = [k for k, count in Counter(all_keywords).items() if count > 1]
            if duplicates:
                print(f"Warning: Duplicate keywords found in {area}: {duplicates}")
    def process_dora_requirements(self):
        """Main process to extract and analyze DORA requirements."""
        print("\nStarting DORA requirements processing...")
        # Extract technical standards
        requirements_count = self.extract_technical_standards()
        print(f"\nExtracted {requirements_count} total requirements")
        # Analyze requirements by policy area
        area_statistics = defaultdict(lambda: {"total": 0, "rts": 0, "its": 0})
        for article_num, reqs in self.rts_requirements.items():
            for req in reqs:
                area = req["policy_area"]
                area_statistics[area]["total"] += 1
                area_statistics[area]["rts"] += 1
        for article_num, reqs in self.its_requirements.items():
            for req in reqs:
                area = req["policy_area"]
                area_statistics[area]["total"] += 1
                area_statistics[area]["its"] += 1
        # Print area statistics
        print("\nRequirements by Policy Area:")
        for area, stats in area_statistics.items():
            print(f"\n{area.replace('_', ' ').title()}:")
            print(f"- Total Requirements: {stats['total']}")
            print(f"- RTS Requirements: {stats['rts']}")
            print(f"- ITS Requirements: {stats['its']}")
        return area_statistics
    def _extract_and_clean_text(self) -> str:
        """Extract and clean text from PDF document."""
        try:
            print(f"Extracting text from: {self.dora_pdf_path}")
            extracted_text = ""
            tables_data = []
            
            with pdfplumber.open(self.dora_pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"Processing page {page_num}/{len(pdf.pages)}")
                    
                    # Extract tables first
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            if table:  # Skip empty tables
                                self._process_completed_table(table, tables_data, page_num, page_num)
                    
                    # Extract and clean text
                    page_text = page.extract_text() or ""
                    cleaned_text = self._clean_text(page_text)
                    
                    # Remove table content from text to avoid duplication
                    if tables:
                        cleaned_text = self._remove_table_content_from_text(cleaned_text, tables)
                    
                    extracted_text += cleaned_text + "\n\n"
            
            print(f"Extracted {len(extracted_text)} characters of text")
            return extracted_text
            
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            raise
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        try:
            if not text:
                return ""
            
            # Convert to string if not already
            text = str(text)
            
            # Basic cleaning
            cleaned = text.strip()
            
            # Remove extra whitespace
            cleaned = ' '.join(cleaned.split())
            
            # Remove special characters but keep essential punctuation
            cleaned = re.sub(r'[^\w\s.,;:?!()\-\'\"]+', ' ', cleaned)
            
            # Normalize whitespace around punctuation
            cleaned = re.sub(r'\s*([.,;:?!()\-])\s*', r'\1 ', cleaned)
            
            # Remove multiple spaces
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Remove empty lines
            cleaned = '\n'.join(line.strip() for line in cleaned.split('\n') if line.strip())
            
            return cleaned.strip()
            
        except Exception as e:
            print(f"Error cleaning text: {str(e)}")
            return text
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using both spaCy and Llama model."""
        try:
            # Traditional NLP-based similarity
            doc1 = self.nlp(text1[:25000])
            doc2 = self.nlp(text2[:25000])
            if not doc1 or not doc2:
                return 0.0
            # Calculate traditional similarity
            traditional_sim = self._calculate_traditional_similarity(doc1, doc2)
            # Use Llama for semantic understanding
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Compare the semantic similarity of these two texts on a scale of 0 to 1, considering regulatory and technical context<|eot_id|><|start_header_id|>user<|end_header_id|>
Text 1: {text1[:1000]}
Text 2: {text2[:1000]}
Provide only a number between 0 and 1.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            llm_response = self.llm(prompt, max_new_tokens=10)
            try:
                # Extract the generated text from the response
                generated_text = llm_response[0]["generated_text"] if isinstance(llm_response, list) else str(llm_response)
                llm_sim = float(
                    re.search(
                        r"([0-9]*[.])?[0-9]+", generated_text
                    ).group()
                )
                llm_sim = max(0.0, min(1.0, llm_sim))
            except (ValueError, AttributeError, TypeError):
                llm_sim = 0.0
            # Combine similarities with weighted average
            combined_sim = (0.4 * traditional_sim) + (0.6 * llm_sim)
            print(
                f"Similarity scores - Traditional: {traditional_sim:.3f}, LLM: {llm_sim:.3f}, Combined: {combined_sim:.3f}"
            )
            return combined_sim
        except Exception as e:
            print(f"Error in similarity calculation: {str(e)}")
            return 0.0
    def _calculate_traditional_similarity(self, doc1, doc2):
        """Calculate traditional similarity using spaCy."""
        # Calculate cosine similarity
        cosine_sim = 0.0
        for sent1 in doc1.sents:
            if not sent1.vector_norm:
                continue
            for sent2 in doc2.sents:
                if not sent2.vector_norm:
                    continue
                try:
                    sim = sent1.similarity(sent2)
                    cosine_sim = max(cosine_sim, sim)
                except Exception:
                    continue
        # Calculate semantic similarity using word overlap
        def get_weighted_words(doc):
            words = {}
            for token in doc:
                if not token.is_stop and len(token.text) > 2:
                    weight = (
                        2.0
                        if token.ent_type_ or token.pos_ in ["NOUN", "VERB"]
                        else 1.0
                    )
                    words[token.text.lower()] = weight
            return words
        words1 = get_weighted_words(doc1)
        words2 = get_weighted_words(doc2)
        if not words1 or not words2:
            return cosine_sim
        # Calculate weighted Jaccard similarity
        intersection_score = sum(
            min(words1.get(w, 0), words2.get(w, 0)) for w in set(words1) & set(words2)
        )
        union_score = sum(
            max(words1.get(w, 0), words2.get(w, 0)) for w in set(words1) | set(words2)
        )
        semantic_sim = intersection_score / union_score if union_score > 0 else 0
        return (0.5 * cosine_sim) + (0.5 * semantic_sim)
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a given PDF file and clean it."""
        try:
            extracted_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    cleaned_text = self._clean_text(page_text)
                    extracted_text += cleaned_text + "\n\n"
            return extracted_text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    def analyze_policy_document(self, policy_path: str, policy_name: str) -> None:
        """Analyze a policy document against DORA requirements."""
        try:
            print(f"\nAnalyzing policy document: {policy_name}")
            
            # Initialize coverage for this policy
            policy_results = []
            
            # Load and process policy document
            policy_text = self._extract_text_from_pdf(policy_path)
            if not policy_text:
                print(f"Warning: No text extracted from {policy_name}")
                return
            # Analyze RTS requirements
            print("\nAnalyzing RTS requirements...")
            for article_num, reqs in self.rts_requirements.items():
                for req in reqs:
                    coverage = self._analyze_requirement_coverage(req, policy_text)
                    req_result = req.copy()  # Create a copy of the requirement
                    req_result.update({
                        'covered': coverage['covered'],
                        'similarity_score': coverage['similarity_score'],
                        'matching_sections': coverage['matching_sections'],
                        'requirement_type': 'RTS',
                        'article_num': article_num
                    })
                    policy_results.append(req_result)
                    
                    if coverage['covered']:
                        print(f"\nFound coverage for RTS requirement in Article {article_num}:")
                        print(f"Requirement: {req['requirement_text'][:200]}...")
                        print(f"Similarity score: {coverage['similarity_score']:.2f}")
            # Analyze ITS requirements
            print("\nAnalyzing ITS requirements...")
            for article_num, reqs in self.its_requirements.items():
                for req in reqs:
                    coverage = self._analyze_requirement_coverage(req, policy_text)
                    req_result = req.copy()  # Create a copy of the requirement
                    req_result.update({
                        'covered': coverage['covered'],
                        'similarity_score': coverage['similarity_score'],
                        'matching_sections': coverage['matching_sections'],
                        'requirement_type': 'ITS',
                        'article_num': article_num
                    })
                    policy_results.append(req_result)
                    
                    if coverage['covered']:
                        print(f"\nFound coverage for ITS requirement in Article {article_num}:")
                        print(f"Requirement: {req['requirement_text'][:200]}...")
                        print(f"Similarity score: {coverage['similarity_score']:.2f}")
            # Store results for this policy
            self.policy_coverage[policy_name] = policy_results
            print(f"\nCompleted analysis of {policy_name}")
            print(f"Total requirements covered: {len([r for r in policy_results if r['covered']])}")
        except Exception as e:
            print(f"Error analyzing policy document {policy_name}: {str(e)}")
            raise  # Re-raise the exception to see the full error trace
    def generate_gap_analysis_report(self):
        """Generate a comprehensive gap analysis report with detailed breakdowns."""
        try:
            report_sections = [
                "DORA Gap Analysis Report",
                "=" * 50,
                "\n1. Extraction Summary",
                "-" * 20,
                f"Articles processed: {self.articles_processed}",
                f"RTS requirements found: {sum(len(reqs) for reqs in self.rts_requirements.values())}",
                f"ITS requirements found: {sum(len(reqs) for reqs in self.its_requirements.values())}",
                "\n2. Coverage Analysis",
                "-" * 20,
            ]
            # Analyze RTS Requirements
            report_sections.extend([
                "\nRTS Requirements Coverage:",
                "-" * 25
            ])
            for article_num, reqs in sorted(self.rts_requirements.items()):
                covered_reqs = [r for r in reqs if r.get('covered', False)]
                uncovered_reqs = [r for r in reqs if not r.get('covered', False)]
                
                report_sections.extend([
                    f"\nArticle {article_num}:",
                    f"Total Requirements: {len(reqs)}",
                    f"Coverage Rate: {(len(covered_reqs)/len(reqs)*100):.1f}%"
                ])
                if covered_reqs:
                    report_sections.append("\n✓ Covered Requirements:")
                    for req in covered_reqs:
                        report_sections.extend([
                            f"\nRequirement: {req['requirement_text'][:200]}...",
                            f"Policy Area: {req['policy_area']}",
                            f"Similarity Score: {req.get('similarity_score', 0):.2f}",
                            "Matching Policies:"
                        ])
                        for policy_name, results in self.policy_coverage.items():
                            matches = [r for r in results if r['requirement_text'] == req['requirement_text'] and r['covered']]
                            if matches:
                                for match in matches:
                                    report_sections.append(f"- {policy_name}: {match['similarity_score']:.2f}")
                if uncovered_reqs:
                    report_sections.append("\n✗ Gaps (Not Covered):")
                    for req in uncovered_reqs:
                        report_sections.extend([
                            f"\nRequirement: {req['requirement_text'][:200]}...",
                            f"Policy Area: {req['policy_area']}"
                        ])
            # Analyze ITS Requirements
            report_sections.extend([
                "\nITS Requirements Coverage:",
                "-" * 25
            ])
            for article_num, reqs in sorted(self.its_requirements.items()):
                covered_reqs = [r for r in reqs if r.get('covered', False)]
                uncovered_reqs = [r for r in reqs if not r.get('covered', False)]
                
                report_sections.extend([
                    f"\nArticle {article_num}:",
                    f"Total Requirements: {len(reqs)}",
                    f"Coverage Rate: {(len(covered_reqs)/len(reqs)*100):.1f}%"
                ])
                if covered_reqs:
                    report_sections.append("\n✓ Covered Requirements:")
                    for req in covered_reqs:
                        report_sections.extend([
                            f"\nRequirement: {req['requirement_text'][:200]}...",
                            f"Policy Area: {req['policy_area']}",
                            f"Similarity Score: {req.get('similarity_score', 0):.2f}",
                            "Matching Policies:"
                        ])
                        for policy_name, results in self.policy_coverage.items():
                            matches = [r for r in results if r['requirement_text'] == req['requirement_text'] and r['covered']]
                            if matches:
                                for match in matches:
                                    report_sections.append(f"- {policy_name}: {match['similarity_score']:.2f}")
                if uncovered_reqs:
                    report_sections.append("\n✗ Gaps (Not Covered):")
                    for req in uncovered_reqs:
                        report_sections.extend([
                            f"\nRequirement: {req['requirement_text'][:200]}...",
                            f"Policy Area: {req['policy_area']}"
                        ])
            # Policy Area Summary
            report_sections.extend([
                "\n3. Policy Area Coverage Summary",
                "-" * 30
            ])
            for area in sorted(set(req['policy_area'] for reqs in self.rts_requirements.values() for req in reqs)):
                area_reqs = []
                for reqs in self.rts_requirements.values():
                    area_reqs.extend([r for r in reqs if r['policy_area'] == area])
                for reqs in self.its_requirements.values():
                    area_reqs.extend([r for r in reqs if r['policy_area'] == area])
                
                total_area = len(area_reqs)
                covered_area = len([r for r in area_reqs if r.get('covered', False)])
                coverage_rate = (covered_area/total_area*100) if total_area > 0 else 0
                
                report_sections.extend([
                    f"\n{area.replace('_', ' ').title()}:",
                    f"Total Requirements: {total_area}",
                    f"Requirements Covered: {covered_area}",
                    f"Coverage Rate: {coverage_rate:.1f}%"
                ])
            return "\n".join(report_sections)
            
        except Exception as e:
            print(f"Error generating gap analysis report: {str(e)}")
            return "Error generating report"
    def _remove_table_content_from_text(self, text: str, tables: List[List[List[str]]]) -> str:
        """Remove table content from text."""
        if not tables:
            return text
        
        # Create a set of table content for faster lookup
        table_content = set()
        for table in tables:
            for row in table:
                for cell in row:
                    if isinstance(cell, str):
                        table_content.add(cell.strip())
        
        # Split text into lines and remove those containing table content
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            should_keep = True
            for content in table_content:
                if content in line:
                    should_keep = False
                    break
            
            if should_keep:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    def extract_technical_standards(self) -> int:
        """Extract RTS and ITS requirements with enhanced detection and logging."""
        try:
            print("\nStarting technical standards extraction...")
            
            # More comprehensive patterns for RTS and ITS
            rts_patterns = [
                r"(?:the\s+)?(?:RTS|regulatory\s+technical\s+standards?)\s+(?:shall|should|must|will|to|that|which)\s+([^\.]+\.[^\.]+)",
                r"develop\s+(?:draft\s+)?regulatory\s+technical\s+standards?\s+(?:to|that|which|for|on)\s+([^\.]+\.[^\.]+)",
                r"specify\s+(?:in|through)\s+(?:the\s+)?regulatory\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
                r"requirements?\s+(?:shall|should|must|will)\s+be\s+specified\s+in\s+regulatory\s+technical\s+standards?\s+([^\.]+\.[^\.]+)"
            ]
            
            its_patterns = [
                r"(?:the\s+)?(?:ITS|implementing\s+technical\s+standards?)\s+(?:shall|should|must|will|to|that|which)\s+([^\.]+\.[^\.]+)",
                r"develop\s+(?:draft\s+)?implementing\s+technical\s+standards?\s+(?:to|that|which|for|on)\s+([^\.]+\.[^\.]+)",
                r"specify\s+(?:in|through)\s+(?:the\s+)?implementing\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
                r"format\s+(?:shall|should|must|will)\s+be\s+specified\s+in\s+implementing\s+technical\s+standards?\s+([^\.]+\.[^\.]+)"
            ]
            # Extract articles with enhanced pattern
            article_pattern = r"Article\s+(\d+[a-z]?)\s*[–-]?\s*([^\n]+)(?:\n|\r\n?)(.*?)(?=Article\s+\d+[a-z]?|$)"
            articles = re.finditer(article_pattern, self.dora_text, re.DOTALL | re.IGNORECASE)
            
            articles_processed = 0
            rts_found = 0
            its_found = 0
            
            for article in articles:
                articles_processed += 1
                article_num = article.group(1)
                article_title = article.group(2).strip()
                article_content = article.group(3).strip()
                
                print(f"\nProcessing Article {article_num}: {article_title[:100]}...")
                
                # Identify policy area
                article_area = self._identify_policy_area(f"{article_title} {article_content}")
                
                # Process RTS requirements
                for pattern in rts_patterns:
                    matches = re.finditer(pattern, article_content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        rts_found += 1
                        requirement_text = match.group(1).strip()
                        print(f"\nFound RTS requirement:")
                        print(f"Text: {requirement_text[:200]}...")
                        
                        requirement = {
                            'article_num': article_num,
                            'article_title': article_title,
                            'requirement_text': requirement_text,
                            'full_context': self._extract_full_requirement(article_content, match.start()),
                            'type': 'RTS',
                            'policy_area': article_area,
                            'covered': False,  # Initialize as not covered
                            'coverage_details': [],  # Store matching policy sections
                            'similarity_score': 0.0  # Initialize similarity score
                        }
                        self.rts_requirements[article_num].append(requirement)
                
                # Process ITS requirements
                for pattern in its_patterns:
                    matches = re.finditer(pattern, article_content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        its_found += 1
                        requirement_text = match.group(1).strip()
                        print(f"\nFound ITS requirement:")
                        print(f"Text: {requirement_text[:200]}...")
                        
                        requirement = {
                            'article_num': article_num,
                            'article_title': article_title,
                            'requirement_text': requirement_text,
                            'full_context': self._extract_full_requirement(article_content, match.start()),
                            'type': 'ITS',
                            'policy_area': article_area,
                            'covered': False,  # Initialize as not covered
                            'coverage_details': [],  # Store matching policy sections
                            'similarity_score': 0.0  # Initialize similarity score
                        }
                        self.its_requirements[article_num].append(requirement)
            
            # Print summary statistics
            print("\nExtraction Summary:")
            print(f"Articles processed: {articles_processed}")
            print(f"RTS requirements found: {rts_found}")
            print(f"ITS requirements found: {its_found}")
            print("\nRequirements by Policy Area:")
            
            # Aggregate requirements by policy area
            area_stats = defaultdict(lambda: {'rts': 0, 'its': 0})
            for reqs in self.rts_requirements.values():
                for req in reqs:
                    area_stats[req['policy_area']]['rts'] += 1
            for reqs in self.its_requirements.values():
                for req in reqs:
                    area_stats[req['policy_area']]['its'] += 1
            
            # Print area statistics
            for area, stats in area_stats.items():
                print(f"\n{area.replace('_', ' ').title()}:")
                print(f"- RTS Requirements: {stats['rts']}")
                print(f"- ITS Requirements: {stats['its']}")
                print(f"- Total: {stats['rts'] + stats['its']}")
            
            return rts_found + its_found
            
        except Exception as e:
            print(f"Error extracting technical standards: {str(e)}")
            return 0
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
                cleaned_row = [self._clean_cell_content(cell) for cell in row]
                if any(cleaned_row):  # Only keep rows with at least one non-empty cell
                    cleaned_table.append(cleaned_row)
            if not cleaned_table:
                return
            # Create DataFrame
            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
            # Store table data with metadata
            tables_data.append(
                {
                    "data": df,
                    "header": cleaned_table[0],
                    "num_rows": len(df),
                    "num_cols": len(df.columns),
                    "start_page": start_page,
                    "end_page": end_page,
                }
            )
        except Exception as e:
            print(f"Error processing table: {str(e)}")
    def _clean_cell_content(self, cell: str) -> str:
        """Clean individual cell content."""
        if cell is None:
            return ""
        # Convert non-string values to string
        cell = str(cell)
        # Remove extra whitespace and newlines
        cleaned = " ".join(cell.split())
        return cleaned.strip()
    def _remove_table_content_from_text(
        self, text: str, tables: List[List[List[str]]]
    ) -> str:
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
                        cell = cell.strip()
                        table_content.add(cell)
                        table_content.add(" ".join(cell.split()))
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
    def _format_tables_data(self, tables_data: List[Dict]) -> str:
        """Format extracted tables data into readable text."""
        formatted_text = []
        for i, table in enumerate(tables_data, 1):
            formatted_text.append(f"\nTable {i}:")
            formatted_text.append("-" * 40)
            # Convert DataFrame to string with proper formatting
            table_str = table["data"].to_string(index=False)
            formatted_text.append(table_str)
            formatted_text.append("")  # Empty line after table
        return "\n".join(formatted_text)
    def _identify_table_type(self, table_data: pd.DataFrame) -> str:
        """Identify the type of table based on its content and structure."""
        header_text = " ".join(str(col).lower() for col in table_data.columns)
        # Define patterns for different table types
        table_patterns = {
            "requirements": ["requirement", "control", "measure", "standard"],
            "risk_assessment": ["risk", "impact", "likelihood", "score"],
            "technical_standards": [
                "rts",
                "its",
                "technical standard",
                "specification",
            ],
            "compliance": ["compliance", "status", "gap", "assessment"],
            "controls": ["control", "description", "owner", "status"],
        }
        for table_type, patterns in table_patterns.items():
            if any(pattern in header_text for pattern in patterns):
                return table_type
        return "other"
    def _identify_policy_area(self, text: str) -> str:
        """Identify the policy area with enhanced accuracy and logging."""
        try:
            print("\nAnalyzing policy area for text snippet...")
            text = text.lower()
            doc = self.nlp(text[:25000])  # Limit text length for processing
            # Initialize area scores with debug information
            area_scores = defaultdict(
                lambda: {"score": 0.0, "keyword_matches": [], "context_score": 0.0}
            )
            # Get meaningful sentences
            sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 10]
            if not sentences:
                print("Warning: No meaningful sentences found in text")
                return "general"
            print(f"Analyzing {len(sentences)} sentences for policy area matching")
            # First pass: Keyword and semantic matching
            for area, keywords in self.policy_areas.items():
                print(f"\nAnalyzing area: {area}")
                area_data = area_scores[area]
                for keyword in keywords:
                    keyword_doc = self.nlp(keyword)
                    if not keyword_doc.vector_norm:
                        continue
                    for sent in sentences:
                        try:
                            # Calculate similarity
                            similarity = keyword_doc.similarity(sent)
                            if similarity > 0.6:  # Threshold for strong match
                                area_data["score"] += similarity
                                area_data["keyword_matches"].append(
                                    {
                                        "keyword": keyword,
                                        "sentence": sent.text,
                                        "similarity": similarity,
                                    }
                                )
                                print(
                                    f"Strong match found: '{keyword}' in '{sent.text[:100]}...' (similarity: {similarity:.3f})"
                                )
                        except Exception as e:
                            continue
            # Second pass: Context analysis using LLM
            try:
                # Prepare context for top scoring areas
                top_areas = sorted(
                    area_scores.items(), key=lambda x: x[1]["score"], reverse=True
                )[
                    :3
                ]  # Consider top 3 candidates
                if top_areas:
                    context_prompt = [
                        {
                            "role": "system",
                            "content": "Analyze which policy area best matches the given text. Consider regulatory and technical context.",
                        },
                        {
                            "role": "user",
                            "content": f"""
Text: {text[:1000]}
Candidate areas:
{chr(10).join(f'- {area}' for area, _ in top_areas)}
Respond with the most appropriate area name only.""",
                        },
                    ]
                    llm_response = self.llm(context_prompt, max_new_tokens=20)
                    # Extract the generated text from the response
                    if isinstance(llm_response, list) and len(llm_response) > 0:
                        llm_area = llm_response[0]["generated_text"]
                        if isinstance(llm_area, str):
                            llm_area = llm_area.strip().lower()
                        else:
                            llm_area = str(llm_response).strip().lower()
                    else:
                        llm_area = str(llm_response).strip().lower()
                    # Add context score to matching area
                    for area, _ in top_areas:
                        if area.lower() in llm_area or llm_area in area.lower():
                            area_scores[area]["context_score"] = 1.0
                            print(f"LLM confirmed area: {area}")
                            break
            except Exception as e:
                print(f"LLM context analysis failed: {str(e)}")
            # Calculate final scores
            final_scores = {}
            for area, data in area_scores.items():
                # Combine keyword matching score and context score
                keyword_score = data["score"] / max(len(data["keyword_matches"]), 1)
                context_weight = 0.4  # Weight for LLM context analysis
                final_scores[area] = (
                    1 - context_weight
                ) * keyword_score + context_weight * data["context_score"]
            # Get the best matching area
            if final_scores:
                best_area = max(final_scores.items(), key=lambda x: x[1])
                print(
                    f"\nSelected policy area: {best_area[0]} (score: {best_area[1]:.3f})"
                )
                # Print detailed matching information
                area_data = area_scores[best_area[0]]
                if area_data["keyword_matches"]:
                    print("\nKey matches:")
                    for match in area_data["keyword_matches"][:3]:  # Show top 3 matches
                        print(f"- Keyword: '{match['keyword']}'")
                        print(f"  In: '{match['sentence'][:100]}...'")
                        print(f"  Similarity: {match['similarity']:.3f}")
                return best_area[0]
            print("No clear policy area identified, defaulting to 'general'")
            return "general"
        except Exception as e:
            print(f"Error in policy area identification: {str(e)}")
            return "general"
    def _analyze_requirement_coverage(self, requirement: Dict, policy_text: str) -> Dict:
        """Analyze how well a requirement is covered in the policy text."""
        try:
            # Ensure we're working with strings
            if isinstance(requirement['requirement_text'], list):
                req_text = ' '.join(requirement['requirement_text']).lower()
            else:
                req_text = str(requirement['requirement_text']).lower()
            if isinstance(policy_text, list):
                policy_text = ' '.join(policy_text).lower()
            else:
                policy_text = str(policy_text).lower()
            
            # Skip if either text is empty
            if not req_text or not policy_text:
                return {'covered': False, 'similarity_score': 0.0, 'matching_sections': []}
            # Split into sentences for more granular analysis
            policy_sentences = [s.strip() for s in re.split(r'[.!?]+', policy_text) if s.strip()]
            
            # Calculate similarity scores using spaCy
            req_doc = self.nlp(req_text)
            max_similarity = 0.0
            matching_sections = []
            
            print(f"Analyzing requirement: {requirement['requirement_text'][:100]}...")
            
            for sentence in policy_sentences:
                try:
                    # Skip very short sentences
                    if len(str(sentence).split()) < 5:
                        continue
                        
                    sentence_doc = self.nlp(str(sentence))
                    if sentence_doc.vector_norm and req_doc.vector_norm:
                        similarity = req_doc.similarity(sentence_doc)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            print(f"Max similarity found: {max_similarity:.2f}")
                        if similarity > 0.7:  # Adjusted threshold
                            matching_sections.append({
                                'text': sentence,
                                'similarity': similarity
                            })
                except Exception as e:
                    print(f"Error processing sentence: {str(e)}")
                    continue
            
            # Determine coverage based on similarity and matching sections
            #is_covered = max_similarity > 0.7 or len(matching_sections) > 0
            is_covered = max_similarity > 0.8  
            
            if is_covered:
                print("Requirement considered covered.")
            else:
                print("Requirement considered uncovered (gap).")
            
            return {
                'covered': is_covered,
                'similarity_score': max_similarity,
                'matching_sections': sorted(matching_sections, 
                                         key=lambda x: x['similarity'], 
                                         reverse=True)[:3]  # Top 3 matching sections
            }
            
        except Exception as e:
            print(f"Error analyzing requirement coverage: {str(e)}")
            return {'covered': False, 'similarity_score': 0.0, 'matching_sections': []}
    def _extract_full_requirement(self, text: str, start_pos: int) -> str:
        """Extract the full requirement context from the text."""
        # Find the start of the sentence
        sentence_start = text.rfind('.', 0, start_pos) + 1
        if sentence_start == 0:
            sentence_start = text.rfind('\n', 0, start_pos) + 1
        
        # Find the end of the sentence
        sentence_end = text.find('.', start_pos)
        if sentence_end == -1:
            sentence_end = len(text)
        
        # Extract and clean the requirement text
        requirement = text[sentence_start:sentence_end].strip()
        return requirement if requirement else text[max(0, start_pos-100):min(len(text), start_pos+100)]
def main():
    # Initialize analyzer with DORA document
    try:
        dora_path = "CELEX_32022R2554_EN_TXT.pdf"
        analyzer = DORAComplianceAnalyzer(dora_path)
        # Extract technical standards
        print("Extracting technical standards from DORA...")
        analyzer.extract_technical_standards()
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise
    # Define the folder containing policy documents
    policy_folder = "policies"
    # Get all PDF files from the folder
    print(f"Scanning for policy documents in: {policy_folder}")
    try:
        pdf_files = [f for f in Path(policy_folder).glob("**/*.pdf")]
        if not pdf_files:
            print("No PDF files found in the specified folder!")
            return
        print(f"Found {len(pdf_files)} policy documents")
        # Analyze each policy document
        for pdf_path in pdf_files:
            try:
                # Use the filename (without extension) as the policy name
                policy_name = pdf_path.stem.replace("_", " ").title()
                print(f"Analyzing: {policy_name}")
                analyzer.analyze_policy_document(str(pdf_path), policy_name)
            except Exception as e:
                print(f"Error analyzing {pdf_path.name}: {str(e)}")
        # Generate and save gap analysis report
        print("Generating gap analysis report...")
        report = analyzer.generate_gap_analysis_report()
        # Create output folder if it doesn't exist
        output_folder = Path(policy_folder) / "analysis_output"
        output_folder.mkdir(exist_ok=True)
        # Save the report with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_folder / f"dora_gap_analysis_{timestamp}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Gap analysis report has been written to {output_file}")
    except Exception as e:
        print(f"Error processing policy documents: {str(e)}")
if __name__ == "__main__":
    main()
