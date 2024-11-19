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
            self.rts_requirements = defaultdict(list)
            self.its_requirements = defaultdict(list)
            self.policy_coverage = defaultdict(list)

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

    def analyze_policy_document(self, policy_pdf_path: str, policy_name: str):
        """Analyze a policy document and identify gaps."""
        try:
            print(f"\nAnalyzing policy document: {policy_name}")
            
            # Extract text from policy document
            with pdfplumber.open(policy_pdf_path) as pdf:
                policy_text = ""
                for page in pdf.pages:
                    policy_text += page.extract_text() + "\n"
            
            coverage_results = []
            total_requirements = 0
            covered_requirements = 0
            
            # Analyze RTS requirements
            print("\nAnalyzing RTS requirements...")
            for article_num, reqs in self.rts_requirements.items():
                for req in reqs:
                    total_requirements += 1
                    coverage = self._analyze_requirement_coverage(req, policy_text)
                    if coverage['covered']:
                        covered_requirements += 1
                        print(f"\nFound coverage for RTS requirement in Article {article_num}:")
                        print(f"Requirement: {req['requirement_text'][:200]}...")
                        print(f"Similarity score: {coverage['similarity_score']:.2f}")
                    
                    coverage_results.append({
                        'article_num': article_num,
                        'requirement_text': req['requirement_text'],
                        'requirement_type': 'RTS',
                        'covered': coverage['covered'],
                        'similarity_score': coverage['similarity_score']
                    })
            
            # Analyze ITS requirements
            print("\nAnalyzing ITS requirements...")
            for article_num, reqs in self.its_requirements.items():
                for req in reqs:
                    total_requirements += 1
                    coverage = self._analyze_requirement_coverage(req, policy_text)
                    if coverage['covered']:
                        covered_requirements += 1
                        print(f"\nFound coverage for ITS requirement in Article {article_num}:")
                        print(f"Requirement: {req['requirement_text'][:200]}...")
                        print(f"Similarity score: {coverage['similarity_score']:.2f}")
                    
                    coverage_results.append({
                        'article_num': article_num,
                        'requirement_text': req['requirement_text'],
                        'requirement_type': 'ITS',
                        'covered': coverage['covered'],
                        'similarity_score': coverage['similarity_score']
                    })
            
            # Store results
            self.policy_coverage[policy_name] = coverage_results
            
            # Print analysis summary
            print("\nAnalysis Summary:")
            print(f"Total Requirements Analyzed: {total_requirements}")
            print(f"Requirements Covered: {covered_requirements}")
            if total_requirements > 0:
                coverage_rate = (covered_requirements / total_requirements) * 100
                print(f"Coverage Rate: {coverage_rate:.1f}%")
            
            return coverage_results
            
        except Exception as e:
            print(f"Error analyzing policy document: {str(e)}")
            raise

    def generate_gap_analysis_report(self):
        """Generate a comprehensive gap analysis report."""
        try:
            report_sections = []
            
            # Executive Summary
            total_policies = len(self.policy_coverage)
            total_requirements = sum(len(reqs) for reqs in self.policy_coverage.values())
            covered_requirements = sum(
                len([r for r in reqs if r['covered']])
                for reqs in self.policy_coverage.values()
            )
            
            coverage_rate = (covered_requirements / total_requirements * 100) if total_requirements > 0 else 0
            
            report_sections.extend([
                "DORA Gap Analysis Report",
                "=" * 50,
                "\nExecutive Summary",
                "-" * 20,
                f"Total Policies Analyzed: {total_policies}",
                f"Total Requirements: {total_requirements}",
                f"Requirements Covered: {covered_requirements}",
                f"Overall Coverage Rate: {coverage_rate:.1f}%",
                "\nDetailed Analysis",
                "-" * 20
            ])
            
            # Policy-specific analysis
            for policy_name, requirements in self.policy_coverage.items():
                report_sections.extend([
                    f"\nPolicy: {policy_name}",
                    "-" * (len(policy_name) + 8)
                ])
                
                # Group requirements by type
                rts_reqs = [r for r in requirements if r['requirement_type'] == 'RTS']
                its_reqs = [r for r in requirements if r['requirement_type'] == 'ITS']
                
                # RTS requirements analysis
                report_sections.extend([
                    "\nRTS Requirements:",
                    f"Total: {len(rts_reqs)}",
                    f"Covered: {len([r for r in rts_reqs if r['covered']])}",
                    "\nMajor RTS Gaps:"
                ])
                
                for req in [r for r in rts_reqs if not r['covered']][:3]:
                    report_sections.append(
                        f"- Article {req['article_num']}: {req['requirement_text'][:200]}..."
                    )
                
                # ITS requirements analysis
                report_sections.extend([
                    "\nITS Requirements:",
                    f"Total: {len(its_reqs)}",
                    f"Covered: {len([r for r in its_reqs if r['covered']])}",
                    "\nMajor ITS Gaps:"
                ])
                
                for req in [r for r in its_reqs if not r['covered']][:3]:
                    report_sections.append(
                        f"- Article {req['article_num']}: {req['requirement_text'][:200]}..."
                    )
            
            # Recommendations
            report_sections.extend([
                "\nRecommendations",
                "-" * 15,
                "1. Address critical gaps in RTS requirements",
                "2. Implement missing ITS controls",
                "3. Enhance documentation coverage",
                "4. Establish regular compliance monitoring"
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
                        if similarity > 0.6:  # Adjusted threshold
                            matching_sections.append({
                                'text': sentence,
                                'similarity': similarity
                            })
                except Exception as e:
                    print(f"Error processing sentence: {str(e)}")
                    continue
            
            # Determine coverage based on similarity and matching sections
            is_covered = max_similarity > 0.7 or len(matching_sections) > 0
            
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

    def generate_gap_analysis_visualization(self):
        """Generate visualizations for gap analysis."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Prepare data for visualization
            areas = []
            covered = []
            partial = []
            gaps = []

            for area, data in self.policy_coverage.items():
                areas.append(area.replace("_", " ").title())

                total_reqs = len(data)
                covered_reqs = len([r for r in data if r["covered"]])
                partial_reqs = len(
                    [r for r in data if r.get("partial_coverage", False)]
                )
                gap_reqs = total_reqs - covered_reqs - partial_reqs

                covered.append(covered_reqs)
                partial.append(partial_reqs)
                gaps.append(gap_reqs)

            # Create stacked bar chart
            plt.figure(figsize=(15, 8))

            bar_width = 0.35
            index = range(len(areas))

            plt.bar(
                index, covered, bar_width, label="Covered", color="green", alpha=0.7
            )
            plt.bar(
                index,
                partial,
                bar_width,
                bottom=covered,
                label="Partial",
                color="yellow",
                alpha=0.7,
            )
            plt.bar(
                index,
                gaps,
                bar_width,
                bottom=[i + j for i, j in zip(covered, partial)],
                label="Gaps",
                color="red",
                alpha=0.7,
            )

            plt.xlabel("Policy Areas")
            plt.ylabel("Number of Requirements")
            plt.title("DORA Compliance Gap Analysis")
            plt.xticks(index, areas, rotation=45, ha="right")
            plt.legend()

            plt.tight_layout()
            plt.savefig("dora_gap_analysis.png")
            plt.close()

            # Create heatmap of coverage
            coverage_matrix = []
            for area in areas:
                area_data = self.policy_coverage[area.lower().replace(" ", "_")]
                coverage_row = []
                for req in area_data:
                    if req["covered"]:
                        coverage_row.append(1)
                    elif req.get("partial_coverage", False):
                        coverage_row.append(0.5)
                    else:
                        coverage_row.append(0)
                coverage_matrix.append(coverage_row)

            plt.figure(figsize=(12, 8))
            sns.heatmap(
                coverage_matrix,
                xticklabels=False,
                yticklabels=areas,
                cmap="RdYlGn",
                cbar_kws={"label": "Coverage Level"},
            )

            plt.title("DORA Requirements Coverage Heatmap")
            plt.tight_layout()
            plt.savefig("dora_coverage_heatmap.png")
            plt.close()

        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")

    def _assess_implementation_complexity(self, requirement):
        """Assess the implementation complexity of a requirement."""
        try:
            # Prepare analysis prompt
            complexity_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Analyze the implementation complexity of this regulatory requirement.
Consider:
1. Technical complexity
2. Resource requirements
3. Dependencies
4. Timeline implications
5. Operational impact

Rate each factor 1-5 and provide brief justification.<|eot_id|><|start_header_id|>user<|end_header_id|>
Requirement: {requirement['requirement_text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            analysis = self.llm(complexity_prompt, max_new_tokens=200)[0]["generated_text"]

            # Extract scores and calculate weighted complexity
            scores = {
                "technical": self._extract_score(analysis, "Technical complexity"),
                "resources": self._extract_score(analysis, "Resource requirements"),
                "dependencies": self._extract_score(analysis, "Dependencies"),
                "timeline": self._extract_score(analysis, "Timeline"),
                "impact": self._extract_score(analysis, "Operational impact"),
            }

            # Calculate weighted score
            weights = {
                "technical": 0.3,
                "resources": 0.2,
                "dependencies": 0.2,
                "timeline": 0.15,
                "impact": 0.15,
            }

            complexity_score = sum(scores[k] * weights[k] for k in scores)

            return {
                "score": complexity_score,
                "details": scores,
                "analysis": analysis,
                "complexity_level": self._categorize_complexity(complexity_score),
            }

        except Exception as e:
            print(f"Error in complexity assessment: {str(e)}")
            return {"score": 0, "error": str(e)}

    def _calculate_risk_level(self, requirement, complexity_data):
        """Calculate the risk level for a requirement."""
        try:
            # Prepare risk analysis prompt
            risk_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Analyze the risk implications of this regulatory requirement.
Consider:
1. Compliance risk
2. Operational risk
3. Reputational risk
4. Financial risk
5. Security risk

Rate each risk 1-5 and provide justification.<|eot_id|><|start_header_id|>user<|end_header_id|>
Requirement: {requirement['requirement_text']}
Implementation Complexity: {complexity_data['complexity_level']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            risk_analysis = self.llm(risk_prompt, max_new_tokens=200)[0]["generated_text"]

            # Extract risk scores
            risk_scores = {
                "compliance": self._extract_score(risk_analysis, "Compliance risk"),
                "operational": self._extract_score(risk_analysis, "Operational risk"),
                "reputational": self._extract_score(risk_analysis, "Reputational risk"),
                "financial": self._extract_score(risk_analysis, "Financial risk"),
                "security": self._extract_score(risk_analysis, "Security risk"),
            }

            # Calculate weighted risk score
            risk_weights = {
                "compliance": 0.3,
                "operational": 0.2,
                "reputational": 0.15,
                "financial": 0.15,
                "security": 0.2,
            }

            risk_score = sum(risk_scores[k] * risk_weights[k] for k in risk_scores)

            return {
                "score": risk_score,
                "details": risk_scores,
                "analysis": risk_analysis,
                "risk_level": self._categorize_risk(risk_score),
            }

        except Exception as e:
            print(f"Error in risk calculation: {str(e)}")
            return {"score": 0, "error": str(e)}

    def generate_detailed_report(self):
        """Generate a comprehensive analysis report."""
        try:
            report_sections = []

            # Executive Summary
            report_sections.append(self._generate_executive_summary())

            # Detailed Analysis by Policy Area
            report_sections.append(self._generate_policy_area_analysis())

            # Risk and Complexity Assessment
            report_sections.append(self._generate_risk_complexity_analysis())

            # Gap Analysis
            report_sections.append(self._generate_gap_analysis())

            # Implementation Recommendations
            report_sections.append(self._generate_implementation_recommendations())

            # Save report
            full_report = "\n\n".join(report_sections)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            with open(
                f"dora_analysis_report_{timestamp}.md", "w", encoding="utf-8"
            ) as f:
                f.write(full_report)

            return full_report

        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return str(e)

    def _generate_executive_summary(self):
        """Generate executive summary of the analysis."""
        total_reqs = sum(len(reqs) for reqs in self.policy_coverage.values())
        covered_reqs = sum(len([req for req in reqs if req.get("covered", False)]) 
                         for reqs in self.policy_coverage.values())

        summary = [
            "# DORA Compliance Analysis - Executive Summary",
            "",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "## Key Findings",
            "",
            f"- Total Requirements Analyzed: {total_reqs}",
            f"- Requirements Covered: {covered_reqs}",
            f"- Coverage Rate: {(covered_reqs/total_reqs*100):.1f}%",
            "",
            "## DORA Requirements Details",
            "",
            "### RTS Requirements",
            self._format_requirements_details(self.rts_requirements),
            "",
            "### ITS Requirements",
            self._format_requirements_details(self.its_requirements),
            "",
            "## Critical Gaps",
            self._identify_critical_gaps(),
            "",
            "## Risk Summary",
            self._generate_risk_summary(),
            "",
            "## Implementation Priorities",
            self._generate_priority_recommendations(),
        ]

        return "\n".join(summary)

    def _format_requirements_details(self, requirements_dict):
        """Format detailed requirements from DORA articles."""
        formatted = []
        
        for article_num, reqs in sorted(requirements_dict.items()):
            formatted.extend([
                f"\n#### Article {article_num}",
                "```",
                f"Total Requirements: {len(reqs)}",
                "Requirements:",
            ])
            
            for idx, req in enumerate(reqs, 1):
                formatted.extend([
                    f"\n{idx}. Requirement Text:",
                    f"   {req['requirement_text']}",
                    f"   Policy Area: {req['policy_area']}",
                    f"   Coverage Status: {'Covered' if req.get('covered', False) else 'Not Covered'}",
                    f"   Similarity Score: {req.get('similarity_score', 0):.2f}",
                ])
                
                # Add matching sections if available
                if req.get('matching_sections'):
                    formatted.append("   Matching Policy Sections:")
                    for match in req['matching_sections']:
                        formatted.append(f"   - {match['text'][:200]}...")
                
            formatted.append("```\n")
        
        return "\n".join(formatted)

    def _generate_policy_area_analysis(self):
        """Generate detailed analysis by policy area."""
        analysis = ["# Detailed Policy Area Analysis", ""]

        for area, data in self.policy_coverage.items():
            area_title = area.replace("_", " ").title()
            analysis.extend(
                [
                    f"## {area_title}",
                    "",
                    "### Requirements Overview",
                    self._generate_area_requirements_summary(area, data),
                    "",
                    "### Implementation Complexity",
                    self._generate_area_complexity_summary(area, data),
                    "",
                    "### Risk Assessment",
                    self._generate_area_risk_summary(area, data),
                    "",
                    "### Coverage Analysis",
                    self._generate_area_coverage_summary(area, data),
                    "",
                ]
            )

        return "\n".join(analysis)

    def _generate_area_requirements_summary(self, area, data):
        """Generate requirements overview for a policy area."""
        total_reqs = len(data)
        rts_reqs = len([req for req in data if req.get("requirement_type") == "RTS"])
        its_reqs = len([req for req in data if req.get("requirement_type") == "ITS"])

        return "\n".join(
            [
                f"Total Requirements: {total_reqs}",
                f"RTS Requirements: {rts_reqs}",
                f"ITS Requirements: {its_reqs}",
                "",
                "Key Requirements:",
                *[f"- {req['requirement_text'][:200]}..." for req in data[:3]],
            ]
        )

    def _generate_area_complexity_summary(self, area, data):
        """Generate complexity summary for a policy area."""
        complexity_scores = [req.get("similarity_score", 0) for req in data]
        avg_complexity = (
            sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        )

        return "\n".join(
            [
                f"Average Complexity Score: {avg_complexity:.2f}",
                "Complexity Distribution:",
                f"- High Complexity: {len([s for s in complexity_scores if s > 0.7])}",
                f"- Medium Complexity: {len([s for s in complexity_scores if 0.3 < s <= 0.7])}",
                f"- Low Complexity: {len([s for s in complexity_scores if s <= 0.3])}",
            ]
        )

    def _generate_area_risk_summary(self, area, data):
        """Generate risk summary for a policy area."""
        covered = len([req for req in data if req.get("covered", False)])
        total = len(data)
        risk_level = (
            "High"
            if covered / total < 0.6
            else "Medium" if covered / total < 0.8 else "Low"
        )

        return "\n".join(
            [
                f"Risk Level: {risk_level}",
                f"Coverage Rate: {(covered/total*100):.1f}%",
                "Risk Factors:",
                "- Compliance gaps" if covered / total < 0.8 else "",
                (
                    "- Implementation complexity"
                    if any(req.get("similarity_score", 0) > 0.7 for req in data)
                    else ""
                ),
                "- Technical dependencies" if len(data) > 5 else "",
            ]
        )

    def _generate_area_coverage_summary(self, area, data):
        """Generate coverage summary for a policy area."""
        covered = [req for req in data if req.get("covered", False)]
        gaps = [req for req in data if not req.get("covered", False)]

        return "\n".join(
            [
                f"Coverage Statistics:",
                f"- Requirements Covered: {len(covered)}",
                f"- Requirements Not Covered: {len(gaps)}",
                f"- Coverage Percentage: {(len(covered)/len(data)*100):.1f}%",
                "",
                "Major Gaps:" if gaps else "No Major Gaps Identified",
                *[f"- {gap['requirement_text'][:150]}..." for gap in gaps[:3]],
            ]
        )

    def _generate_risk_complexity_analysis(self):
        """Generate risk and complexity analysis."""
        all_requirements = []
        for reqs in self.policy_coverage.values():
            all_requirements.extend(reqs)

        high_risk = [
            req for req in all_requirements if req.get("similarity_score", 0) < 0.3
        ]
        medium_risk = [
            req
            for req in all_requirements
            if 0.3 <= req.get("similarity_score", 0) < 0.7
        ]

        return "\n".join(
            [
                "# Risk and Complexity Analysis",
                "",
                "## High Risk Areas",
                *[f"- {req['requirement_text'][:150]}..." for req in high_risk[:5]],
                "",
                "## Medium Risk Areas",
                *[f"- {req['requirement_text'][:150]}..." for req in medium_risk[:5]],
            ]
        )

    def _generate_gap_analysis(self):
        """Generate gap analysis."""
        gaps_by_area = {}
        for area, reqs in self.policy_coverage.items():
            gaps = [
                req
                for req in reqs
                if not req.get("covered", False)
                and req.get("similarity_score", 0) < 0.3
            ]
            if gaps:
                gaps_by_area[area] = gaps

        return "\n".join(
            [
                "# Gap Analysis",
                "",
                *[
                    f"## {area.replace('_', ' ').title()}\n"
                    + "\n".join(
                        [f"- {gap['requirement_text'][:150]}..." for gap in gaps]
                    )
                    for area, gaps in gaps_by_area.items()
                ],
            ]
        )

    def _generate_implementation_recommendations(self):
        """Generate implementation recommendations."""
        all_gaps = []
        for reqs in self.policy_coverage.values():
            all_gaps.extend([req for req in reqs if not req.get("covered", False)])

        prioritized_gaps = sorted(all_gaps, key=lambda x: x.get("similarity_score", 0))

        return "\n".join(
            [
                "# Implementation Recommendations",
                "",
                "## Immediate Actions",
                *[
                    f"- {gap['requirement_text'][:150]}..."
                    for gap in prioritized_gaps[:3]
                ],
                "",
                "## Short-term Actions",
                *[
                    f"- {gap['requirement_text'][:150]}..."
                    for gap in prioritized_gaps[3:6]
                ],
                "",
                "## Long-term Actions",
                *[
                    f"- {gap['requirement_text'][:150]}..."
                    for gap in prioritized_gaps[6:9]
                ],
            ]
        )

    def _generate_risk_summary(self):
        """Generate risk summary."""
        total_reqs = sum(len(reqs) for reqs in self.policy_coverage.values())
        covered_reqs = sum(
            len([req for req in reqs if req.get("covered", False)])
            for reqs in self.policy_coverage.values()
        )

        risk_level = (
            "High"
            if covered_reqs / total_reqs < 0.6
            else "Medium" if covered_reqs / total_reqs < 0.8 else "Low"
        )

        return "\n".join(
            [
                f"Overall Risk Level: {risk_level}",
                f"Overall Coverage: {(covered_reqs/total_reqs*100):.1f}%",
                "",
                "Key Risk Areas:",
                "- Compliance Risk" if covered_reqs / total_reqs < 0.8 else "",
                (
                    "- Operational Risk"
                    if any(len(reqs) > 10 for reqs in self.policy_coverage.values())
                    else ""
                ),
                (
                    "- Technical Risk"
                    if any(
                        req.get("similarity_score", 0) > 0.7
                        for reqs in self.policy_coverage.values()
                        for req in reqs
                    )
                    else ""
                ),
            ]
        )

    def _generate_priority_recommendations(self):
        """Generate priority recommendations."""
        all_requirements = []
        for reqs in self.policy_coverage.values():
            all_requirements.extend(reqs)

        critical_gaps = [
            req
            for req in all_requirements
            if not req.get("covered", False) and req.get("similarity_score", 0) < 0.3
        ]

        return "\n".join(
            [
                "Priority Recommendations:",
                "",
                "Critical Actions:",
                *[f"- {gap['requirement_text'][:150]}..." for gap in critical_gaps[:3]],
                "",
                "Key Focus Areas:",
                "- Implement missing technical controls",
                "- Enhance documentation coverage",
                "- Establish monitoring mechanisms",
            ]
        )

    def _identify_critical_gaps(self):
        """Identify critical gaps."""
        critical_gaps = []
        for area, reqs in self.policy_coverage.items():
            area_gaps = [
                req
                for req in reqs
                if not req.get("covered", False)
                and req.get("similarity_score", 0) < 0.3
            ]
            if area_gaps:
                critical_gaps.append(f"\n{area.replace('_', ' ').title()}:")
                critical_gaps.extend(
                    [f"- {gap['requirement_text'][:150]}..." for gap in area_gaps[:2]]
                )

        return (
            "\n".join(critical_gaps)
            if critical_gaps
            else "No critical gaps identified."
        )

    def analyze_and_report(self):
        """Main method to run complete analysis and generate reports."""
        try:
            print("Starting comprehensive DORA analysis...")

            # Step 1: Extract and analyze requirements
            self.extract_technical_standards()

            # Step 2: Perform detailed analysis
            analysis_results = self.analyze_requirements_by_policy_area()

            # Step 3: Generate dependency graph
            self.generate_dependency_graph()

            # Step 4: Generate gap analysis visualizations
            self.generate_gap_analysis_visualization()

            # Step 5: Generate detailed report
            report = self.generate_detailed_report()

            # Step 6: Generate implementation roadmap
            roadmap = self.generate_implementation_roadmap(analysis_results)

            print("Analysis complete. Reports and visualizations generated.")

            return {
                "report": report,
                "roadmap": roadmap,
                "analysis_results": analysis_results,
            }

        except Exception as e:
            print(f"Error in analysis and reporting: {str(e)}")
            return None

    def generate_implementation_roadmap(self, analysis_results):
        """Generate implementation roadmap based on analysis."""
        try:
            roadmap = {
                "immediate": [],
                "short_term": [],
                "medium_term": [],
                "long_term": [],
            }

            for area, data in analysis_results.items():
                for req in data["requirements"]:
                    priority = self._determine_implementation_priority(
                        req["analysis"]["complexity_score"],
                        req.get("risk_level", {}).get("score", 0),
                        req.get("coverage", False),
                    )

                    roadmap[priority].append(
                        {
                            "area": area,
                            "requirement": req["text"],
                            "complexity": req["analysis"]["complexity_level"],
                            "risk_level": req.get("risk_level", {}).get(
                                "risk_level", "Unknown"
                            ),
                            "dependencies": req["analysis"]["dependencies"],
                        }
                    )

            return self._format_roadmap(roadmap)

        except Exception as e:
            print(f"Error generating roadmap: {str(e)}")
            return None

    def _format_roadmap(self, roadmap):
        """Format the implementation roadmap as a structured document."""
        formatted = ["# DORA Implementation Roadmap", ""]

        timeframes = {
            "immediate": "0-3 months",
            "short_term": "3-6 months",
            "medium_term": "6-12 months",
            "long_term": "12+ months",
        }

        for phase, items in roadmap.items():
            formatted.extend(
                [
                    f"## {phase.replace('_', ' ').title()} Priority ({timeframes[phase]})",
                    "",
                ]
            )

            for item in items:
                formatted.extend(
                    [
                        f"### {item['area'].replace('_', ' ').title()}",
                        "",
                        f"**Requirement:** {item['requirement'][:200]}...",
                        f"**Complexity:** {item['complexity']}",
                        f"**Risk Level:** {item['risk_level']}",
                        "",
                        "**Dependencies:**",
                    ]
                )

                for dep in item["dependencies"]:
                    formatted.append(f"- {dep}")

                formatted.append("")

        return "\n".join(formatted)

    def _remove_table_content_from_text_enhanced(
        self, text: str, tables: List[List[List[str]]]
    ) -> str:
        """Remove table content from text with enhanced accuracy."""
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

    def _print_analysis_summary(
        self, policy_name: str, coverage_results: List[Dict]
    ) -> None:
        """Print summary of analysis results."""
        total_reqs = len(coverage_results)
        covered_reqs = len([r for r in coverage_results if r["covered"]])

        print(f"\nAnalysis Summary for {policy_name}")
        print("-" * 40)
        print(f"Total Requirements: {total_reqs}")
        print(f"Requirements Covered: {covered_reqs}")
        print(
            f"Coverage Rate: {(covered_reqs/total_reqs*100):.1f}%"
            if total_reqs > 0
            else "Coverage Rate: N/A"
        )

        # Print coverage by requirement type
        rts_reqs = len([r for r in coverage_results if r["requirement_type"] == "RTS"])
        its_reqs = len([r for r in coverage_results if r["requirement_type"] == "ITS"])
        print(f"\nRTS Requirements: {rts_reqs}")
        print(f"ITS Requirements: {its_reqs}")

        # Print major gaps if any
        gaps = [r for r in coverage_results if not r["covered"]]
        if gaps:
            print("\nMajor Gaps Identified:")
            for gap in gaps[:3]:  # Show top 3 gaps
                print(
                    f"- Article {gap['article_num']}: {gap['requirement_text'][:100]}..."
                )

    def _calculate_header_similarity(
        self, header1: List[str], header2: List[str]
    ) -> float:
        """Calculate similarity between two table headers."""
        try:
            # Clean and normalize headers
            def normalize_header(header):
                return [str(col).lower().strip() for col in header]

            norm_header1 = normalize_header(header1)
            norm_header2 = normalize_header(header2)

            # Calculate exact matches
            exact_matches = sum(
                1 for h1, h2 in zip(norm_header1, norm_header2) if h1 == h2
            )

            # Calculate fuzzy matches for non-exact matches
            fuzzy_similarity = 0
            for h1, h2 in zip(norm_header1, norm_header2):
                if h1 != h2:
                    # Use spaCy for semantic similarity
                    doc1 = self.nlp(h1)
                    doc2 = self.nlp(h2)
                    if doc1.vector_norm and doc2.vector_norm:
                        fuzzy_similarity += doc1.similarity(doc2)

            # Combine exact and fuzzy matches
            max_len = max(len(header1), len(header2))
            if max_len == 0:
                return 0.0

            similarity = (exact_matches + 0.5 * fuzzy_similarity) / max_len
            return min(1.0, max(0.0, similarity))

        except Exception as e:
            print(f"Error calculating header similarity: {str(e)}")
            return 0.0

    def print_coverage_analysis(self):
        """Print detailed coverage analysis of requirements."""
        print("\nDORA Requirements Coverage Analysis")
        print("=" * 40)
        
        # RTS Coverage
        total_rts = sum(len(reqs) for reqs in self.rts_requirements.values())
        covered_rts = 0
        for reqs in self.rts_requirements.values():
            for req in reqs:
                if any(coverage.get('covered', False) 
                      for coverage in self.policy_coverage.values()):
                    covered_rts += 1
        
        # ITS Coverage
        total_its = sum(len(reqs) for reqs in self.its_requirements.values())
        covered_its = 0
        for reqs in self.its_requirements.values():
            for req in reqs:
                if any(coverage.get('covered', False) 
                      for coverage in self.policy_coverage.values()):
                    covered_its += 1
        
        print("\nRTS Requirements:")
        print(f"Total: {total_rts}")
        print(f"Covered: {covered_rts}")
        if total_rts > 0:
            print(f"Coverage Rate: {(covered_rts/total_rts*100):.1f}%")
        
        print("\nITS Requirements:")
        print(f"Total: {total_its}")
        print(f"Covered: {covered_its}")
        if total_its > 0:
            print(f"Coverage Rate: {(covered_its/total_its*100):.1f}%")
        
        # Print gaps
        print("\nMajor Gaps:")
        for req_type, reqs_dict in [("RTS", self.rts_requirements), ("ITS", self.its_requirements)]:
            for article_num, reqs in reqs_dict.items():
                for req in reqs:
                    if not any(coverage.get('covered', False) 
                             for coverage in self.policy_coverage.values()):
                        print(f"\n{req_type} Gap in Article {article_num}:")
                        print(f"Requirement: {req['requirement_text'][:200]}...")

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
