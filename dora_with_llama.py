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
            max_length=2048,
            temperature=0.1,  # Lower temperature for more focused analysis
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
        """Extract and clean text and tables from PDF document."""
        print("Extracting content from PDF...")

        full_text = []
        tables_data = []
        current_table = None

        try:
            with pdfplumber.open(self.dora_pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"Processing page {page_num}/{len(pdf.pages)}")

                    # Extract tables from the page
                    tables = page.extract_tables()

                    if tables:
                        for table in tables:
                            if current_table is not None:
                                if len(table[0]) == len(current_table[0]):
                                    current_table.extend(table[1:])
                                    continue
                                else:
                                    self._process_completed_table(
                                        current_table, tables_data
                                    )
                                    current_table = table
                            else:
                                current_table = table
                    elif current_table is not None:
                        self._process_completed_table(current_table, tables_data)
                        current_table = None

                    # Extract text
                    text = page.extract_text()
                    if text:
                        text = self._remove_table_content_from_text(text, tables)
                        full_text.append(text)

                if current_table is not None:
                    self._process_completed_table(current_table, tables_data)

            combined_text = "\n".join(full_text)
            if tables_data:
                combined_text += "\n\nExtracted Tables:\n"
                combined_text += self._format_tables_data(tables_data)

            return self._clean_text(combined_text)

        except Exception as e:
            print(f"Error extracting content from PDF: {str(e)}")
            return ""

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
            prompt = [
                {
                    "role": "system",
                    "content": "Compare the semantic similarity of these two texts on a scale of 0 to 1, considering regulatory and technical context:",
                },
                {
                    "role": "user",
                    "content": f"Text 1: {text1[:1000]}\nText 2: {text2[:1000]}\nProvide only a number between 0 and 1.",
                },
            ]

            llm_response = self.llm(prompt, max_new_tokens=10)
            try:
                llm_sim = float(
                    re.search(
                        r"([0-9]*[.])?[0-9]+", llm_response[0]["generated_text"]
                    ).group()
                )
                llm_sim = max(0.0, min(1.0, llm_sim))
            except:
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
                        table_content.add(cell.strip())

        # Split text into lines and remove those matching table content
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Keep line if it's not in table content
            if line and not any(table_text in line for table_text in table_content):
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
                    llm_area = llm_response[0]["generated_text"].strip().lower()

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

    def extract_technical_standards(self):
        """Extract RTS and ITS requirements with enhanced detection and logging."""
        print("Inside extract_technical_standards method")  # Debug print
        if not hasattr(self, "dora_text"):
            print("Error: dora_text not initialized")
            return
        print("\nStarting technical standards extraction...")

        # More comprehensive patterns for RTS
        rts_patterns = [
            # Direct RTS references
            r"(?:the\s+)?(?:RTS|regulatory\s+technical\s+standards?)\s+(?:shall|should|must|will|to|that|which)\s+([^\.]+\.[^\.]+)",
            r"develop\s+(?:draft\s+)?regulatory\s+technical\s+standards?\s+(?:to|that|which|for|on)\s+([^\.]+\.[^\.]+)",
            r"specify\s+(?:in|through)\s+(?:the\s+)?regulatory\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            r"requirements?\s+(?:shall|should|must|will)\s+be\s+specified\s+in\s+regulatory\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            r"detailed\s+provisions?\s+(?:shall|should|must|will)\s+be\s+laid\s+down\s+in\s+regulatory\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            r"regulatory\s+technical\s+standards?\s+(?:shall|should|must|will)\s+specify[^\.]+([^\.]+\.[^\.]+)",
            r"(?:EBA|ESMA)\s+shall\s+develop\s+(?:draft\s+)?regulatory\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
        ]

        # More comprehensive patterns for ITS
        its_patterns = [
            # Direct ITS references
            r"(?:the\s+)?(?:ITS|implementing\s+technical\s+standards?)\s+(?:shall|should|must|will|to|that|which)\s+([^\.]+\.[^\.]+)",
            r"develop\s+(?:draft\s+)?implementing\s+technical\s+standards?\s+(?:to|that|which|for|on)\s+([^\.]+\.[^\.]+)",
            r"specify\s+(?:in|through)\s+(?:the\s+)?implementing\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            r"implement(?:ed|ing)?\s+through\s+implementing\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            r"format\s+(?:shall|should|must|will)\s+be\s+specified\s+in\s+implementing\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            r"uniform\s+(?:format|templates|procedures|forms)\s+(?:shall|should|must|will)\s+be\s+specified\s+in\s+implementing\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            r"(?:EBA|ESMA)\s+shall\s+develop\s+(?:draft\s+)?implementing\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
        ]

        try:
            # Extract articles with enhanced pattern
            article_pattern = r"Article\s+(\d+[a-z]?)\s*[–-]?\s*([^\n]+)(?:\n|\r\n?)(.*?)(?=Article\s+\d+[a-z]?|$)"
            articles = re.finditer(
                article_pattern, self.dora_text, re.DOTALL | re.IGNORECASE
            )

            print("\nExtracting requirements from articles...")
            articles_processed = 0
            rts_found = 0
            its_found = 0

            for article in articles:
                articles_processed += 1
                article_num = article.group(1)
                article_title = article.group(2).strip()
                article_content = article.group(3).strip()

                print(f"\nProcessing Article {article_num}: {article_title}")

                # Identify policy area for the article
                article_area = self._identify_policy_area(
                    f"{article_title} {article_content}"
                )
                print(f"Identified policy area: {article_area}")

                # Process RTS requirements
                for pattern in rts_patterns:
                    matches = re.finditer(
                        pattern, article_content, re.IGNORECASE | re.MULTILINE
                    )
                    for match in matches:
                        rts_found += 1
                        full_requirement = self._extract_full_requirement(
                            article_content, match.start()
                        )
                        print(f"\nFound RTS requirement in Article {article_num}:")
                        print(f"Requirement text: {match.group(1)[:200]}...")

                        requirement = {
                            "article_num": article_num,
                            "article_title": article_title,
                            "requirement_text": match.group(1).strip(),
                            "full_context": full_requirement,
                            "type": "RTS",
                            "policy_area": article_area,
                            "pattern_matched": pattern,
                        }
                        self.rts_requirements[article_num].append(requirement)

                # Process ITS requirements
                for pattern in its_patterns:
                    matches = re.finditer(
                        pattern, article_content, re.IGNORECASE | re.MULTILINE
                    )
                    for match in matches:
                        its_found += 1
                        full_requirement = self._extract_full_requirement(
                            article_content, match.start()
                        )
                        print(f"\nFound ITS requirement in Article {article_num}:")
                        print(f"Requirement text: {match.group(1)[:200]}...")

                        requirement = {
                            "article_num": article_num,
                            "article_title": article_title,
                            "requirement_text": match.group(1).strip(),
                            "full_context": full_requirement,
                            "type": "ITS",
                            "policy_area": article_area,
                            "pattern_matched": pattern,
                        }
                        self.its_requirements[article_num].append(requirement)

            # Summary statistics
            total_rts = sum(len(reqs) for reqs in self.rts_requirements.values())
            total_its = sum(len(reqs) for reqs in self.its_requirements.values())

            print("\nExtraction Summary:")
            print(f"Articles processed: {articles_processed}")
            print(f"RTS requirements found: {total_rts}")
            print(f"ITS requirements found: {total_its}")
            print(f"Total requirements: {total_rts + total_its}")

            return total_rts + total_its

        except Exception as e:
            print(f"Error in technical standards extraction: {str(e)}")
            return 0

    def analyze_policy_document(self, policy_pdf_path: str, policy_name: str):
        """Analyze a policy document and identify gaps."""
        if not policy_pdf_path or not policy_name:
            raise ValueError("Policy path and name must be provided")

        if not Path(policy_pdf_path).exists():
            raise FileNotFoundError(f"Policy file not found: {policy_pdf_path}")

        print(f"\nAnalyzing policy document: {policy_name}")

        try:
            policy_text = []
            tables_data = []
            current_table = None
            table_header = None
            table_page_start = None

            # Validate PDF file before processing
            if not self._is_valid_pdf(policy_pdf_path):
                raise ValueError(f"Invalid or corrupted PDF file: {policy_pdf_path}")

            with pdfplumber.open(policy_pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"Total pages in document: {total_pages}")

                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"Processing page {page_num}/{total_pages}")

                    try:
                        # Extract tables from the page
                        tables = page.extract_tables()

                        if tables:
                            for table_idx, table in enumerate(tables):
                                # Validate table structure
                                if not self._is_valid_table(table):
                                    print(
                                        f"Skipping invalid table {table_idx + 1} on page {page_num}"
                                    )
                                    continue

                                if not table or not any(
                                    row for row in table if any(cell for cell in row)
                                ):
                                    continue  # Skip empty tables

                                # Process table continuation
                                if current_table is not None:
                                    is_continuation = self._check_table_continuation(
                                        current_table, table, page_num
                                    )

                                    if is_continuation:
                                        print(
                                            f"Detected table continuation on page {page_num}"
                                        )
                                        current_table.extend(
                                            table[1:]
                                        )  # Add rows without header
                                        continue
                                    else:
                                        # Process the completed table
                                        self._process_completed_table(
                                            current_table,
                                            tables_data,
                                            start_page=table_page_start,
                                            end_page=page_num - 1,
                                        )
                                        current_table = table
                                        table_header = table[0]
                                        table_page_start = page_num
                                else:
                                    current_table = table
                                    table_header = table[0]
                                    table_page_start = page_num

                        elif current_table is not None:
                            # Process the completed table
                            self._process_completed_table(
                                current_table,
                                tables_data,
                                start_page=table_page_start,
                                end_page=page_num - 1,
                            )
                            current_table = None
                            table_header = None
                            table_page_start = None

                        # Extract and clean text
                        text = page.extract_text(x_tolerance=2, y_tolerance=2)
                        if text:
                            cleaned_text = (
                                self._remove_table_content_from_text_enhanced(
                                    text, tables
                                )
                            )
                            if cleaned_text.strip():
                                policy_text.append(cleaned_text)

                    except Exception as e:
                        print(f"Error processing page {page_num}: {str(e)}")
                        continue  # Continue with next page

                # Process any remaining table
                if current_table is not None:
                    self._process_completed_table(
                        current_table,
                        tables_data,
                        start_page=table_page_start,
                        end_page=total_pages,
                    )

            # Validate and combine extracted content
            if not policy_text and not tables_data:
                raise ValueError("No content extracted from the document")

            # Combine text and formatted table data
            combined_text = "\n".join(policy_text)
            if tables_data:
                combined_text += "\n\nExtracted Tables:\n"
                combined_text += self._format_tables_data(tables_data)

            policy_text = self._clean_text(combined_text)

            # Analyze requirements
            coverage_results = self._analyze_requirements(policy_text, tables_data)

            # Store results
            self.policy_coverage[policy_name] = coverage_results

            # Generate summary
            self._print_analysis_summary(policy_name, coverage_results)

        except Exception as e:
            print(f"Error processing policy document {policy_name}: {str(e)}")
            self.policy_coverage[policy_name] = []
            raise

    def _is_valid_pdf(self, pdf_path: str) -> bool:
        """Validate PDF file."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages) > 0
        except:
            return False

    def _is_valid_table(self, table: List[List[str]]) -> bool:
        """Validate table structure."""
        if not table or not isinstance(table, list):
            return False

        # Check if all rows have the same number of columns
        if not all(isinstance(row, list) for row in table):
            return False

        num_cols = len(table[0])
        return all(len(row) == num_cols for row in table)

    def _check_table_continuation(
        self, current_table: List[List[str]], new_table: List[List[str]], page_num: int
    ) -> bool:
        """Check if new table is a continuation of current table."""
        try:
            # Exact column match
            if len(new_table[0]) == len(current_table[0]):
                return True

            # Similar column count with header similarity
            if abs(len(new_table[0]) - len(current_table[0])) <= 1:
                header_similarity = self._calculate_header_similarity(
                    current_table[0], new_table[0]
                )
                return header_similarity > 0.8

            return False
        except Exception as e:
            print(f"Error checking table continuation on page {page_num}: {str(e)}")
            return False

    def _analyze_requirements(
        self, policy_text: str, tables_data: List[Dict]
    ) -> List[Dict]:
        """Analyze requirements against policy content with detailed coverage tracking."""
        coverage_results = []
        all_requirements = []

        print("\nStarting Detailed Requirements Analysis...")

        # Collect all requirements with detailed logging
        for article_num, requirements in self.rts_requirements.items():
            print(f"\nProcessing RTS requirements from Article {article_num}")
            for req in requirements:
                print(f"Found requirement: {req['requirement_text'][:100]}...")
                all_requirements.append(
                    {
                        "article_num": article_num,
                        "requirement_type": "RTS",
                        "requirement_text": req["requirement_text"],
                        "full_context": req.get("full_context", ""),
                        "policy_area": req["policy_area"],
                    }
                )

        for article_num, requirements in self.its_requirements.items():
            print(f"\nProcessing ITS requirements from Article {article_num}")
            for req in requirements:
                print(f"Found requirement: {req['requirement_text'][:100]}...")
                all_requirements.append(
                    {
                        "article_num": article_num,
                        "requirement_type": "ITS",
                        "requirement_text": req["requirement_text"],
                        "full_context": req.get("full_context", ""),
                        "policy_area": req["policy_area"],
                    }
                )

        total_reqs = len(all_requirements)
        print(f"\nTotal requirements identified: {total_reqs}")

        if total_reqs == 0:
            print(
                "WARNING: No requirements were extracted. Check the DORA text extraction."
            )
            return coverage_results

        for idx, req in enumerate(all_requirements, 1):
            print(f"\nAnalyzing requirement {idx}/{total_reqs}")
            print(f"Article: {req['article_num']}, Type: {req['requirement_type']}")
            print(f"Text: {req['requirement_text'][:200]}...")

            try:
                requirement_text = req["full_context"] or req["requirement_text"]

                # Calculate similarity scores
                text_similarity = self._calculate_similarity(
                    requirement_text, policy_text
                )
                print(f"Text similarity score: {text_similarity:.3f}")

                table_similarity = 0.0
                if tables_data:
                    table_text = self._format_tables_data(tables_data)
                    table_similarity = self._calculate_similarity(
                        requirement_text, table_text
                    )
                    print(f"Table similarity score: {table_similarity:.3f}")

                similarity_score = max(text_similarity, table_similarity)

                # Use LLM for context analysis
                context_prompt = [
                    {
                        "role": "system",
                        "content": "Analyze if the requirement is adequately addressed in the policy text. Consider regulatory compliance context.",
                    },
                    {
                        "role": "user",
                        "content": f"Requirement: {requirement_text[:500]}\nPolicy Text: {policy_text[:500]}\nRespond with only a number between 0 and 1.",
                    },
                ]

                try:
                    llm_response = self.llm(context_prompt, max_new_tokens=10)
                    context_score = float(
                        re.search(
                            r"([0-9]*[.])?[0-9]+", llm_response[0]["generated_text"]
                        ).group()
                    )
                    context_score = max(0.0, min(1.0, context_score))
                    print(f"LLM context score: {context_score:.3f}")
                except Exception as e:
                    print(f"LLM analysis failed: {str(e)}")
                    context_score = 0.0

                # Combine scores with weights
                final_score = (0.6 * similarity_score) + (0.4 * context_score)
                is_covered = final_score > 0.35

                coverage_results.append(
                    {
                        "article_num": req["article_num"],
                        "requirement_type": req["requirement_type"],
                        "requirement_text": req["requirement_text"],
                        "full_context": req.get("full_context", ""),
                        "covered": is_covered,
                        "similarity_score": final_score,
                        "text_similarity": text_similarity,
                        "table_similarity": table_similarity,
                        "context_score": context_score,
                        "policy_area": req["policy_area"],
                    }
                )

                print(
                    f"Final score: {final_score:.3f} - {'Covered' if is_covered else 'Not covered'}"
                )

            except Exception as e:
                print(f"Error analyzing requirement: {str(e)}")
                coverage_results.append(
                    {
                        "article_num": req["article_num"],
                        "requirement_type": req["requirement_type"],
                        "requirement_text": req["requirement_text"],
                        "full_context": req.get("full_context", ""),
                        "covered": False,
                        "similarity_score": 0.0,
                        "error": str(e),
                        "policy_area": req["policy_area"],
                    }
                )

        return coverage_results

    def generate_gap_analysis_report(self):
        """Generate a comprehensive analysis report showing both coverage and gaps."""
        output = []

        output.append("DORA Compliance Analysis Report")
        output.append("=" * 50 + "\n")

        total_requirements = 0
        total_covered = 0
        total_gaps = 0

        print("\nGenerating Comprehensive Analysis Report")

        for policy_name, coverage in self.policy_coverage.items():
            covered_items = [item for item in coverage if item["covered"]]
            gap_items = [item for item in coverage if not item["covered"]]

            print(f"\nPolicy: {policy_name}")
            print(f"Total requirements: {len(coverage)}")
            print(f"Covered requirements: {len(covered_items)}")
            print(f"Gaps identified: {len(gap_items)}")

            output.append(f"\nPolicy Document: {policy_name}")
            output.append("-" * 30)

            # Group by policy area
            area_coverage = defaultdict(lambda: {"covered": [], "gaps": []})
            for item in coverage:
                if item["covered"]:
                    area_coverage[item["policy_area"]]["covered"].append(item)
                else:
                    area_coverage[item["policy_area"]]["gaps"].append(item)

            for area, items in area_coverage.items():
                output.append(f"\nPolicy Area: {area.replace('_', ' ').title()}")
                output.append(f"Coverage Statistics:")
                output.append(f"- Requirements Covered: {len(items['covered'])}")
                output.append(f"- Gaps Identified: {len(items['gaps'])}")

                if items["covered"]:
                    output.append("\nCovered Requirements:")
                    for item in items["covered"]:
                        output.append(f"\nArticle {item['article_num']}:")
                        output.append(f"Type: {item['requirement_type']}")
                        output.append(
                            f"Requirement: {item['requirement_text'][:200]}..."
                        )
                        output.append(f"Coverage Score: {item['similarity_score']:.2f}")
                        if "text_similarity" in item:
                            output.append(
                                f"- Text Similarity: {item['text_similarity']:.2f}"
                            )
                            output.append(
                                f"- Table Similarity: {item['table_similarity']:.2f}"
                            )
                            output.append(
                                f"- Context Score: {item['context_score']:.2f}"
                            )

                if items["gaps"]:
                    output.append("\nGaps Identified:")
                    for item in items["gaps"]:
                        output.append(f"\nArticle {item['article_num']}:")
                        output.append(f"Type: {item['requirement_type']}")
                        output.append(
                            f"Requirement: {item['requirement_text'][:200]}..."
                        )
                        output.append(f"Coverage Score: {item['similarity_score']:.2f}")
                        if "text_similarity" in item:
                            output.append(
                                f"- Text Similarity: {item['text_similarity']:.2f}"
                            )
                            output.append(
                                f"- Table Similarity: {item['table_similarity']:.2f}"
                            )
                            output.append(
                                f"- Context Score: {item['context_score']:.2f}"
                            )

                output.append("-" * 30)

            total_requirements += len(coverage)
            total_covered += len(covered_items)
            total_gaps += len(gap_items)

        # Overall statistics
        output.insert(2, "\nOverall Analysis Statistics:")
        output.insert(3, f"Total Requirements Analyzed: {total_requirements}")
        output.insert(4, f"Requirements Covered: {total_covered}")
        output.insert(5, f"Gaps Identified: {total_gaps}")

        if total_requirements > 0:
            coverage_percentage = total_covered / total_requirements * 100
            output.insert(6, f"Overall Coverage: {coverage_percentage:.1f}%\n")
        else:
            output.insert(6, "Overall Coverage: N/A (no requirements analyzed)\n")

        return "\n".join(output)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize the extracted text."""
        try:
            # Remove extra whitespace and newlines
            cleaned_text = " ".join(text.split())
            return cleaned_text
        except Exception as e:
            print(f"Error cleaning text: {str(e)}")
            return text

    def analyze_requirements_by_policy_area(self):
        """Perform detailed analysis of requirements by policy area."""
        print("\nPerforming detailed policy area analysis...")

        analysis_results = defaultdict(
            lambda: {
                "requirements": [],
                "dependencies": set(),
                "coverage_metrics": {"total": 0, "covered": 0, "partial": 0, "gaps": 0},
                "risk_level": None,
                "implementation_complexity": None,
            }
        )

        # Analyze each requirement
        for article_num, reqs in self.rts_requirements.items():
            for req in reqs:
                area = req["policy_area"]
                analysis_results[area]["requirements"].append(
                    {
                        "type": "RTS",
                        "article": article_num,
                        "text": req["requirement_text"],
                        "context": req.get("full_context", ""),
                        "analysis": self._analyze_single_requirement(req),
                    }
                )

        for article_num, reqs in self.its_requirements.items():
            for req in reqs:
                area = req["policy_area"]
                analysis_results[area]["requirements"].append(
                    {
                        "type": "ITS",
                        "article": article_num,
                        "text": req["requirement_text"],
                        "context": req.get("full_context", ""),
                        "analysis": self._analyze_single_requirement(req),
                    }
                )

        # Calculate metrics and assess risk for each area
        for area, data in analysis_results.items():
            self._calculate_area_metrics(area, data)
            self._assess_area_risk(area, data)
            self._determine_implementation_complexity(area, data)

        return analysis_results

    def _analyze_single_requirement(self, requirement):
        """Analyze a single requirement in detail."""
        try:
            # Prepare requirement text for analysis
            req_text = requirement.get("full_context", requirement["requirement_text"])

            # Use LLM for detailed analysis
            analysis_prompt = [
                {
                    "role": "system",
                    "content": "Analyze this regulatory requirement and provide structured insights:",
                },
                {
                    "role": "user",
                    "content": f"Requirement: {req_text}\n\nProvide analysis of:\n1. Key controls needed\n2. Technical implications\n3. Implementation challenges",
                },
            ]

            llm_response = self.llm(analysis_prompt, max_new_tokens=200)
            analysis_text = llm_response[0]["generated_text"]

            # Extract key components from analysis
            controls = re.findall(
                r"Key controls:(.*?)(?=Technical implications:|$)",
                analysis_text,
                re.DOTALL,
            )
            technical = re.findall(
                r"Technical implications:(.*?)(?=Implementation challenges:|$)",
                analysis_text,
                re.DOTALL,
            )
            challenges = re.findall(
                r"Implementation challenges:(.*?)$", analysis_text, re.DOTALL
            )

            return {
                "key_controls": controls[0].strip() if controls else "",
                "technical_implications": technical[0].strip() if technical else "",
                "implementation_challenges": (
                    challenges[0].strip() if challenges else ""
                ),
                "complexity_score": self._calculate_complexity_score(analysis_text),
                "dependencies": self._identify_dependencies(req_text),
            }

        except Exception as e:
            print(f"Error in requirement analysis: {str(e)}")
            return {"error": str(e), "complexity_score": 0, "dependencies": []}

    def _identify_dependencies(self, requirement_text):
        """Identify dependencies between requirements."""
        dependencies = []

        # Look for references to other articles
        article_refs = re.findall(r"Article\s+(\d+[a-z]?)", requirement_text)

        # Look for related technical standards
        for article_num in article_refs:
            # Check RTS requirements
            if article_num in self.rts_requirements:
                dependencies.extend(
                    [
                        {
                            "type": "RTS",
                            "article": article_num,
                            "requirement": req["requirement_text"][:100],
                        }
                        for req in self.rts_requirements[article_num]
                    ]
                )

            # Check ITS requirements
            if article_num in self.its_requirements:
                dependencies.extend(
                    [
                        {
                            "type": "ITS",
                            "article": article_num,
                            "requirement": req["requirement_text"][:100],
                        }
                        for req in self.its_requirements[article_num]
                    ]
                )

        return dependencies

    def generate_dependency_graph(self):
        """Generate a visualization of requirement dependencies."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            G = nx.DiGraph()

            # Add nodes for each requirement
            for article_num, reqs in self.rts_requirements.items():
                for req in reqs:
                    node_id = f"RTS-{article_num}"
                    G.add_node(node_id, type="RTS", area=req["policy_area"])

            for article_num, reqs in self.its_requirements.items():
                for req in reqs:
                    node_id = f"ITS-{article_num}"
                    G.add_node(node_id, type="ITS", area=req["policy_area"])

            # Add edges for dependencies
            for article_num, reqs in self.rts_requirements.items():
                for req in reqs:
                    source = f"RTS-{article_num}"
                    for dep in self._identify_dependencies(req["requirement_text"]):
                        target = f"{dep['type']}-{dep['article']}"
                        if target in G:
                            G.add_edge(source, target)

            # Create visualization
            plt.figure(figsize=(15, 10))
            pos = nx.spring_layout(G)

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1000)

            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True)

            # Add labels
            nx.draw_networkx_labels(G, pos)

            plt.title("DORA Requirements Dependency Graph")
            plt.axis("off")

            # Save the graph
            plt.savefig("dora_dependencies.png")
            plt.close()

            return G

        except Exception as e:
            print(f"Error generating dependency graph: {str(e)}")
            return None

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
            complexity_prompt = [
                {
                    "role": "system",
                    "content": """
                Analyze the implementation complexity of this regulatory requirement.
                Consider:
                1. Technical complexity
                2. Resource requirements
                3. Dependencies
                4. Timeline implications
                5. Operational impact
                
                Rate each factor 1-5 and provide brief justification.
                """,
                },
                {
                    "role": "user",
                    "content": f"Requirement: {requirement['requirement_text']}",
                },
            ]

            analysis = self.llm(complexity_prompt, max_new_tokens=200)[0][
                "generated_text"
            ]

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
            risk_prompt = [
                {
                    "role": "system",
                    "content": """
                Analyze the risk implications of this regulatory requirement.
                Consider:
                1. Compliance risk
                2. Operational risk
                3. Reputational risk
                4. Financial risk
                5. Security risk
                
                Rate each risk 1-5 and provide justification.
                """,
                },
                {
                    "role": "user",
                    "content": f"""
                Requirement: {requirement['requirement_text']}
                Implementation Complexity: {complexity_data['complexity_level']}
                """,
                },
            ]

            risk_analysis = self.llm(risk_prompt, max_new_tokens=200)[0][
                "generated_text"
            ]

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

        covered_reqs = sum(
            len([req for req in reqs if req.get("covered", False)])
            for reqs in self.policy_coverage.values()
        )

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
