import PyPDF2
import re
import spacy
from collections import defaultdict
import pandas as pd

class DORAComplianceAnalyzer:
    def __init__(self, dora_pdf_path):
        self.dora_pdf_path = dora_pdf_path
        self.nlp = spacy.load("en_core_web_sm")
        self.dora_text = self._extract_and_clean_text()
        self.rts_requirements = defaultdict(list)
        self.its_requirements = defaultdict(list)
        self.policy_coverage = defaultdict(list)
        
    def _extract_and_clean_text(self):
        """Extract and clean text from PDF document."""
        with open(self.dora_pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
        return self._clean_text(text)

    def _clean_text(self, text):
        """Clean the extracted text."""
        # Fix common broken words and formatting issues
        common_fixes = {
            r'regulat\s*ory': 'regulatory',
            r'tech\s*nical': 'technical',
            r'stand\s*ards': 'standards',
            r'require\s*ments': 'requirements',
            r'implement\s*ing': 'implementing',
            r'frame\s*work': 'framework',
            r'manage\s*ment': 'management'
        }
        
        for pattern, replacement in common_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()

    def extract_technical_standards(self):
        """Extract RTS and ITS requirements from DORA text."""
        # Pattern for identifying RTS and ITS references
        rts_pattern = r"regulatory\s+technical\s+standards?\s*(?:[\w\s,]+?)(?:shall|should|must)\s+([^\.]+)"
        its_pattern = r"implementing\s+technical\s+standards?\s*(?:[\w\s,]+?)(?:shall|should|must)\s+([^\.]+)"
        
        # Find all articles
        article_pattern = r"Article\s+(\d+)\s*([^\n]+)\n(.*?)(?=Article\s+\d+|\Z)"
        articles = re.finditer(article_pattern, self.dora_text, re.DOTALL)
        
        for article in articles:
            article_num = article.group(1)
            article_title = article.group(2).strip()
            article_content = article.group(3).strip()
            
            # Extract RTS requirements
            rts_matches = re.finditer(rts_pattern, article_content, re.IGNORECASE | re.DOTALL)
            for match in rts_matches:
                requirement = {
                    'article_num': article_num,
                    'article_title': article_title,
                    'requirement_text': match.group(1).strip(),
                    'type': 'RTS'
                }
                self.rts_requirements[article_num].append(requirement)
            
            # Extract ITS requirements
            its_matches = re.finditer(its_pattern, article_content, re.IGNORECASE | re.DOTALL)
            for match in its_matches:
                requirement = {
                    'article_num': article_num,
                    'article_title': article_title,
                    'requirement_text': match.group(1).strip(),
                    'type': 'ITS'
                }
                self.its_requirements[article_num].append(requirement)

    def analyze_policy_document(self, policy_pdf_path, policy_name):
        """Analyze a policy document for coverage of RTS and ITS requirements."""
        # Extract text from policy document
        with open(policy_pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            policy_text = ""
            for page in reader.pages:
                if page.extract_text():
                    policy_text += page.extract_text() + "\n"
        
        policy_text = self._clean_text(policy_text)
        
        # Check coverage for each requirement
        for article_num, requirements in self.rts_requirements.items():
            for req in requirements:
                # Use NLP to check if requirement is covered in policy
                requirement_doc = self.nlp(req['requirement_text'])
                policy_doc = self.nlp(policy_text)
                
                # Calculate similarity score
                similarity_score = requirement_doc.similarity(policy_doc)
                
                self.policy_coverage[policy_name].append({
                    'article_num': article_num,
                    'requirement_type': 'RTS',
                    'requirement_text': req['requirement_text'],
                    'covered': similarity_score > 0.7,
                    'similarity_score': similarity_score
                })
        
        # Repeat for ITS requirements
        for article_num, requirements in self.its_requirements.items():
            for req in requirements:
                requirement_doc = self.nlp(req['requirement_text'])
                policy_doc = self.nlp(policy_text)
                similarity_score = requirement_doc.similarity(policy_doc)
                
                self.policy_coverage[policy_name].append({
                    'article_num': article_num,
                    'requirement_type': 'ITS',
                    'requirement_text': req['requirement_text'],
                    'covered': similarity_score > 0.7,
                    'similarity_score': similarity_score
                })

    def generate_gap_analysis_report(self):
        """Generate a detailed gap analysis report."""
        output = []
        
        output.append("DORA Compliance Gap Analysis Report")
        output.append("=" * 50 + "\n")
        
        # Analyze gaps for each policy
        for policy_name, coverage in self.policy_coverage.items():
            output.append(f"\nPolicy Document: {policy_name}")
            output.append("-" * 30 + "\n")
            
            # Group by article number
            by_article = defaultdict(list)
            for item in coverage:
                by_article[item['article_num']].append(item)
            
            # Report gaps by article
            for article_num, requirements in by_article.items():
                output.append(f"\nArticle {article_num}:")
                
                # Report uncovered requirements
                uncovered = [req for req in requirements if not req['covered']]
                if uncovered:
                    output.append("\nGaps Identified:")
                    for req in uncovered:
                        output.append(f"\n{req['requirement_type']} Requirement:")
                        output.append(f"- {req['requirement_text']}")
                        output.append(f"- Similarity Score: {req['similarity_score']:.2f}")
                
                # Report covered requirements
                covered = [req for req in requirements if req['covered']]
                if covered:
                    output.append("\nCovered Requirements:")
                    for req in covered:
                        output.append(f"\n{req['requirement_type']} Requirement:")
                        output.append(f"- {req['requirement_text']}")
                        output.append(f"- Similarity Score: {req['similarity_score']:.2f}")
            
            output.append("\n" + "=" * 50 + "\n")
        
        return "\n".join(output)

def main():
    # Initialize analyzer with DORA document
    dora_path = 'CELEX_32022R2554_EN_TXT.pdf'
    analyzer = DORAComplianceAnalyzer(dora_path)
    
    # Extract technical standards
    print("Extracting technical standards from DORA...")
    analyzer.extract_technical_standards()
    
    # Analyze policy documents
    policy_documents = {
        "Risk Management Policy": "path/to/risk_policy.pdf",
        "Security Policy": "path/to/security_policy.pdf",
        # Add more policy documents as needed
    }
    
    print("Analyzing policy documents...")
    for policy_name, policy_path in policy_documents.items():
        try:
            analyzer.analyze_policy_document(policy_path, policy_name)
        except Exception as e:
            print(f"Error analyzing {policy_name}: {str(e)}")
    
    # Generate and save gap analysis report
    print("Generating gap analysis report...")
    report = analyzer.generate_gap_analysis_report()
    
    with open('dora_gap_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Gap analysis report has been written to dora_gap_analysis.txt")

if __name__ == "__main__":
    main()
