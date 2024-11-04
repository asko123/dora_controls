import PyPDF2
import re
from collections import defaultdict

class DORARequirementExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = self._extract_and_clean_text()
        
    def _extract_and_clean_text(self):
        """Extract text from PDF document and clean it."""
        print("Extracting text from PDF...")  # Debug print
        with open(self.pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
            
            # Clean the extracted text
            text = self._clean_text(text)
            print(f"Extracted {len(text)} characters")  # Debug print
            return text

    def _clean_text(self, text):
        """Clean the extracted text by fixing common issues."""
        # Fix common broken words
        common_fixes = {
            r'arrang\s*ements': 'arrangements',
            r'require\s*ments': 'requirements',
            r'manage\s*ment': 'management',
            r'frame\s*work': 'framework',
            r'cyber\s*security': 'cybersecurity',
            r'guide\s*lines': 'guidelines',
            r'pro\s*cedures': 'procedures',
            r'tech\s*nical': 'technical',
            r'opera\s*tional': 'operational',
            r'infor\s*mation': 'information',
            r'sys\s*tems': 'systems',
            r'pro\s*viders': 'providers',
            r'ser\s*vices': 'services',
            r'secur\s*ity': 'security',
            r'organ\s*isation': 'organisation',
            r'author\s*ities': 'authorities',
            r'assess\s*ment': 'assessment',
            r'monitor\s*ing': 'monitoring',
            r'report\s*ing': 'reporting',
            r'test\s*ing': 'testing',
            r'con\s*trol': 'control',
            r'resil\s*ience': 'resilience',
            r'third\s*-\s*party': 'third-party'
        }
        
        # Apply fixes
        for pattern, replacement in common_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()

    def _extract_articles(self):
        """Extract individual articles from the text."""
        print("Extracting articles...")  # Debug print
        # Updated pattern to better match DORA document structure
        article_pattern = r"Article\s+(\d+)\s*\n([^\n]+)\n(.*?)(?=Article\s+\d+|\Z)"
        articles = re.finditer(article_pattern, self.text, re.DOTALL)
        extracted_articles = [(match.group(1), match.group(2).strip(), match.group(3).strip()) 
                            for match in articles]
        print(f"Found {len(extracted_articles)} articles")  # Debug print
        return extracted_articles

    def _identify_requirements(self, text):
        """Identify requirements from legal text."""
        requirement_patterns = [
            r"shall\s+([^\.]+\.[^\.]+)",
            r"must\s+([^\.]+\.[^\.]+)",
            r"required\s+to\s+([^\.]+\.[^\.]+)",
            r"ensure\s+that\s+([^\.]+\.[^\.]+)",
            r"implement\s+([^\.]+\.[^\.]+)",
            r"establish\s+([^\.]+\.[^\.]+)",
            r"maintain\s+([^\.]+\.[^\.]+)",
            r"provide\s+([^\.]+\.[^\.]+)",
            r"develop\s+([^\.]+\.[^\.]+)",
            r"set\s+up\s+([^\.]+\.[^\.]+)"
        ]
        
        requirements = []
        for pattern in requirement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                requirement = match.group(1).strip()
                if len(requirement) > 10:  # Filter out very short matches
                    requirements.append(requirement)
        
        return requirements

    def _generate_technical_controls(self, requirement):
        """Generate technical controls based on requirement text."""
        controls = defaultdict(list)
        
        # Risk Management Controls
        if any(word in requirement.lower() for word in ['risk', 'assess', 'evaluate']):
            controls['Risk Management'].extend([
                'Implement automated risk assessment tools',
                'Deploy continuous monitoring systems',
                'Establish risk metrics and KRIs',
                'Develop risk reporting dashboards'
            ])

        # Security Controls
        if any(word in requirement.lower() for word in ['security', 'protect', 'safeguard']):
            controls['Security Controls'].extend([
                'Deploy SIEM solution',
                'Implement IDS/IPS systems',
                'Enable security logging and monitoring',
                'Configure security baseline controls'
            ])

        # Incident Management
        if any(word in requirement.lower() for word in ['incident', 'breach', 'response']):
            controls['Incident Management'].extend([
                'Deploy incident management platform',
                'Implement automated incident response',
                'Configure alert management system',
                'Establish incident tracking and metrics'
            ])

        # If no specific controls were identified, add general controls
        if not controls:
            controls['General Controls'].extend([
                'Implement automated compliance monitoring',
                'Deploy relevant control systems',
                'Configure monitoring and alerting',
                'Establish measurement and reporting'
            ])

        return dict(controls)

    def process_document(self):
        """Process the entire document and extract requirements."""
        articles = self._extract_articles()
        output = []

        for article_num, article_title, article_text in articles:
            print(f"Processing Article {article_num}")  # Debug print
            output.append(f"\nArticle {article_num}: {article_title}")
            
            requirements = self._identify_requirements(article_text)
            if requirements:
                for req in requirements:
                    output.append(f"\nLegal Requirement:")
                    output.append(f"{req}")
                    output.append("\nTechnical Implementation:")
                    
                    controls = self._generate_technical_controls(req)
                    for control_type, control_list in controls.items():
                        output.append(f"\n{control_type}:")
                        for control in control_list:
                            output.append(f"- {control}")
                    output.append("\n" + "-"*80)
            else:
                output.append("\nNo specific technical requirements identified.")
                output.append("\n" + "-"*80)

        return "\n".join(output)

def main():
    pdf_path = '/Users/Tawfiq/Desktop/gpt-pilot-backup-0-2-7-73a8c223/workspace/WorkShop/CELEX_32022R2554_EN_TXT.pdf'
    output_file = 'dora_requirements_detailed.txt'
    
    print("Starting DORA document processing...")
    try:
        extractor = DORARequirementExtractor(pdf_path)
        content = extractor.process_document()
        
        if not content.strip():
            print("Warning: No content generated!")
            return
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Requirements have been written to {output_file}")
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    main()
