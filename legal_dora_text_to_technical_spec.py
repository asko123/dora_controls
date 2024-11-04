import PyPDF2
import re
import spacy
from collections import defaultdict

class DORARequirementExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = self._extract_and_clean_text()
        self.nlp = spacy.load("en_core_web_sm")
        
    def _extract_and_clean_text(self):
        """Extract and clean text from PDF document."""
        with open(self.pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
        return self._clean_text(text)

    def _clean_text(self, text):
        """Clean the extracted text."""
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
        for pattern, replacement in common_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text.strip()

    def _extract_technical_requirements(self, legal_text):
        """
        Extract technical requirements using NLP analysis.
        """
        doc = self.nlp(legal_text)
        
        # Initialize requirement components
        actions = []
        systems = []
        specifications = []
        timeframes = []
        
        # Extract key components using dependency parsing
        for token in doc:
            if token.dep_ in ['ROOT', 'VERB'] and token.pos_ == 'VERB':
                actions.append(token.text)
            if token.dep_ == 'dobj' and token.pos_ == 'NOUN':
                systems.append(token.text)
            if token.dep_ == 'pobj' and token.pos_ == 'NOUN':
                specifications.append(token.text)
            if token.ent_type_ in ['TIME', 'DATE']:
                timeframes.append(token.text)

        return {
            'actions': list(set(actions)),
            'systems': list(set(systems)),
            'specifications': list(set(specifications)),
            'timeframes': list(set(timeframes))
        }

    def _generate_technical_controls(self, requirement_components):
        """Generate specific technical controls based on requirement components."""
        technical_controls = {
            'Implementation Requirements': [],
            'Technical Specifications': [],
            'Monitoring Requirements': [],
            'Documentation Requirements': []
        }

        # Map actions to technical implementations
        action_mappings = {
            'implement': [
                'Deploy automated solution with following specifications:',
                'Establish implementation timeline and milestones',
                'Configure system according to security baseline'
            ],
            'monitor': [
                'Set up real-time monitoring system',
                'Configure alerting thresholds',
                'Implement automated reporting'
            ],
            'report': [
                'Develop automated reporting mechanism',
                'Configure dashboard with key metrics',
                'Implement incident tracking system'
            ],
            'assess': [
                'Deploy risk assessment framework',
                'Implement continuous evaluation system',
                'Configure automated testing tools'
            ]
        }

        # Generate specific technical requirements based on components
        for action in requirement_components['actions']:
            if action.lower() in action_mappings:
                technical_controls['Implementation Requirements'].extend(action_mappings[action.lower()])

        # Add system-specific controls
        for system in requirement_components['systems']:
            technical_controls['Technical Specifications'].append(f"System Component: {system}")
            technical_controls['Technical Specifications'].extend([
                f"- Implement access control mechanisms",
                f"- Configure monitoring and logging",
                f"- Deploy backup and recovery procedures",
                f"- Establish security baseline"
            ])

        # Add monitoring requirements
        technical_controls['Monitoring Requirements'].extend([
            "Configure continuous monitoring:",
            "- Set up automated data collection",
            "- Implement real-time alerting",
            "- Deploy performance metrics tracking",
            "- Establish compliance monitoring"
        ])

        # Add documentation requirements
        technical_controls['Documentation Requirements'].extend([
            "Required Documentation:",
            "- System architecture diagrams",
            "- Configuration specifications",
            "- Security controls documentation",
            "- Operational procedures"
        ])

        return technical_controls

    def process_article(self, article_text):
        """Process a single article and generate technical specifications."""
        # Extract technical components using NLP
        components = self._extract_technical_requirements(article_text)
        
        # Generate technical controls
        controls = self._generate_technical_controls(components)
        
        return controls

    def process_document(self):
        """Process the entire document and generate technical specifications."""
        # Pattern to identify articles and their content
        article_pattern = r"Article\s+(\d+)\s*([^\n]+)\n(.*?)(?=Article\s+\d+|\Z)"
        articles = re.finditer(article_pattern, self.text, re.DOTALL)
        
        output = []
        
        for article in articles:
            article_num = article.group(1)
            article_title = article.group(2).strip()
            article_content = article.group(3).strip()
            
            output.append(f"\nArticle {article_num}: {article_title}")
            output.append("\nLegal Requirement:")
            output.append(article_content)
            
            # Process the article content
            technical_specs = self.process_article(article_content)
            
            output.append("\nTechnical Implementation Specifications:")
            for category, controls in technical_specs.items():
                output.append(f"\n{category}:")
                for control in controls:
                    output.append(f"- {control}")
            
            output.append("\n" + "="*80 + "\n")
        
        return "\n".join(output)

def main():
    pdf_path = '/Users/Tawfiq/Desktop/gpt-pilot-backup-0-2-7-73a8c223/workspace/WorkShop/CELEX_32022R2554_EN_TXT.pdf'
    output_file = 'dora_technical_specifications.txt'
    
    print("Processing DORA document...")
    try:
        extractor = DORARequirementExtractor(pdf_path)
        content = extractor.process_document()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Technical specifications have been written to {output_file}")
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    main()
