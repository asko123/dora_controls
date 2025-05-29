"""
DORA Compliance Domains Workbook

This module defines a structured classification of DORA (Digital Operational Resilience Act) 
requirements organized into domains and subdomains, similar to SOC2 structure.
"""

from typing import Dict, List, Any
import json
import pandas as pd
import os

# DORA Domain Structure
DORA_DOMAINS = [
    # ICT Risk Management Framework (Chapter II)
    {"code": "RM 1.1", "article": "Art. 5", "domain": "ICT Risk Management", "requirement": "Governance and strategy for ICT risk"},
    {"code": "RM 1.2", "article": "Art. 6", "domain": "ICT Risk Management", "requirement": "ICT risk management framework implementation"},
    {"code": "RM 1.3", "article": "Art. 6", "domain": "ICT Risk Management", "requirement": "Protection and prevention measures"},
    {"code": "RM 1.4", "article": "Art. 6", "domain": "ICT Risk Management", "requirement": "Detection mechanisms and processes"},
    {"code": "RM 1.5", "article": "Art. 6", "domain": "ICT Risk Management", "requirement": "Response and recovery procedures"},
    {"code": "RM 1.6", "article": "Art. 6", "domain": "ICT Risk Management", "requirement": "Communication strategies and protocols"},
    {"code": "RM 2.1", "article": "Art. 7", "domain": "ICT Risk Management", "requirement": "Identification of critical or important functions"},
    {"code": "RM 2.2", "article": "Art. 7", "domain": "ICT Risk Management", "requirement": "Mapping of ICT dependencies"},
    {"code": "RM 2.3", "article": "Art. 7", "domain": "ICT Risk Management", "requirement": "Risk assessment processes"},
    
    # ICT Incident Management (Chapter III)
    {"code": "IM 1.1", "article": "Art. 17", "domain": "ICT Incident Management", "requirement": "ICT incident management process implementation"},
    {"code": "IM 1.2", "article": "Art. 17", "domain": "ICT Incident Management", "requirement": "Early warning indicators for incidents"},
    {"code": "IM 1.3", "article": "Art. 17", "domain": "ICT Incident Management", "requirement": "Incident detection procedures"},
    {"code": "IM 2.1", "article": "Art. 18", "domain": "ICT Incident Management", "requirement": "Classification of major ICT incidents"},
    {"code": "IM 2.2", "article": "Art. 19", "domain": "ICT Incident Management", "requirement": "Major incident reporting to authorities"},
    {"code": "IM 2.3", "article": "Art. 20", "domain": "ICT Incident Management", "requirement": "Harmonized reporting templates and timelines"},
    {"code": "IM 3.1", "article": "Art. 21", "domain": "ICT Incident Management", "requirement": "Incident response coordination"},
    {"code": "IM 3.2", "article": "Art. 22", "domain": "ICT Incident Management", "requirement": "Incident notification to clients and counterparties"},
    
    # Digital Operational Resilience Testing (Chapter IV)
    {"code": "DT 1.1", "article": "Art. 23", "domain": "Digital Testing", "requirement": "Basic testing program requirements"},
    {"code": "DT 1.2", "article": "Art. 23", "domain": "Digital Testing", "requirement": "Testing of ICT tools and systems"},
    {"code": "DT 1.3", "article": "Art. 24", "domain": "Digital Testing", "requirement": "Advanced testing using TLPT for significant entities"},
    {"code": "DT 2.1", "article": "Art. 25", "domain": "Digital Testing", "requirement": "Requirements for testers for TLPT"},
    {"code": "DT 2.2", "article": "Art. 26", "domain": "Digital Testing", "requirement": "Recognition of TLPT test results within EU"},
    
    # ICT Third-Party Risk Management (Chapter V)
    {"code": "TP 1.1", "article": "Art. 28", "domain": "Third-Party Risk", "requirement": "General principles for third-party risk management"},
    {"code": "TP 1.2", "article": "Art. 28", "domain": "Third-Party Risk", "requirement": "Risk assessment of ICT third-party service providers"},
    {"code": "TP 1.3", "article": "Art. 28", "domain": "Third-Party Risk", "requirement": "Contractual agreements with ICT providers"},
    {"code": "TP 2.1", "article": "Art. 29", "domain": "Third-Party Risk", "requirement": "Key contractual provisions with ICT providers"},
    {"code": "TP 2.2", "article": "Art. 30", "domain": "Third-Party Risk", "requirement": "Sub-outsourcing of critical functions"},
    {"code": "TP 3.1", "article": "Art. 31", "domain": "Third-Party Risk", "requirement": "Designation of critical ICT third-party service providers"},
    {"code": "TP 3.2", "article": "Art. 32", "domain": "Third-Party Risk", "requirement": "Structure of the oversight framework for critical providers"},
    
    # Information Sharing (Chapter VI)
    {"code": "IS 1.1", "article": "Art. 40", "domain": "Information Sharing", "requirement": "Information sharing arrangements on cyber threats"},
    {"code": "IS 1.2", "article": "Art. 40", "domain": "Information Sharing", "requirement": "Participation in information sharing arrangements"},
    {"code": "IS 1.3", "article": "Art. 40", "domain": "Information Sharing", "requirement": "Processing of personal data in sharing arrangements"},
    
    # Competent Authorities (Chapter VII)
    {"code": "CA 1.1", "article": "Art. 46", "domain": "Competent Authorities", "requirement": "Cooperation with competent authorities"},
    {"code": "CA 1.2", "article": "Art. 46", "domain": "Competent Authorities", "requirement": "Notification requirements to authorities"},
    
    # ICT Business Continuity (Multiple Chapters)
    {"code": "BC 1.1", "article": "Art. 11", "domain": "Business Continuity", "requirement": "ICT business continuity policy"},
    {"code": "BC 1.2", "article": "Art. 11", "domain": "Business Continuity", "requirement": "Implementation of business continuity plans"},
    {"code": "BC 1.3", "article": "Art. 11", "domain": "Business Continuity", "requirement": "Backup policies and procedures"},
    {"code": "BC 2.1", "article": "Art. 12", "domain": "Business Continuity", "requirement": "Business impact analysis of ICT disruptions"},
    {"code": "BC 2.2", "article": "Art. 12", "domain": "Business Continuity", "requirement": "Scenario-based contingency plans"},
    {"code": "BC 2.3", "article": "Art. 13", "domain": "Business Continuity", "requirement": "Testing of ICT continuity plans"},
    {"code": "BC 2.4", "article": "Art. 14", "domain": "Business Continuity", "requirement": "Crisis communication plans"},
    
    # ICT Security Awareness and Training
    {"code": "AT 1.1", "article": "Art. 13", "domain": "Awareness & Training", "requirement": "Digital operational resilience awareness programs"},
    {"code": "AT 1.2", "article": "Art. 13", "domain": "Awareness & Training", "requirement": "Regular ICT security training"},
    {"code": "AT 1.3", "article": "Art. 13", "domain": "Awareness & Training", "requirement": "Security training for management"},
    
    # Governance and Oversight
    {"code": "GO 1.1", "article": "Art. 5", "domain": "Governance & Oversight", "requirement": "Management body responsibility for ICT risk"},
    {"code": "GO 1.2", "article": "Art. 5", "domain": "Governance & Oversight", "requirement": "Governance arrangements for ICT risk management"},
    {"code": "GO 1.3", "article": "Art. 5", "domain": "Governance & Oversight", "requirement": "Risk tolerance statements for ICT risk"}
]


class DORAWorkbook:
    """
    Class to handle DORA domains workbook operations
    """
    
    def __init__(self):
        """Initialize the workbook with the DORA domains"""
        self.domains = DORA_DOMAINS
        
    def get_domains_by_category(self) -> Dict[str, List[Dict]]:
        """Group domains by their main category"""
        grouped = {}
        for domain in self.domains:
            category = domain["domain"]
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(domain)
        return grouped
    
    def get_domain_by_code(self, code: str) -> Dict:
        """Retrieve a domain by its code"""
        for domain in self.domains:
            if domain["code"] == code:
                return domain
        return {}
    
    def get_domains_by_article(self, article: str) -> List[Dict]:
        """Retrieve all domains related to a specific article"""
        return [domain for domain in self.domains if domain["article"] == article]
    
    def export_to_csv(self, filename: str = "dora_compliance_workbook.csv") -> str:
        """Export the domains to a CSV file"""
        df = pd.DataFrame(self.domains)
        df.to_csv(filename, index=False)
        return filename
    
    def export_to_excel(self, filename: str = "dora_compliance_workbook.xlsx") -> str:
        """Export the domains to an Excel file with formatting"""
        df = pd.DataFrame(self.domains)
        
        # Create a writer object
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='DORA Domains', index=False)
        
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['DORA Domains']
        
        # Add a format for the header
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Write the column headers with the defined format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Set column widths
        worksheet.set_column('A:A', 8)   # Code
        worksheet.set_column('B:B', 10)  # Article
        worksheet.set_column('C:C', 20)  # Domain
        worksheet.set_column('D:D', 50)  # Requirement
        
        # Close the writer and save the Excel file
        writer.close()
        
        return filename
    
    def map_requirements_to_policy(self, policy_text: str) -> Dict[str, float]:
        """
        Map DORA requirements to a policy text and return similarity scores
        
        Args:
            policy_text: The text content of a policy document
            
        Returns:
            Dictionary mapping domain codes to similarity scores
        """
        # This is a placeholder for the actual implementation
        # The real implementation would use NLP similarity calculations
        similarity_scores = {}
        for domain in self.domains:
            # Placeholder for similarity calculation
            similarity_scores[domain["code"]] = 0.0
        
        return similarity_scores


def create_dora_workbook():
    """Create and return a DORA workbook instance"""
    return DORAWorkbook()


if __name__ == "__main__":
    # Create the workbook
    workbook = create_dora_workbook()
    
    # Export to Excel file
    excel_file = workbook.export_to_excel()
    print(f"DORA Compliance Workbook exported to {excel_file}")
    
    # Export to CSV file
    csv_file = workbook.export_to_csv()
    print(f"DORA Compliance Workbook exported to {csv_file}") 
