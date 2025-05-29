"""
DORA Workbook Integration Module

This module integrates the DORA domains workbook with the main DORA analyzer
to provide domain-specific compliance analysis and reporting.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import logging
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Import required modules from the main DORA analyzer
from .dora import DORAComplianceAnalyzer, TextProcessor
from .dora_domains import DORAWorkbook, create_dora_workbook

class DORAWorkbookAnalyzer:
    """
    Class to analyze policy documents against DORA domains and generate
    domain-specific compliance reports.
    """
    
    def __init__(self, dora_path: str):
        """
        Initialize the analyzer with the DORA legislation file path.
        
        Args:
            dora_path: Path to the DORA legislation PDF
        """
        self.logger = logging.getLogger('DORAAnalyzer')
        self.dora_path = dora_path
        
        # Initialize DORA analyzer
        self.dora_analyzer = DORAComplianceAnalyzer(dora_path)
        
        # Initialize DORA workbook
        self.workbook = create_dora_workbook()
        
        # Load sentence transformer for similarity calculations
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Initialize results storage
        self.results = {}
    
    def analyze_policy_against_domains(self, policy_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a single policy against all DORA domains.
        
        Args:
            policy_path: Path to the policy PDF file
            
        Returns:
            Dictionary mapping domain codes to compliance results
        """
        self.logger.info(f"Analyzing policy against DORA domains: {os.path.basename(policy_path)}")
        
        try:
            # Extract text from policy document
            with self.dora_analyzer._open_pdf(policy_path) as pdf:
                policy_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    policy_text += page_text + "\n\n"
            
            if not policy_text.strip():
                self.logger.error(f"Failed to extract text from {policy_path}")
                return {}
                
            # Clean and normalize the extracted text
            policy_text = TextProcessor.clean_text(policy_text)
            policy_name = os.path.basename(policy_path)
            
            # Analyze against each domain
            domain_results = {}
            
            for domain in tqdm(self.workbook.domains, desc=f"Analyzing {policy_name}", disable=False):
                domain_code = domain["code"]
                requirement_text = domain["requirement"]
                article = domain["article"]
                domain_category = domain["domain"]
                
                # Calculate similarity between domain requirement and policy text
                similarity = self._calculate_similarity(requirement_text, policy_text)
                
                # Determine coverage status
                is_covered = similarity >= 0.65  # Threshold for considering a domain covered
                
                # Store result
                domain_results[domain_code] = {
                    "domain_code": domain_code,
                    "requirement": requirement_text,
                    "article": article,
                    "domain": domain_category,
                    "similarity_score": similarity,
                    "covered": is_covered,
                    "policy_name": policy_name
                }
            
            # Store in results dictionary
            self.results[policy_name] = domain_results
            return domain_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing policy {os.path.basename(policy_path)}: {str(e)}", exc_info=True)
            return {}
    
    def analyze_all_policies(self, policies_folder: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Analyze all policy documents in a folder against all DORA domains.
        
        Args:
            policies_folder: Path to the folder containing policy PDFs
            
        Returns:
            Nested dictionary mapping policy names to domain codes to results
        """
        self.logger.info(f"Analyzing all policies in folder: {policies_folder}")
        
        # Get all PDF files from the folder
        pdf_files = list(Path(policies_folder).glob("**/*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {policies_folder}")
            return {}
        
        # Process each policy
        for pdf_file in pdf_files:
            self.analyze_policy_against_domains(str(pdf_file))
        
        return self.results
    
    def generate_domain_compliance_report(self, output_file: str = "dora_domain_compliance.xlsx") -> str:
        """
        Generate a comprehensive domain-based compliance report as an Excel file.
        
        Args:
            output_file: Path for the output Excel file
            
        Returns:
            Path to the generated report file
        """
        if not self.results:
            self.logger.warning("No analysis results available. Run analyze_all_policies first.")
            return ""
        
        self.logger.info("Generating domain compliance report...")
        
        # Create a writer for Excel output
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        
        # Create summary sheet data
        summary_data = []
        
        # Group domains by category
        domain_categories = self.workbook.get_domains_by_category()
        
        # Process each policy
        for policy_name, domain_results in self.results.items():
            for domain_code, result in domain_results.items():
                summary_data.append({
                    "Policy": policy_name,
                    "Domain Code": domain_code,
                    "Domain Category": result["domain"],
                    "Article": result["article"],
                    "Requirement": result["requirement"],
                    "Similarity Score": result["similarity_score"],
                    "Covered": "Yes" if result["covered"] else "No"
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="All Results", index=False)
        
        # Format the summary sheet
        workbook = writer.book
        summary_sheet = writer.sheets["All Results"]
        
        # Add a format for headers
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Add a format for covered/not covered cells
        covered_format = workbook.add_format({'bg_color': '#C6EFCE'})
        not_covered_format = workbook.add_format({'bg_color': '#FFC7CE'})
        
        # Write headers with format
        for col_num, value in enumerate(summary_df.columns.values):
            summary_sheet.write(0, col_num, value, header_format)
        
        # Set column widths
        summary_sheet.set_column('A:A', 30)  # Policy
        summary_sheet.set_column('B:B', 10)  # Domain Code
        summary_sheet.set_column('C:C', 20)  # Domain Category
        summary_sheet.set_column('D:D', 10)  # Article
        summary_sheet.set_column('E:E', 50)  # Requirement
        summary_sheet.set_column('F:F', 15)  # Similarity Score
        summary_sheet.set_column('G:G', 10)  # Covered
        
        # Add conditional formatting for covered/not covered
        summary_sheet.conditional_format(1, 6, len(summary_data), 6, {
            'type': 'text',
            'criteria': 'containing',
            'value': 'Yes',
            'format': covered_format
        })
        
        summary_sheet.conditional_format(1, 6, len(summary_data), 6, {
            'type': 'text',
            'criteria': 'containing',
            'value': 'No',
            'format': not_covered_format
        })
        
        # Create a pivot table of domain coverage by policy
        pivot_data = pd.pivot_table(
            summary_df, 
            values='Similarity Score',
            index=['Domain Category', 'Domain Code', 'Requirement'],
            columns=['Policy'],
            aggfunc=np.mean,
            fill_value=0
        )
        
        # Add coverage indicator to pivot
        for policy in pivot_data.columns:
            pivot_data[f"{policy} (Covered)"] = pivot_data[policy].apply(lambda x: "Yes" if x >= 0.65 else "No")
        
        # Write pivot table to a new sheet
        pivot_data.to_excel(writer, sheet_name="Domain Coverage")
        
        # Format the pivot sheet
        pivot_sheet = writer.sheets["Domain Coverage"]
        pivot_sheet.set_column('A:A', 20)  # Domain Category
        pivot_sheet.set_column('B:B', 10)  # Domain Code
        pivot_sheet.set_column('C:C', 50)  # Requirement
        
        # Create a coverage summary sheet
        coverage_summary = []
        
        # Calculate coverage by domain category
        domain_categories = summary_df.groupby('Domain Category')
        
        for category, group in domain_categories:
            total_requirements = len(group)
            covered_requirements = len(group[group['Covered'] == 'Yes'])
            coverage_pct = (covered_requirements / total_requirements * 100) if total_requirements > 0 else 0
            
            coverage_summary.append({
                'Domain Category': category,
                'Total Requirements': total_requirements,
                'Covered Requirements': covered_requirements,
                'Coverage %': coverage_pct
            })
        
        # Create coverage summary DataFrame
        coverage_df = pd.DataFrame(coverage_summary)
        coverage_df.to_excel(writer, sheet_name="Coverage Summary", index=False)
        
        # Format the coverage summary sheet
        coverage_sheet = writer.sheets["Coverage Summary"]
        
        # Add a format for headers
        for col_num, value in enumerate(coverage_df.columns.values):
            coverage_sheet.write(0, col_num, value, header_format)
        
        # Set column widths
        coverage_sheet.set_column('A:A', 20)  # Domain Category
        coverage_sheet.set_column('B:B', 18)  # Total Requirements
        coverage_sheet.set_column('C:C', 20)  # Covered Requirements
        coverage_sheet.set_column('D:D', 12)  # Coverage %
        
        # Create a chart for the coverage summary
        chart = workbook.add_chart({'type': 'column'})
        
        # Add data series to the chart
        chart.add_series({
            'name': 'Coverage %',
            'categories': ['Coverage Summary', 1, 0, len(coverage_summary), 0],
            'values': ['Coverage Summary', 1, 3, len(coverage_summary), 3],
            'data_labels': {'value': True, 'num_format': '0.0%'},
        })
        
        # Configure the chart
        chart.set_title({'name': 'DORA Compliance Coverage by Domain'})
        chart.set_x_axis({'name': 'Domain Category'})
        chart.set_y_axis({'name': 'Coverage %', 'min': 0, 'max': 100})
        
        # Insert the chart into the worksheet
        coverage_sheet.insert_chart('F2', chart, {'x_scale': 1.5, 'y_scale': 1.5})
        
        # Close the writer and save the Excel file
        writer.close()
        
        self.logger.info(f"Domain compliance report generated: {output_file}")
        return output_file
    
    def _calculate_similarity(self, req_text: str, policy_text: str) -> float:
        """
        Calculate similarity between a requirement and policy text.
        
        Args:
            req_text: The requirement text
            policy_text: The policy text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Create embeddings
            req_embedding = self.sentence_transformer.encode(req_text, convert_to_tensor=True)
            
            # For longer policy texts, we'll use a sliding window approach
            if len(policy_text) > 5000:
                # Split policy text into chunks of ~1000 characters with overlap
                chunks = []
                chunk_size = 1000
                overlap = 200
                
                for i in range(0, len(policy_text), chunk_size - overlap):
                    chunk = policy_text[i:i + chunk_size]
                    if len(chunk) > 100:  # Only consider substantial chunks
                        chunks.append(chunk)
                
                # Calculate similarity for each chunk
                max_similarity = 0.0
                
                for chunk in chunks:
                    chunk_embedding = self.sentence_transformer.encode(chunk, convert_to_tensor=True)
                    similarity = torch.nn.functional.cosine_similarity(
                        req_embedding.unsqueeze(0),
                        chunk_embedding.unsqueeze(0)
                    ).item()
                    
                    max_similarity = max(max_similarity, similarity)
                
                return max_similarity
            else:
                # For shorter texts, calculate similarity directly
                policy_embedding = self.sentence_transformer.encode(policy_text, convert_to_tensor=True)
                similarity = torch.nn.functional.cosine_similarity(
                    req_embedding.unsqueeze(0),
                    policy_embedding.unsqueeze(0)
                ).item()
                
                return similarity
                
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}", exc_info=True)
            return 0.0


def run_workbook_analysis(dora_path: str, policies_folder: str, output_file: str = None) -> str:
    """
    Run the DORA workbook analysis on all policies in a folder.
    
    Args:
        dora_path: Path to the DORA legislation PDF
        policies_folder: Path to the folder containing policy PDFs
        output_file: Optional path for the output report file
        
    Returns:
        Path to the generated report file
    """
    # Set up logging
    logger = logging.getLogger('DORAAnalyzer')
    
    try:
        # Create output filename if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"dora_domain_compliance_{timestamp}.xlsx"
        
        # Initialize analyzer
        analyzer = DORAWorkbookAnalyzer(dora_path)
        
        # Analyze all policies
        analyzer.analyze_all_policies(policies_folder)
        
        # Generate report
        report_path = analyzer.generate_domain_compliance_report(output_file)
        
        return report_path
        
    except Exception as e:
        logger.error(f"Error running workbook analysis: {str(e)}", exc_info=True)
        return ""


if __name__ == "__main__":
    # Example usage
    dora_path = "CELEX_32022R2554_EN_TXT.pdf"
    policies_folder = "policies"
    
    report_path = run_workbook_analysis(dora_path, policies_folder)
    
    if report_path:
        print(f"Analysis complete. Report generated at: {report_path}")
    else:
        print("Analysis failed. Check logs for details.") 
