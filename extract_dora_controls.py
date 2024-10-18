import PyPDF2
import re
import random

def extract_dora_controls(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Define control categories aligned with NIST Cybersecurity Framework and DORA
    control_categories = {
        'Identify': [
            "ICT Risk Management Framework (ID.RM): Develop a comprehensive ICT risk management framework that includes processes for identifying, assessing, and mitigating risks related to all ICT systems and services. Ensure the framework covers both internal systems and third-party providers, aligning with DORA Article 5.",
            "Asset Management and Inventory (ID.AM): Implement an automated asset discovery and management system to maintain an up-to-date inventory of all ICT assets, including hardware, software, and data. Classify assets based on criticality and sensitivity, as required by DORA Article 9.",
            "ICT Third-Party Risk Assessment (ID.SC): Establish a rigorous process for assessing and monitoring the ICT risk posed by third-party providers. Include contractual provisions for security measures, audit rights, and incident reporting obligations, in line with DORA Article 28.",
            "Governance Structure (ID.GV): Establish a clear governance structure for ICT risk management, including board-level oversight and a designated senior executive responsible for digital operational resilience, as outlined in DORA Article 4.",
            "Regulatory Compliance Mapping (ID.GV): Develop a comprehensive mapping of all applicable regulatory requirements, including DORA, to ensure full compliance across all ICT systems and processes."
        ],
        'Protect': [
            "Access Control and Identity Management (PR.AC): Implement a robust Identity and Access Management (IAM) system with multi-factor authentication, least privilege principles, and regular access reviews. Ensure alignment with DORA Article 8 requirements for ICT security.",
            "Data Protection and Encryption (PR.DS): Deploy comprehensive data protection measures, including encryption for data at rest and in transit, data loss prevention tools, and secure data disposal methods. Address DORA Article 10 requirements for data and system backup.",
            "Secure Configuration Management (PR.IP): Establish processes for secure configuration management of all ICT systems, including hardening guidelines, patch management, and change control procedures. Align with DORA Article 8 on ICT security policies.",
            "Security Awareness and Training (PR.AT): Develop and implement a comprehensive security awareness and training program for all staff, with specialized training for ICT and security personnel. Address DORA Article 13 requirements for digital operational resilience awareness.",
            "Vulnerability Management (PR.IP): Implement a robust vulnerability management program, including regular vulnerability assessments, penetration testing, and timely remediation of identified vulnerabilities, as per DORA Article 23."
        ],
        'Detect': [
            "Security Information and Event Management (DE.AE): Deploy an advanced SIEM solution to collect, correlate, and analyze security events across all ICT systems. Configure automated alerting based on predefined rules and anomaly detection algorithms, addressing DORA Article 11 on ICT-related incident detection.",
            "Continuous Monitoring (DE.CM): Establish a 24/7 security monitoring capability, including network and endpoint monitoring, user behavior analytics, and threat hunting activities. Align with DORA Article 10 requirements for continuous monitoring mechanisms.",
            "Threat Intelligence Program (DE.DP): Implement a comprehensive threat intelligence program, including participation in information sharing forums and integration of threat feeds into detection and response processes. Address DORA Article 19 on information sharing arrangements.",
            "Anomaly Detection (DE.AE): Develop and implement machine learning-based anomaly detection capabilities to identify unusual patterns in system behavior, network traffic, and user activities that may indicate security incidents.",
            "Third-Party Monitoring (DE.CM): Establish processes for continuous monitoring of third-party service providers' security posture, including real-time monitoring of their externally facing assets and regular security assessments, as per DORA Article 28."
        ],
        'Respond': [
            "Incident Response Planning (RS.RP): Develop and maintain a comprehensive incident response plan that outlines procedures for various types of ICT-related incidents. Include clear roles, responsibilities, and escalation procedures, aligning with DORA Article 11.",
            "Automated Incident Response (RS.MI): Implement automated incident response capabilities for common types of security incidents, including predefined playbooks and integration with security tools for rapid containment and mitigation.",
            "Crisis Communication Plan (RS.CO): Establish a crisis communication plan that includes procedures for notifying internal stakeholders, customers, regulators, and the public about significant ICT incidents. Ensure compliance with DORA Article 15 on incident reporting to competent authorities.",
            "Forensic Capabilities (RS.AN): Develop in-house digital forensics capabilities or establish retainer agreements with external forensics providers to ensure rapid and thorough investigation of security incidents.",
            "Incident Classification and Triage (RS.AN): Implement a standardized incident classification and triage process to ensure appropriate prioritization and allocation of resources during incident response, in line with DORA Article 16 requirements."
        ],
        'Recover': [
            "Business Continuity and Disaster Recovery (RC.RP): Develop, implement, and regularly test comprehensive business continuity and disaster recovery plans for all critical ICT systems and services. Ensure alignment with DORA Article 11 on digital operational resilience testing.",
            "Post-Incident Review Process (RC.IM): Establish a formal post-incident review process to analyze the root causes of incidents, identify areas for improvement, and update security controls and processes accordingly. Address DORA Article 11 requirements for continuous learning and improvement.",
            "ICT Service Continuity Testing (RC.RP): Conduct regular ICT service continuity tests, including full-scale disaster recovery exercises and failover testing for critical systems. Ensure compliance with DORA Article 23 on advanced testing of ICT tools, systems, and processes.",
            "Data and System Recovery (RC.RP): Implement and regularly test data and system recovery procedures, including point-in-time recovery capabilities and validation of data integrity post-recovery. Align with DORA Article 10 requirements for backup policies and restoration.",
            "Third-Party Dependency Recovery (RC.CO): Develop specific recovery plans and procedures for critical third-party service disruptions, including alternative providers or in-house fallback options. Address DORA Article 28 considerations for ICT third-party risk management."
        ]
    }

    # Regular expressions to find relevant sections
    patterns = {
        'Identify': re.compile(r'(?i)(risk management|asset management|third.?party risk|governance).*?(?=\n\n|\Z)', re.DOTALL),
        'Protect': re.compile(r'(?i)(access control|data protection|security policies|training|vulnerability management).*?(?=\n\n|\Z)', re.DOTALL),
        'Detect': re.compile(r'(?i)(incident detection|monitoring|threat intelligence|anomaly detection).*?(?=\n\n|\Z)', re.DOTALL),
        'Respond': re.compile(r'(?i)(incident response|crisis communication|forensics|incident classification).*?(?=\n\n|\Z)', re.DOTALL),
        'Recover': re.compile(r'(?i)(business continuity|disaster recovery|post.?incident|service continuity|data recovery).*?(?=\n\n|\Z)', re.DOTALL)
    }

    suggestions = []

    # Extracting different sections and providing specific control suggestions
    for category, pattern in patterns.items():
        matches = pattern.findall(text)
        if matches:
            suggestions.append(f"{category} Function Controls:")
            # Select 3 random control suggestions for each category
            selected_controls = random.sample(control_categories[category], 3)
            for control in selected_controls:
                suggestions.append(f"- {control}\n")
            suggestions.append("")  # Add a blank line for readability

    return "\n".join(suggestions)

# Example usage
print("Extracting DORA Controls aligned with NIST Cybersecurity Framework...")
pdf_path = 'CELEX_32022R2554_EN_TXT.pdf'
suggested_controls = extract_dora_controls(pdf_path)

output_path = 'suggested_controls_dora_nist.txt'
with open(output_path, 'w') as output_file:
    output_file.write(suggested_controls)

print(f"Suggested controls have been written to {output_path}")
