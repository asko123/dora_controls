import pdfplumber
import pandas as pd
import spacy
import re
from typing import Tuple, List, Dict
from collections import defaultdict
from pathlib import Path
import difflib

class DORAComplianceAnalyzer:
    def __init__(self, dora_pdf_path):
        self.dora_pdf_path = dora_pdf_path
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_lg")
        self.dora_text = self._extract_and_clean_text()
        self.rts_requirements = defaultdict(list)
        self.its_requirements = defaultdict(list)
        self.policy_coverage = defaultdict(list)
        
        # Define key areas for policy matching
        self.policy_areas = {
            'authentication_security': [
                'authentication', 'login', 'credentials', 'access control',
                'authentication methods', 'multi-factor', 'password', 'identity verification',
                'biometric', 'single sign-on', 'SSO', 'MFA', '2FA', 'two-factor',
                'authentication factors', 'token authentication', 'oauth', 'SAML',
                'federated authentication', 'authentication protocols', 'password policy',
                'password complexity', 'authentication strength', 'authentication logs',
                'authentication failure', 'brute force protection', 'session management',
                'session timeout', 'authentication audit', 'strong authentication',
                'identity provider', 'authentication service', 'authentication gateway',
                'authentication server', 'authentication framework', 'authentication mechanism',
                'authentication flow', 'authentication interface', 'authentication API',
                'authentication plugin', 'authentication module', 'authentication library',
                'authentication component', 'authentication configuration', 'authentication settings',
                'authentication rules', 'authentication policy', 'authentication standard',
                'authentication requirement', 'authentication control', 'authentication monitoring',
                'authentication logging', 'authentication reporting', 'authentication metrics',
                'authentication dashboard', 'authentication analytics', 'authentication intelligence',
                'authentication insight'
            ],
            'cryptography': [
                'cryptography', 'encryption', 'cryptographic controls', 'key management',
                'digital signatures', 'certificates', 'PKI', 'cryptographic algorithms',
                'TLS', 'SSL', 'HTTPS', 'symmetric encryption', 'asymmetric encryption',
                'public key', 'private key', 'key rotation', 'key storage', 'HSM',
                'hardware security module', 'certificate authority', 'CA', 'key ceremony',
                'cryptographic protocols', 'cipher suites', 'hash functions', 'SHA',
                'AES', 'RSA', 'elliptic curve', 'ECDSA', 'key derivation', 'PKCS',
                'certificate management', 'certificate validation', 'revocation',
                'cryptographic operations', 'secure key generation', 'random number generation',
                'cryptographic module', 'cryptographic service', 'cryptographic provider',
                'cryptographic library', 'cryptographic algorithm', 'cryptographic protocol',
                'cryptographic standard', 'cryptographic requirement', 'cryptographic policy',
                'cryptographic control', 'cryptographic monitoring', 'cryptographic logging',
                'cryptographic reporting', 'cryptographic metrics', 'cryptographic dashboard',
                'cryptographic analytics', 'cryptographic intelligence', 'cryptographic insight',
                'key lifecycle', 'key backup', 'key recovery', 'key archival', 'key escrow',
                'key distribution', 'key agreement', 'key exchange', 'key transport',
                'key wrapping', 'key unwrapping', 'key verification', 'key validation',
                'key compromise', 'key revocation', 'key destruction', 'key deletion'
            ],
            'data_protection': [
                'data protection', 'data privacy', 'data classification', 'data handling',
                'data security', 'information protection', 'data governance', 'data lifecycle',
                'PII', 'sensitive data', 'confidential information', 'data retention',
                'data disposal', 'data access', 'data transfer', 'data storage',
                'data encryption', 'data masking', 'data anonymization', 'data pseudonymization',
                'data minimization', 'data accuracy', 'data integrity', 'data availability',
                'data backup', 'data recovery', 'data breach', 'data loss prevention',
                'DLP', 'data mapping', 'data inventory', 'data flow', 'cross-border data',
                'data sovereignty', 'data residency', 'data processing', 'GDPR',
                'data subject rights', 'privacy impact assessment', 'DPIA',
                'data classification policy', 'data handling procedure', 'data security control',
                'data protection requirement', 'data privacy standard', 'data governance framework',
                'data lifecycle management', 'data retention policy', 'data disposal procedure',
                'data access control', 'data transfer security', 'data storage security',
                'data encryption requirement', 'data masking policy', 'data anonymization standard',
                'data pseudonymization procedure', 'data minimization principle', 'data accuracy check',
                'data integrity control', 'data availability requirement', 'data backup policy',
                'data recovery procedure', 'data breach response', 'data loss prevention system',
                'data mapping exercise', 'data inventory management', 'data flow diagram',
                'cross-border data transfer', 'data sovereignty requirement', 'data residency policy',
                'data processing agreement', 'privacy impact assessment procedure'
            ],
            'incident_response': [
                'incident response', 'incident handling', 'incident reporting',
                'incident management', 'security incidents', 'breach response',
                'incident investigation', 'incident recovery', 'incident detection',
                'incident containment', 'incident eradication', 'incident documentation',
                'incident classification', 'incident prioritization', 'incident escalation',
                'incident notification', 'incident tracking', 'incident metrics',
                'incident playbooks', 'incident response team', 'CSIRT', 'CERT',
                'incident response plan', 'incident coordination', 'incident communication',
                'post-incident analysis', 'lessons learned', 'root cause analysis',
                'forensic analysis', 'digital forensics', 'incident timeline',
                'incident evidence', 'chain of custody', 'incident remediation',
                'incident response procedure', 'incident handling policy', 'incident reporting requirement',
                'incident management system', 'security incident process', 'breach response plan',
                'incident investigation procedure', 'incident recovery policy', 'incident detection system',
                'incident containment strategy', 'incident eradication procedure', 'incident documentation standard',
                'incident classification scheme', 'incident prioritization matrix', 'incident escalation procedure',
                'incident notification policy', 'incident tracking system', 'incident metrics dashboard',
                'incident playbook template', 'incident response team structure', 'CSIRT policy',
                'CERT procedure', 'incident response plan review', 'incident coordination protocol',
                'incident communication plan', 'post-incident analysis procedure', 'lessons learned process',
                'root cause analysis methodology', 'forensic analysis procedure', 'digital forensics policy',
                'incident timeline documentation', 'incident evidence handling', 'chain of custody procedure',
                'incident remediation plan'
            ],
            'open_source': [
                'open source software', 'third-party components', 'software dependencies',
                'dependency management', 'component security', 'vulnerability scanning',
                'license compliance', 'open source policy', 'component versioning',
                'dependency updates', 'security patches', 'version control',
                'source code review', 'code security analysis', 'dependency check',
                'software composition analysis', 'SCA', 'component inventory',
                'library management', 'package management', 'repository management',
                'artifact management', 'build dependencies', 'runtime dependencies',
                'development dependencies', 'test dependencies', 'production dependencies',
                'dependency resolution', 'dependency conflict', 'dependency tree',
                'dependency graph', 'dependency audit', 'dependency security',
                'dependency vulnerability', 'dependency update', 'dependency patch'
            ],
            'production_access': [
                'production access', 'privileged access', 'access management',
                'production environment', 'administrative access', 'elevated privileges',
                'production environment access', 'privileged access management', 'PAM',
                'access control policy', 'production system access', 'administrative privileges',
                'elevated access', 'emergency access', 'break-glass access', 'just-in-time access',
                'temporary access', 'access review', 'access certification', 'access audit',
                'access monitoring', 'access logging', 'access reporting', 'access metrics',
                'access dashboard', 'access analytics', 'access intelligence', 'access insight',
                'production support access', 'production maintenance access', 'production operations access',
                'production deployment access', 'production release access', 'production change access',
                'production incident access', 'production problem access', 'production emergency access'
            ],
            'asset_inventory': [
                'asset inventory', 'asset management', 'technology assets',
                'asset tracking', 'inventory control', 'asset lifecycle',
                'asset registration', 'asset documentation',
                'technology asset inventory', 'asset management system', 'asset tracking',
                'asset lifecycle management', 'asset registration', 'asset documentation',
                'asset classification', 'asset categorization', 'asset tagging',
                'asset labeling', 'asset identification', 'asset discovery',
                'asset scanning', 'asset monitoring', 'asset reporting',
                'asset metrics', 'asset dashboard', 'asset analytics',
                'asset intelligence', 'asset insight', 'asset audit',
                'asset compliance', 'asset risk', 'asset security',
                'asset vulnerability', 'asset patch', 'asset update',
                'asset maintenance', 'asset support', 'asset warranty',
                'asset contract', 'asset license', 'asset ownership',
                'asset custodian', 'asset steward', 'asset administrator'
            ],
            'network_security': [
                'network security', 'network platforms', 'network services',
                'network infrastructure', 'connectivity', 'network architecture',
                'network controls', 'network protection', 'firewall', 'IDS', 'IPS',
                'network segmentation', 'VLAN', 'DMZ', 'network access control', 'NAC',
                'network monitoring', 'network traffic analysis', 'packet inspection',
                'network protocols', 'routing security', 'switching security',
                'wireless security', 'WPA', 'network authentication', '802.1x',
                'VPN', 'remote access', 'network encryption', 'network isolation',
                'network zones', 'network documentation', 'network diagram',
                'network inventory', 'network baseline', 'network hardening',
                'network vulnerability', 'network penetration testing', 'network audit',
                'network compliance', 'SDN', 'zero trust network',
                'network access policy', 'network security standard', 'network protection requirement',
                'firewall rule', 'IDS signature', 'IPS policy', 'network segmentation design',
                'VLAN configuration', 'DMZ architecture', 'network access control policy',
                'network monitoring requirement', 'traffic analysis procedure', 'packet inspection policy',
                'protocol security', 'routing security policy', 'switching security standard',
                'wireless security requirement', 'WPA configuration', 'network authentication policy',
                '802.1x implementation', 'VPN configuration', 'remote access policy',
                'network encryption standard', 'network isolation requirement', 'network zone design',
                'network documentation standard', 'network diagram requirement', 'network inventory policy',
                'network baseline configuration', 'network hardening standard', 'network vulnerability assessment',
                'network penetration test', 'network audit requirement', 'network compliance standard',
                'SDN implementation', 'zero trust architecture'
            ],
            'application_resiliency': [
                'application availability', 'system resilience', 'service reliability',
                'fault tolerance', 'high availability', 'disaster recovery',
                'business continuity', 'service continuity', 'application recovery',
                'system recovery', 'failover', 'failback', 'load balancing',
                'redundancy', 'replication', 'backup', 'restore', 'recovery point objective',
                'RPO', 'recovery time objective', 'RTO', 'business impact analysis',
                'BIA', 'continuity planning', 'recovery planning', 'resilience testing',
                'availability monitoring', 'performance monitoring', 'capacity planning',
                'scalability', 'elasticity', 'auto-scaling', 'resource management',
                'service level agreement', 'SLA', 'operational level agreement', 'OLA'
            ],
            'backup_restoration': [
                'backup', 'restoration', 'data recovery', 'backup security',
                'recovery procedures', 'backup management', 'restore testing',
                'backup policy', 'restoration procedure', 'data recovery plan',
                'backup security control', 'recovery procedure', 'backup management system',
                'restore testing protocol', 'backup schedule', 'backup retention',
                'backup verification', 'backup validation', 'backup monitoring',
                'backup reporting', 'backup metrics', 'backup performance',
                'backup capacity', 'backup storage', 'backup media',
                'backup encryption', 'backup compression', 'backup deduplication',
                'backup replication', 'backup archival', 'backup catalog',
                'backup inventory', 'backup audit', 'backup compliance',
                'backup security', 'backup access control', 'backup authentication',
                'backup authorization', 'backup logging', 'backup monitoring'
            ],
            'datacenter_security': [
                'datacenter security', 'physical security', 'environmental controls',
                'facility security', 'datacenter operations', 'infrastructure security',
                'physical security', 'environmental controls', 'facility security',
                'datacenter operations', 'infrastructure security', 'access control',
                'surveillance', 'monitoring', 'power management', 'cooling system',
                'fire suppression', 'environmental monitoring', 'physical access',
                'visitor management', 'security zones', 'perimeter security',
                'badge access', 'biometric access', 'security guards',
                'video surveillance', 'CCTV', 'alarm systems', 'emergency response',
                'disaster recovery', 'business continuity', 'facility management',
                'maintenance procedures', 'cleaning procedures', 'vendor access',
                'loading dock', 'shipping area', 'receiving area',
                'equipment installation', 'equipment removal', 'asset management'
            ],
            'entitlement_management': [
                'entitlement management', 'access rights', 'permissions',
                'role management', 'authorization', 'privilege management',
                'access rights', 'permissions management', 'role management',
                'authorization policy', 'privilege management', 'access control',
                'role-based access', 'RBAC', 'attribute-based access', 'ABAC',
                'policy-based access', 'access governance', 'access review',
                'access certification', 'access audit', 'access monitoring',
                'access reporting', 'access metrics', 'access dashboard',
                'access analytics', 'access intelligence', 'access insight',
                'entitlement review', 'entitlement certification', 'entitlement audit',
                'entitlement monitoring', 'entitlement reporting', 'entitlement metrics',
                'entitlement dashboard', 'entitlement analytics', 'entitlement intelligence'
            ],
            'identity_management': [
                'identity management', 'user identity', 'identity lifecycle',
                'identity governance', 'account management', 'identity controls',
                'user identity', 'identity lifecycle', 'identity governance',
                'account management', 'identity controls', 'identity provider',
                'identity store', 'identity repository', 'identity directory',
                'identity federation', 'identity synchronization', 'identity provisioning',
                'identity deprovisioning', 'identity reconciliation', 'identity attestation',
                'identity verification', 'identity validation', 'identity authentication',
                'identity authorization', 'identity audit', 'identity compliance',
                'identity security', 'identity risk', 'identity threat',
                'identity protection', 'identity monitoring', 'identity analytics',
                'identity intelligence', 'identity insight', 'identity metrics',
                'identity dashboard', 'identity reporting', 'identity review'
            ],
            'patch_management': [
                'patch management', 'security patches', 'system updates',
                'patch deployment', 'vulnerability remediation', 'update management',
                'patch testing', 'patch validation', 'patch verification',
                'patch rollback', 'patch schedule', 'patch window',
                'patch cycle', 'patch priority', 'patch classification',
                'patch inventory', 'patch compliance', 'patch reporting',
                'patch metrics', 'patch dashboard', 'patch analytics',
                'patch intelligence', 'patch automation', 'patch distribution',
                'patch installation', 'patch verification', 'patch documentation',
                'patch history', 'patch audit', 'patch review',
                'patch assessment', 'patch risk', 'patch impact',
                'patch dependency'
            ],
            'vulnerability_management': [
                'vulnerability management', 'security vulnerabilities', 'vulnerability assessment',
                'security scanning', 'vulnerability remediation', 'security testing',
                'vulnerability identification', 'vulnerability classification',
                'vulnerability prioritization', 'vulnerability tracking', 'vulnerability reporting',
                'vulnerability metrics', 'vulnerability dashboard', 'vulnerability analytics',
                'vulnerability intelligence', 'vulnerability insight', 'vulnerability scan',
                'vulnerability assessment', 'vulnerability test', 'vulnerability audit',
                'vulnerability review', 'vulnerability monitoring', 'vulnerability management system',
                'VMS', 'security information', 'security event management', 'SIEM',
                'threat intelligence', 'threat detection', 'threat response',
                'incident management', 'incident response', 'incident handling',
                'security operations', 'SOC', 'security monitoring'
            ],
            'threat_intelligence': [
                'threat intelligence', 'cyber threats', 'threat analysis',
                'threat detection', 'threat monitoring', 'security intelligence',
                'threat hunting', 'threat assessment', 'threat modeling',
                'threat landscape', 'threat actor', 'threat vector',
                'threat surface', 'threat indicator', 'indicator of compromise',
                'IOC', 'threat feed', 'threat platform', 'threat sharing',
                'threat response', 'threat mitigation', 'threat prevention',
                'threat protection', 'threat defense', 'threat intelligence platform',
                'TIP', 'cyber intelligence', 'security intelligence',
                'intelligence sharing', 'intelligence analysis', 'intelligence reporting',
                'intelligence dashboard', 'intelligence metrics', 'intelligence analytics'
            ],
            'system_monitoring': [
                'system monitoring', 'performance monitoring', 'security monitoring',
                'monitoring controls', 'alerts', 'monitoring tools',
                'system metrics', 'performance metrics', 'security metrics',
                'monitoring dashboard', 'monitoring analytics', 'monitoring intelligence',
                'monitoring insight', 'system health', 'system performance',
                'system availability', 'system capacity', 'system utilization',
                'system resources', 'system logs', 'log management',
                'log analysis', 'log monitoring', 'log correlation',
                'log aggregation', 'log retention', 'log archival',
                'log search', 'log review', 'log audit', 'log compliance',
                'log security'
            ],
            'media_sanitization': [
                'media sanitization', 'data destruction', 'secure disposal',
                'media disposal', 'data wiping', 'secure erasure',
                'data destruction', 'secure disposal', 'media disposal',
                'data wiping', 'secure erasure', 'media destruction',
                'physical destruction', 'degaussing', 'shredding',
                'sanitization verification', 'sanitization validation', 'sanitization certification',
                'disposal procedure', 'disposal policy', 'disposal standard',
                'disposal requirement', 'disposal control', 'disposal monitoring',
                'disposal logging', 'disposal reporting', 'disposal metrics',
                'disposal dashboard', 'disposal analytics', 'disposal intelligence',
                'disposal insight', 'media handling', 'media storage',
                'media inventory', 'media tracking', 'media classification',
                'media labeling', 'media protection', 'media security'
            ],
            'change_management': [
                'change management', 'production change', 'change control',
                'release management', 'change procedures', 'deployment management',
                'change request', 'change ticket', 'change record',
                'change documentation', 'change approval', 'change authorization',
                'change implementation', 'change validation', 'change verification',
                'change testing', 'change rollback', 'change window',
                'change schedule', 'change calendar', 'change freeze',
                'change blackout', 'emergency change', 'standard change',
                'normal change', 'change advisory board', 'CAB',
                'change management system', 'change tracking', 'change monitoring',
                'change reporting', 'change metrics', 'change dashboard',
                'change analytics'
            ]
        }

    def _extract_and_clean_text(self) -> str:
        """Extract and clean text and tables from PDF document."""
        print("Extracting content from PDF...")
        
        full_text = []
        tables_data = []
        current_table = None
        table_header = None
        
        try:
            with pdfplumber.open(self.dora_pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"Processing page {page_num}/{len(pdf.pages)}")
                    
                    # Extract tables from the page
                    tables = page.extract_tables()
                    
                    if tables:
                        for table in tables:
                            # Check if this is a continuation of a previous table
                            if current_table is not None:
                                if len(table[0]) == len(current_table[0]):  # Same number of columns
                                    current_table.extend(table[1:])  # Add rows without header
                                    continue
                                else:
                                    # Process the completed table
                                    self._process_completed_table(current_table, tables_data)
                                    current_table = table
                                    table_header = table[0]
                            else:
                                current_table = table
                                table_header = table[0]
                    elif current_table is not None:
                        # Process the completed table
                        self._process_completed_table(current_table, tables_data)
                        current_table = None
                        table_header = None
                    
                    # Extract text, excluding table areas
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if text:
                        # Remove table content from text to avoid duplication
                        text = self._remove_table_content_from_text(text, tables)
                        full_text.append(text)
                
                # Process any remaining table
                if current_table is not None:
                    self._process_completed_table(current_table, tables_data)
            
            # Combine text and formatted table data
            combined_text = "\n".join(full_text)
            if tables_data:
                combined_text += "\n\nExtracted Tables:\n"
                combined_text += self._format_tables_data(tables_data)
            
            return self._clean_text(combined_text)
        
        except Exception as e:
            print(f"Error extracting content from PDF: {str(e)}")
            return ""

    def _process_completed_table(self, table: List[List[str]], tables_data: List[Dict]) -> None:
        """Process a completed table and add it to tables_data."""
        try:
            # Remove empty rows and clean cell content
            cleaned_table = [
                [self._clean_cell_content(cell) for cell in row]
                for row in table
                if any(cell.strip() for cell in row)
            ]
            
            if not cleaned_table:
                return
            
            # Create DataFrame
            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
            
            # Store table data with metadata
            tables_data.append({
                'data': df,
                'header': cleaned_table[0],
                'num_rows': len(df),
                'num_cols': len(df.columns)
            })
        
        except Exception as e:
            print(f"Error processing table: {str(e)}")

    def _clean_cell_content(self, cell: str) -> str:
        """Clean individual cell content."""
        if not isinstance(cell, str):
            return str(cell)
        
        # Remove extra whitespace and newlines
        cleaned = ' '.join(cell.split())
        return cleaned.strip()

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
                        table_content.add(cell.strip())
        
        # Split text into lines and remove those matching table content
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Keep line if it's not in table content
            if line and not any(table_text in line for table_text in table_content):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _format_tables_data(self, tables_data: List[Dict]) -> str:
        """Format extracted tables data into readable text."""
        formatted_text = []
        
        for i, table in enumerate(tables_data, 1):
            formatted_text.append(f"\nTable {i}:")
            formatted_text.append("-" * 40)
            
            # Convert DataFrame to string with proper formatting
            table_str = table['data'].to_string(index=False)
            formatted_text.append(table_str)
            formatted_text.append("")  # Empty line after table
        
        return "\n".join(formatted_text)

    def _identify_table_type(self, table_data: pd.DataFrame) -> str:
        """Identify the type of table based on its content and structure."""
        header_text = ' '.join(str(col).lower() for col in table_data.columns)
        
        # Define patterns for different table types
        table_patterns = {
            'requirements': ['requirement', 'control', 'measure', 'standard'],
            'risk_assessment': ['risk', 'impact', 'likelihood', 'score'],
            'technical_standards': ['rts', 'its', 'technical standard', 'specification'],
            'compliance': ['compliance', 'status', 'gap', 'assessment'],
            'controls': ['control', 'description', 'owner', 'status']
        }
        
        for table_type, patterns in table_patterns.items():
            if any(pattern in header_text for pattern in patterns):
                return table_type
        
        return 'other'

    def _identify_policy_area(self, text: str) -> str:
        """Identify the policy area based on content analysis with improved handling."""
        try:
            doc = self.nlp(text.lower())
            area_scores = defaultdict(float)
            
            # Get non-empty sentences
            sentences = [sent for sent in doc.sents if len(sent) > 0]
            
            for area, keywords in self.policy_areas.items():
                area_score = 0.0
                keyword_matches = 0
                
                for keyword in keywords:
                    keyword_doc = self.nlp(keyword)
                    
                    # Skip empty documents
                    if not keyword_doc.vector_norm:
                        continue
                        
                    # Compare with each sentence
                    for sent in sentences:
                        if not sent.vector_norm:
                            continue
                        try:
                            similarity = keyword_doc.similarity(sent)
                            if similarity > 0.6:  # Threshold for keyword match
                                area_score += similarity
                                keyword_matches += 1
                        except Exception as e:
                            continue
                
                # Calculate final score considering both similarity and matches
                if keyword_matches > 0:
                    area_scores[area] = area_score / keyword_matches
            
            # Return the area with highest score or 'general' if no good matches
            if area_scores:
                return max(area_scores.items(), key=lambda x: x[1])[0]
            return 'general'
            
        except Exception as e:
            print(f"Error in policy area identification: {str(e)}")
            return 'general'

    def extract_technical_standards(self):
        """Extract RTS and ITS requirements with enhanced precision."""
        # More comprehensive patterns for RTS
        rts_patterns = [
            # Direct RTS references
            r"RTS\s+(?:shall|should|must|will)\s+([^\.]+\.[^\.]+)",
            
            # Regulatory technical standards patterns
            r"regulatory\s+technical\s+standards?\s*(?:shall|should|must|will)\s+([^\.]+\.[^\.]+)",
            r"regulatory\s+technical\s+standards?\s*(?:to|that|which)\s+specify[^\.]+([^\.]+\.[^\.]+)",
            r"regulatory\s+technical\s+standards?\s*(?:for|on|regarding)\s+([^\.]+\.[^\.]+)",
            
            # EBA/ESMA development patterns
            r"(?:EBA|ESMA)\s+shall\s+develop\s+(?:draft\s+)?regulatory\s+technical\s+standards?\s+(?:to|that|which)\s+([^\.]+\.[^\.]+)",
            r"(?:EBA|ESMA)\s+shall\s+specify[^\.]+through\s+regulatory\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            
            # Mandate patterns
            r"mandate\s+to\s+develop\s+regulatory\s+technical\s+standards?\s+(?:for|on)\s+([^\.]+\.[^\.]+)",
            
            # Specification patterns
            r"specify\s+in\s+regulatory\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            
            # Requirement patterns
            r"requirements?\s+(?:shall|should|must|will)\s+be\s+specified\s+in\s+regulatory\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            
            # Detailed provisions
            r"detailed\s+provisions?\s+(?:shall|should|must|will)\s+be\s+laid\s+down\s+in\s+regulatory\s+technical\s+standards?\s+([^\.]+\.[^\.]+)"
        ]
        
        # More comprehensive patterns for ITS
        its_patterns = [
            # Direct ITS references
            r"ITS\s+(?:shall|should|must|will)\s+([^\.]+\.[^\.]+)",
            
            # Implementing technical standards patterns
            r"implementing\s+technical\s+standards?\s*(?:shall|should|must|will)\s+([^\.]+\.[^\.]+)",
            r"implementing\s+technical\s+standards?\s*(?:to|that|which)\s+specify[^\.]+([^\.]+\.[^\.]+)",
            r"implementing\s+technical\s+standards?\s*(?:for|on|regarding)\s+([^\.]+\.[^\.]+)",
            
            # EBA/ESMA development patterns
            r"(?:EBA|ESMA)\s+shall\s+develop\s+(?:draft\s+)?implementing\s+technical\s+standards?\s+(?:to|that|which)\s+([^\.]+\.[^\.]+)",
            r"(?:EBA|ESMA)\s+shall\s+specify[^\.]+through\s+implementing\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            
            # Implementation patterns
            r"implement(?:ed|ing)?\s+through\s+implementing\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            
            # Format patterns
            r"format\s+(?:shall|should|must|will)\s+be\s+specified\s+in\s+implementing\s+technical\s+standards?\s+([^\.]+\.[^\.]+)",
            
            # Uniform patterns
            r"uniform\s+(?:format|templates|procedures|forms)\s+(?:shall|should|must|will)\s+be\s+specified\s+in\s+implementing\s+technical\s+standards?\s+([^\.]+\.[^\.]+)"
        ]

        def extract_full_requirement(text, match_start, max_chars=1000):
            """Extract the full requirement context around the match."""
            # Find the start of the sentence containing the match
            start = text.rfind('.', max(0, match_start - max_chars), match_start)
            start = start + 1 if start != -1 else max(0, match_start - max_chars)
            
            # Find the end of the requirement (might span multiple sentences)
            end = text.find('.', match_start)
            next_end = text.find('.', end + 1)
            while next_end != -1 and next_end - end < 100:  # Continue if sentences are closely related
                end = next_end
                next_end = text.find('.', end + 1)
            
            return text[start:end + 1].strip()

        # Extract articles with enhanced pattern
        article_pattern = r"Article\s+(\d+)\s*([^\n]+)\n(.*?)(?=Article\s+\d+|\Z)"
        articles = re.finditer(article_pattern, self.dora_text, re.DOTALL)
        
        for article in articles:
            article_num = article.group(1)
            article_title = article.group(2).strip()
            article_content = article.group(3).strip()
            
            # Identify policy area for the article
            article_area = self._identify_policy_area(f"{article_title} {article_content}")
            
            # Process RTS requirements
            for pattern in rts_patterns:
                matches = re.finditer(pattern, article_content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    full_requirement = extract_full_requirement(article_content, match.start())
                    requirement = {
                        'article_num': article_num,
                        'article_title': article_title,
                        'requirement_text': match.group(1).strip(),
                        'full_context': full_requirement,
                        'type': 'RTS',
                        'policy_area': article_area,
                        'pattern_matched': pattern
                    }
                    self.rts_requirements[article_num].append(requirement)
            
            # Process ITS requirements
            for pattern in its_patterns:
                matches = re.finditer(pattern, article_content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    full_requirement = extract_full_requirement(article_content, match.start())
                    requirement = {
                        'article_num': article_num,
                        'article_title': article_title,
                        'requirement_text': match.group(1).strip(),
                        'full_context': full_requirement,
                        'type': 'ITS',
                        'policy_area': article_area,
                        'pattern_matched': pattern
                    }
                    self.its_requirements[article_num].append(requirement)

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
                                    print(f"Skipping invalid table {table_idx + 1} on page {page_num}")
                                    continue
                                    
                                if not table or not any(row for row in table if any(cell for cell in row)):
                                    continue  # Skip empty tables
                                    
                                # Process table continuation
                                if current_table is not None:
                                    is_continuation = self._check_table_continuation(
                                        current_table, table, page_num
                                    )
                                    
                                    if is_continuation:
                                        print(f"Detected table continuation on page {page_num}")
                                        current_table.extend(table[1:])  # Add rows without header
                                        continue
                                    else:
                                        # Process the completed table
                                        self._process_completed_table(
                                            current_table, 
                                            tables_data, 
                                            start_page=table_page_start,
                                            end_page=page_num-1
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
                                end_page=page_num-1
                            )
                            current_table = None
                            table_header = None
                            table_page_start = None
                        
                        # Extract and clean text
                        text = page.extract_text(x_tolerance=2, y_tolerance=2)
                        if text:
                            cleaned_text = self._remove_table_content_from_text_enhanced(text, tables)
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
                        end_page=total_pages
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

    def _check_table_continuation(self, current_table: List[List[str]], 
                                    new_table: List[List[str]], page_num: int) -> bool:
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

    def _analyze_requirements(self, policy_text: str, tables_data: List[Dict]) -> List[Dict]:
        """Analyze requirements against policy content."""
        coverage_results = []
        all_requirements = []
        
        # Collect all requirements
        for article_num, requirements in self.rts_requirements.items():
            for req in requirements:
                all_requirements.append({
                    'article_num': article_num,
                    'requirement_type': 'RTS',
                    'requirement_text': req['requirement_text'],
                    'full_context': req.get('full_context', ''),
                    'policy_area': req['policy_area']
                })
        
        for article_num, requirements in self.its_requirements.items():
            for req in requirements:
                all_requirements.append({
                    'article_num': article_num,
                    'requirement_type': 'ITS',
                    'requirement_text': req['requirement_text'],
                    'full_context': req.get('full_context', ''),
                    'policy_area': req['policy_area']
                })
        
        print(f"Analyzing {len(all_requirements)} requirements...")
        
        for req in all_requirements:
            try:
                requirement_text = req['full_context'] or req['requirement_text']
                
                # Calculate similarity scores
                text_similarity = self._calculate_similarity(requirement_text, policy_text)
                
                table_similarity = 0.0
                if tables_data:
                    table_text = self._format_tables_data(tables_data)
                    table_similarity = self._calculate_similarity(requirement_text, table_text)
                
                similarity_score = max(text_similarity, table_similarity)
                
                coverage_results.append({
                    'article_num': req['article_num'],
                    'requirement_type': req['requirement_type'],
                    'requirement_text': req['requirement_text'],
                    'full_context': req.get('full_context', ''),
                    'covered': similarity_score > 0.7,
                    'similarity_score': similarity_score,
                    'policy_area': req['policy_area']
                })
                
            except Exception as e:
                print(f"Error analyzing requirement from Article {req['article_num']}: {str(e)}")
                coverage_results.append({
                    'article_num': req['article_num'],
                    'requirement_type': req['requirement_type'],
                    'requirement_text': req['requirement_text'],
                    'full_context': req.get('full_context', ''),
                    'covered': False,
                    'similarity_score': 0.0,
                    'policy_area': req['policy_area'],
                    'error': str(e)
                })
        
        return coverage_results

    def _print_analysis_summary(self, policy_name: str, coverage_results: List[Dict]):
        """Print analysis summary."""
        covered = len([r for r in coverage_results if r['covered']])
        gaps = len([r for r in coverage_results if not r['covered']])
        total = len(coverage_results)
        
        print("\nAnalysis Summary for", policy_name)
        print("=" * 40)
        print(f"Total Requirements: {total}")
        print(f"Covered Requirements: {covered}")
        print(f"Gaps Identified: {gaps}")
        if total > 0:
            coverage_percentage = (covered / total) * 100
            print(f"Coverage Percentage: {coverage_percentage:.1f}%")

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts with improved handling."""
        try:
            # Process both texts
            doc1 = self.nlp(text1[:25000])  # Limit text length to avoid memory issues
            doc2 = self.nlp(text2[:25000])
            
            # Get non-empty sentences
            sents1 = [sent for sent in doc1.sents if len(sent) > 0]
            sents2 = [sent for sent in doc2.sents if len(sent) > 0]
            
            if not sents1 or not sents2:
                return 0.0
            
            # Calculate similarity using sentence-level comparison
            max_similarity = 0.0
            for sent1 in sents1:
                if not sent1.vector_norm:
                    continue
                for sent2 in sents2:
                    if not sent2.vector_norm:
                        continue
                    try:
                        similarity = sent1.similarity(sent2)
                        max_similarity = max(max_similarity, similarity)
                    except Exception:
                        continue
            
            return max_similarity
            
        except Exception as e:
            print(f"Error in similarity calculation: {str(e)}")
            return 0.0

    def _calculate_header_similarity(self, header1, header2):
        """Calculate similarity between two table headers."""
        if not header1 or not header2:
            return 0.0
        
        # Clean and normalize headers
        clean_header1 = [str(cell).strip().lower() for cell in header1 if cell]
        clean_header2 = [str(cell).strip().lower() for cell in header2 if cell]
        
        # Calculate similarity using fuzzy matching
        matches = 0
        total = max(len(clean_header1), len(clean_header2))
        
        for h1 in clean_header1:
            for h2 in clean_header2:
                # Use similarity ratio from difflib
                similarity = difflib.SequenceMatcher(None, h1, h2).ratio()
                if similarity > 0.8:  # High similarity threshold
                    matches += 1
                    break
        
        return matches / total if total > 0 else 0.0

    def _remove_table_content_from_text_enhanced(self, text: str, tables: List[List[List[str]]]) -> str:
        """Enhanced removal of table content from extracted text."""
        if not tables:
            return text
        
        # Create a set of table content with variations
        table_content = set()
        for table in tables:
            for row in table:
                for cell in row:
                    if not isinstance(cell, str):
                        cell = str(cell)
                    cell = cell.strip()
                    if cell:
                        # Add original cell content
                        table_content.add(cell)
                        # Add lowercase version
                        table_content.add(cell.lower())
                        # Add without extra spaces
                        table_content.add(' '.join(cell.split()))
        
        # Split text into lines and process each line
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line contains table content
            should_keep = True
            for content in table_content:
                # Use fuzzy matching for more accurate detection
                if content and len(content) > 3:  # Ignore very short content
                    if difflib.SequenceMatcher(None, line.lower(), content.lower()).ratio() > 0.8:
                        should_keep = False
                        break
            
            if should_keep:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _process_completed_table(self, table: List[List[str]], tables_data: List[Dict], 
                                       start_page: int, end_page: int) -> None:
        """Process a completed table with page information."""
        try:
            # Remove empty rows and clean cell content
            cleaned_table = [
                [self._clean_cell_content(cell) for cell in row]
                for row in table
                if any(cell.strip() for cell in row)
            ]
            
            if not cleaned_table:
                return
            
            # Create DataFrame
            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
            
            # Store table data with metadata
            tables_data.append({
                'data': df,
                'header': cleaned_table[0],
                'num_rows': len(df),
                'num_cols': len(df.columns),
                'start_page': start_page,
                'end_page': end_page
            })
            
            print(f"Processed table: {len(df)} rows, {len(df.columns)} columns, "
                  f"pages {start_page}-{end_page}")
        
        except Exception as e:
            print(f"Error processing table: {str(e)}")

    def generate_gap_analysis_report(self):
        """Generate a detailed gap analysis report."""
        output = []
        
        output.append("DORA Compliance Gap Analysis Report")
        output.append("=" * 50 + "\n")
        
        # Track overall statistics
        total_requirements = 0
        total_gaps = 0
        
        # Analyze gaps for each policy
        for policy_name, coverage in self.policy_coverage.items():
            output.append(f"\nPolicy Document: {policy_name}")
            output.append("-" * 30 + "\n")
            
            # Group by policy area
            area_gaps = defaultdict(list)
            for item in coverage:
                if not item['covered']:
                    area_gaps[item['policy_area']].append(item)
            
            # Report gaps by policy area
            if not area_gaps:
                output.append("No gaps identified in this policy.\n")
            else:
                for area, gaps in area_gaps.items():
                    output.append(f"\nPolicy Area: {area.replace('_', ' ').title()}")
                    output.append(f"Number of gaps identified: {len(gaps)}")
                    
                    for gap in gaps:
                        output.append(f"\nArticle {gap['article_num']}:")
                        output.append(f"Requirement Type: {gap['requirement_type']}")
                        output.append("Requirement:")
                        output.append(f"- {gap['requirement_text']}")
                        if gap.get('full_context'):
                            output.append("\nFull Context:")
                            output.append(f"- {gap['full_context']}")
                        output.append(f"Similarity Score: {gap['similarity_score']:.2f}")
                        if 'error' in gap:
                            output.append(f"Analysis Error: {gap['error']}")
                        output.append("")
                    
                    total_gaps += len(gaps)
                    
                output.append("-" * 30 + "\n")
            
            total_requirements += len(coverage)
        
        # Add summary statistics
        output.insert(2, f"\nOverall Statistics:")
        output.insert(3, f"Total Requirements Analyzed: {total_requirements}")
        output.insert(4, f"Total Gaps Identified: {total_gaps}")
        output.insert(5, f"Overall Coverage: {((total_requirements - total_gaps) / total_requirements * 100):.1f}%\n")
        
        return "\n".join(output)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize the extracted text."""
        try:
            # Remove extra whitespace and newlines
            cleaned_text = ' '.join(text.split())
            return cleaned_text
        except Exception as e:
            print(f"Error cleaning text: {str(e)}")
            return text

def main():
    # Initialize analyzer with DORA document
    dora_path = 'CELEX_32022R2554_EN_TXT.pdf'
    analyzer = DORAComplianceAnalyzer(dora_path)
    
    # Extract technical standards
    print("Extracting technical standards from DORA...")
    analyzer.extract_technical_standards()
    
    # Define the folder containing policy documents
    policy_folder = 'policies'
    
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
                policy_name = pdf_path.stem.replace('_', ' ').title()
                print(f"Analyzing: {policy_name}")
                analyzer.analyze_policy_document(str(pdf_path), policy_name)
            except Exception as e:
                print(f"Error analyzing {pdf_path.name}: {str(e)}")
    
        # Generate and save gap analysis report
        print("Generating gap analysis report...")
        report = analyzer.generate_gap_analysis_report()
        
        # Create output folder if it doesn't exist
        output_folder = Path(policy_folder) / 'analysis_output'
        output_folder.mkdir(exist_ok=True)
        
        # Save the report with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_folder / f'dora_gap_analysis_{timestamp}.txt'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Gap analysis report has been written to {output_file}")
        
    except Exception as e:
        print(f"Error processing policy documents: {str(e)}")

if __name__ == "__main__":
    main()
