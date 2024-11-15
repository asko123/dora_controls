import PyPDF2
import re
import spacy
from collections import defaultdict
from pathlib import Path
import logging
from typing import Dict, List, Set

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
        """Extract and clean text from PDF document."""
        with open(self.dora_pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        """Clean the extracted text with enhanced cleaning rules."""
        # Enhanced cleaning patterns
        cleaning_patterns = {
            r'regulat\s*ory\s+technical\s+standards?': 'RTS',
            r'implement\s*ing\s+technical\s+standards?': 'ITS',
            r'tech\s*nical\s+standard': 'technical standard',
            r'\s+': ' ',  # Replace multiple spaces with single space
            r'\n\s*\n': '\n\n',  # Normalize line breaks
        }
        
        cleaned_text = text
        for pattern, replacement in cleaning_patterns.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text.strip()

    def _identify_policy_area(self, text: str) -> str:
        """Identify the policy area based on content analysis."""
        doc = self.nlp(text.lower())
        area_scores = defaultdict(float)
        
        for area, keywords in self.policy_areas.items():
            for keyword in keywords:
                keyword_doc = self.nlp(keyword)
                # Calculate similarity with text
                similarity = max(keyword_doc.similarity(sent) for sent in doc.sents)
                area_scores[area] += similarity
        
        # Return the area with highest score
        return max(area_scores.items(), key=lambda x: x[1])[0] if area_scores else 'general'

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
        """Analyze a policy document with improved relevancy matching."""
        # Extract text from policy document
        with open(policy_pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            policy_text = ""
            for page in reader.pages:
                if page.extract_text():
                    policy_text += page.extract_text() + "\n"
        
        policy_text = self._clean_text(policy_text)
        
        # Identify policy area
        policy_area = self._identify_policy_area(policy_text)
        print(f"Identified policy area for {policy_name}: {policy_area}")
        
        # Process only relevant requirements for this policy area
        for article_num, requirements in self.rts_requirements.items():
            for req in requirements:
                if req['policy_area'] == policy_area:
                    requirement_doc = self.nlp(req['requirement_text'])
                    policy_doc = self.nlp(policy_text)
                    
                    # Calculate similarity score
                    similarity_score = requirement_doc.similarity(policy_doc)
                    
                    if similarity_score > 0.5:  # Adjusted threshold
                        self.policy_coverage[policy_name].append({
                            'article_num': article_num,
                            'requirement_type': 'RTS',
                            'requirement_text': req['requirement_text'],
                            'covered': similarity_score > 0.7,
                            'similarity_score': similarity_score,
                            'policy_area': policy_area
                        })
        
        # Repeat for ITS requirements
        for article_num, requirements in self.its_requirements.items():
            for req in requirements:
                if req['policy_area'] == policy_area:
                    requirement_doc = self.nlp(req['requirement_text'])
                    policy_doc = self.nlp(policy_text)
                    similarity_score = requirement_doc.similarity(policy_doc)
                    
                    if similarity_score > 0.5:  # Adjusted threshold
                        self.policy_coverage[policy_name].append({
                            'article_num': article_num,
                            'requirement_type': 'ITS',
                            'requirement_text': req['requirement_text'],
                            'covered': similarity_score > 0.7,
                            'similarity_score': similarity_score,
                            'policy_area': policy_area
                        })

    def generate_gap_analysis_report(self) -> str:
        """Generate a detailed gap analysis report with improved organization."""
        output = []
        
        output.append("DORA Compliance Gap Analysis Report")
        output.append("=" * 50 + "\n")
        
        # Organize by policy area
        area_gaps = defaultdict(list)
        
        for policy_name, coverage in self.policy_coverage.items():
            for item in coverage:
                if not item['covered']:
                    area_gaps[item['policy_area']].append({
                        'policy_name': policy_name,
                        **item
                    })
        
        # Report gaps by policy area
        for area, gaps in area_gaps.items():
            output.append(f"\nPolicy Area: {area.replace('_', ' ').title()}")
            output.append("-" * 30 + "\n")
            
            if not gaps:
                output.append("No gaps identified in this area.\n")
                continue
            
            for gap in gaps:
                output.append(f"\nArticle {gap['article_num']}:")
                output.append(f"Policy Document: {gap['policy_name']}")
                output.append(f"{gap['requirement_type']} Requirement:")
                output.append(f"- {gap['requirement_text']}")
                output.append(f"- Similarity Score: {gap['similarity_score']:.2f}")
                output.append("")
            
            output.append("=" * 50 + "\n")
        
        return "\n".join(output)

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
