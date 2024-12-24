from typing import List, Dict, Optional
import re
from dataclasses import dataclass
from datetime import datetime
import requests
import json

@dataclass
class TranscriptMetadata:
    # Required fields first
    language: str
    date: datetime
    court_name: str
    case_number: str
    case_title: str
    court_type: str
    presiding_judge: str
    location: str
    start_time: datetime
    end_time: datetime
    # Optional fields with default values last
    jurisdiction: str = "Eswatini Judiciary"
    parties: Dict[str, str] = None  # e.g., {"plaintiff": "John Doe", "defendant": "Jane Smith"}

@dataclass
class TranscriptSegment:
    speaker: str
    text: str
    start_time: float
    end_time: float
    confidence: float
    original_text: str
    legal_terms: List[str] = None  # Track important legal terminology
    line_number: int = None
    is_overlapping: bool = False
    is_inaudible: bool = False
    needs_redaction: bool = False
    speaker_role: str = None
    flags: List[str] = None  # For QA markers

class AgentBase:
    def __init__(self, name="BaseAgent", max_retries=2, verbose=True):
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose
        self.ollama_url = "http://localhost:11434/api/generate"

    def call_llama(self, messages, temperature=0.7, max_tokens=2000):
        prompt = self._format_messages(messages)
        
        data = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(self.ollama_url, json=data)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error calling Ollama: {response.text}")

    def _format_messages(self, messages):
        formatted_prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if isinstance(content, list):
                content_text = " ".join([item["text"] for item in content if item["type"] == "text"])
            else:
                content_text = content
                
            formatted_prompt += f"{role.upper()}: {content_text}\n"
        return formatted_prompt

class SpeakerIdentificationAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="SpeakerIdentificationAgent", max_retries=max_retries, verbose=verbose)
        self.speaker_mapping = {}
        self.court_roles = {
            'judge': [
                'let\'s know whether', 'let me clarify', 'so let\'s come to',
                'must be dealt with', 'when you look at', 'i think that is actually',
                'let\'s come to', 'let me clarify'
            ],
            'attorney': [
                'we brought it to you', 'my lord', 'my submission',
                'we submit', 'our case', 'our argument', 'we argue'
            ]
        }
        
        # Enhanced dialog patterns for legal proceedings
        self.dialog_patterns = {
            'judge_questioning': [
                (r'(?:is|was) it a (?:complaint|case|matter) that', 'judge'),
                (r'(?:let|must) (?:us|me|\'s) (?:know|clarify|understand)', 'judge'),
                (r'so (?:is|was) it', 'judge')
            ],
            'attorney_response': [
                (r'we brought it to you', 'attorney'),
                (r'(?:we|I) submit that', 'attorney'),
                (r'our (?:case|argument|submission)', 'attorney')
            ],
            'legal_discussion': [
                (r'(?:violated|breach|contempt of) (?:the constitution|court order)', 'attorney'),
                (r'(?:serious|judicial) misconduct', 'judge'),
                (r'administration of justice', 'judge')
            ]
        }
        
        # Legal context patterns
        self.legal_context = {
            'judge_indicators': [
                'clarify', 'let\'s', 'must be dealt with', 'look at',
                'that is actually', 'come to the', 'serious misconduct'
            ],
            'attorney_indicators': [
                'brought it to you', 'we submit', 'our case',
                'we argue', 'beneficiaries', 'responded'
            ]
        }
        
    def identify_speakers(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Enhanced speaker identification with improved conversation flow analysis"""
        processed_segments = []
        window_size = 3  # Look at previous and next segments for context
        
        # First pass: Initial role scoring
        for i, segment in enumerate(segments):
            segment = self._initial_role_scoring(segment, i, segments)
            processed_segments.append(segment)
        
        # Second pass: Refine based on conversation flow
        for i in range(len(processed_segments)):
            self._refine_speaker_role(processed_segments, i, window_size)
        
        # Third pass: Ensure consistency
        self._ensure_conversation_consistency(processed_segments)
        
        return processed_segments
    
    def _initial_role_scoring(self, segment: TranscriptSegment, index: int, all_segments: List[TranscriptSegment]) -> TranscriptSegment:
        """Enhanced role scoring for legal proceedings"""
        role_scores = {role: 0.0 for role in self.court_roles.keys()}
        text_lower = segment.text.lower()
        
        # 1. Legal phrase analysis
        for role, phrases in self.court_roles.items():
            for phrase in phrases:
                if phrase.lower() in text_lower:
                    role_scores[role] += 2.0
        
        # 2. Context indicators
        for indicator in self.legal_context['judge_indicators']:
            if indicator.lower() in text_lower:
                role_scores['judge'] += 1.0
        for indicator in self.legal_context['attorney_indicators']:
            if indicator.lower() in text_lower:
                role_scores['attorney'] += 1.0
        
        # 3. Dialog pattern analysis
        for category, patterns in self.dialog_patterns.items():
            for pattern, role in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    role_scores[role] += 1.5
        
        # 4. Contextual analysis
        if index > 0:
            prev_segment = all_segments[index-1]
            if hasattr(prev_segment, 'speaker_role') and prev_segment.speaker_role:
                # Question-Answer pattern
                if prev_segment.speaker_role == 'judge' and '?' in prev_segment.text:
                    role_scores['attorney'] += 1.0
                # Response pattern
                elif prev_segment.speaker_role == 'attorney' and not '?' in text_lower:
                    role_scores['judge'] += 1.0
        
        # 5. Determine role with highest confidence
        max_score = max(role_scores.values())
        if max_score == 0:
            # Default handling
            if '?' in text_lower or any(p in text_lower for p in self.legal_context['judge_indicators']):
                most_likely_role = 'judge'
            else:
                most_likely_role = 'attorney'
            role_scores[most_likely_role] = 0.5
        else:
            most_likely_role = max(role_scores.items(), key=lambda x: x[1])[0]
        
        # Update segment
        segment.speaker_role = most_likely_role
        segment.confidence = max_score / (sum(role_scores.values()) or 1)
        
        # Update speaker mapping
        speaker_key = f"{most_likely_role}_{len(self.speaker_mapping)}"
        if segment.speaker not in self.speaker_mapping:
            self.speaker_mapping[segment.speaker] = speaker_key
        
        return segment
    
    def _refine_speaker_role(self, segments: List[TranscriptSegment], index: int, window_size: int) -> None:
        """Refine speaker role based on conversation context"""
        current_segment = segments[index]
        window_start = max(0, index - window_size)
        window_end = min(len(segments), index + window_size + 1)
        context_segments = segments[window_start:window_end]
        
        # Analyze conversation flow
        if len(context_segments) >= 3:
            # Check for typical court interaction patterns
            if current_segment.speaker_role == 'attorney':
                # Attorneys typically address the judge
                if any(s.speaker_role == 'judge' for s in context_segments):
                    current_segment.confidence += 0.2
            elif current_segment.speaker_role == 'judge':
                # Judges typically respond to attorneys
                if any(s.speaker_role == 'attorney' for s in context_segments):
                    current_segment.confidence += 0.2
        
        # Update speaker mapping for consistency
        speaker_key = f"{current_segment.speaker_role}_{len(self.speaker_mapping)}"
        if current_segment.speaker not in self.speaker_mapping:
            self.speaker_mapping[current_segment.speaker] = speaker_key
        current_segment.speaker = self.speaker_mapping[current_segment.speaker]
    
    def _ensure_conversation_consistency(self, segments: List[TranscriptSegment]) -> None:
        """Ensure consistent speaker roles throughout the conversation"""
        # Track speaker role changes
        role_changes = []
        for i in range(1, len(segments)):
            if segments[i].speaker_role != segments[i-1].speaker_role:
                role_changes.append(i)
        
        # Analyze and correct unlikely rapid role changes
        for i in range(len(role_changes)-1):
            if role_changes[i+1] - role_changes[i] == 1:  # Adjacent role changes
                # Keep the role with higher confidence
                if segments[role_changes[i]].confidence < segments[role_changes[i+1]].confidence:
                    segments[role_changes[i]].speaker_role = segments[role_changes[i+1]].speaker_role
                    segments[role_changes[i]].speaker = segments[role_changes[i+1]].speaker
                else:
                    segments[role_changes[i+1]].speaker_role = segments[role_changes[i]].speaker_role
                    segments[role_changes[i+1]].speaker = segments[role_changes[i]].speaker

class MetadataAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="MetadataAgent", max_retries=max_retries, verbose=verbose)
        
    def extract_metadata(self, transcript_text: str) -> TranscriptMetadata:
        messages = [
            {
                "role": "system",
                "content": """Extract court case metadata from the transcript text. Focus on:
                - Court name and type
                - Case number and title
                - Date and time
                - Location
                - Presiding judge
                - Involved parties"""
            },
            {
                "role": "user",
                "content": transcript_text
            }
        ]
        
        response = self.call_llama(messages, temperature=0.3)
        return self._parse_metadata(response)
    
    def _parse_metadata(self, response: str) -> TranscriptMetadata:
        # Implementation of metadata parsing logic
        pass

class TimestampManagementAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="TimestampManagementAgent", max_retries=max_retries, verbose=verbose)
        
    def process_timestamps(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Process and validate timestamps for all segments"""
        processed_segments = []
        for i, segment in enumerate(segments):
            # Validate timestamp format and sequence
            if i > 0:
                if segment.start_time < processed_segments[-1].end_time:
                    # Handle overlapping speech
                    segment.is_overlapping = True
            
            # Format timestamps consistently
            segment.start_time = self._normalize_timestamp(segment.start_time)
            segment.end_time = self._normalize_timestamp(segment.end_time)
            processed_segments.append(segment)
        
        return processed_segments
    
    def _normalize_timestamp(self, time_value: float) -> float:
        """Ensure timestamp is in the correct format and range"""
        return round(max(0, float(time_value)), 3)

class FormattingStructureAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="FormattingStructureAgent", max_retries=max_retries, verbose=verbose)
    
    def format_transcript(self, segments: List[TranscriptSegment], speaker_mapping: Dict[str, str]) -> str:
        """Format transcript with improved speaker role handling"""
        formatted_lines = []
        current_speaker = None
        
        for segment in segments:
            timestamp = self._format_timestamp(segment.start_time)
            
            # Get speaker name from mapping or use original
            speaker_name = speaker_mapping.get(segment.speaker, segment.speaker)
            if segment.speaker_role and segment.speaker_role != 'unknown':
                speaker_name = f"{speaker_name} ({segment.speaker_role.title()})"
            
            # Only add speaker name if it changes
            if speaker_name != current_speaker:
                formatted_lines.append(f"\n[{timestamp}] {speaker_name}:  {segment.text}")
                current_speaker = speaker_name
            else:
                formatted_lines.append(f"[{timestamp}] {segment.text}")
        
        return "\n".join(formatted_lines)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class LanguageGrammarRefinementAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="LanguageGrammarRefinementAgent", max_retries=max_retries, verbose=verbose)
        self.legal_markers = {
            r'\(unclear\)': '[inaudible]',
            r'\(inaudible\)': '[inaudible]',
            r'\(crosstalk\)': '[multiple speakers]',
            r'\(noise\)': '[background noise]',
            r'\(pause\)': '[pause in proceedings]'
        }

    def refine_text(self, text: str) -> str:
        """Standardize transcript markers while preserving content."""
        refined = text
        for pattern, replacement in self.legal_markers.items():
            refined = re.sub(pattern, replacement, refined, flags=re.IGNORECASE)
        return refined

class QualityAssuranceAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="QualityAssuranceAgent", max_retries=max_retries, verbose=verbose)

    def validate_transcript(self, segments: List[TranscriptSegment], speaker_mapping: Dict[str, str]) -> Dict:
        messages = [
            {
                "role": "system",
                "content": """You are an AI assistant that validates Eswatini Judiciary court transcripts.
                Focus on:
                1. Technical accuracy (audio quality, timing)
                2. Speaker consistency and role accuracy
                3. Procedural completeness
                4. Legal terminology accuracy
                Do not suggest content modifications; only identify potential issues."""
            },
            {
                "role": "user",
                "content": self._prepare_validation_prompt(segments, speaker_mapping)
            }
        ]
        
        response = self.call_llama(messages, temperature=0.3)
        parsed_report = self._parse_quality_report(response)
        
        report = {
            'total_segments': len(segments),
            'speaker_consistency': parsed_report.get('speaker_consistency', True),
            'timing_issues': parsed_report.get('timing_issues', []),
            'unknown_speakers': set(parsed_report.get('unknown_speakers', [])),
            'low_confidence_segments': parsed_report.get('low_confidence_segments', []),
            'report_summary': [],
            'segments_with_issues': [],
            'legal_terminology_issues': []  # Track potential legal term inconsistencies
        }
        
        report['report_summary'] = [
            f"Processed {report['total_segments']} transcript segments",
        ]
        
        if 'findings' in parsed_report:
            report['report_summary'].extend(parsed_report['findings'])
        
        # Add segments that need review
        for i, segment in enumerate(segments):
            issues = []
            if segment.confidence < 0.7:
                issues.append('Low confidence')
            if segment.speaker not in speaker_mapping:
                issues.append('Unidentified speaker')
            
            if issues:
                report['segments_with_issues'].append({
                    'index': i,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'reasons': issues,
                    'speaker': segment.speaker,
                    'text': segment.original_text
                })
        
        return report

    def _prepare_validation_prompt(self, segments, speaker_mapping):
        prompt = """Review the following Eswatini court transcript segments for quality issues.
        Focus on:
        - Speaker role consistency
        - Procedural completeness
        - Audio quality issues
        - Legal terminology accuracy
        Do NOT suggest content changes.\n\n"""
        
        for segment in segments:
            speaker_role = speaker_mapping.get(segment.speaker, segment.speaker)
            timestamp = self._format_timestamp(segment.start_time)
            prompt += f"[{timestamp}] {speaker_role}: {segment.original_text}\n"
        
        return prompt

    def _parse_quality_report(self, response):
        report = {
            'findings': [],
            'speaker_consistency': True,
            'timing_issues': [],
            'unknown_speakers': [],
            'low_confidence_segments': [],
            'legal_terminology_issues': []
        }
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                report['findings'].append(line[2:])
                
                lower_line = line.lower()
                if 'speaker' in lower_line and ('unknown' in lower_line or 'unidentified' in lower_line):
                    report['speaker_consistency'] = False
                if 'timing' in lower_line and 'issue' in lower_line:
                    report['timing_issues'].append(line[2:])
                if 'confidence' in lower_line and 'low' in lower_line:
                    report['low_confidence_segments'].append(line[2:])
                if 'legal term' in lower_line or 'terminology' in lower_line:
                    report['legal_terminology_issues'].append(line[2:])
        
        return report

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class ValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="ValidatorAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, topic, article):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that validates research articles for accuracy, completeness, and adherence to academic standards."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Given the topic and the research article below, assess whether the article comprehensively covers the topic, follows a logical structure, and maintains academic standards.\n"
                            "Provide a brief analysis and rate the article on a scale of 1 to 5, where 5 indicates excellent quality.\n\n"
                            f"Topic: {topic}\n\n"
                            f"Article:\n{article}\n\n"
                            "Validation:"
                        )
                    }
                ]
            }
        ]
        validation = self.call_llama(
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        return validation
