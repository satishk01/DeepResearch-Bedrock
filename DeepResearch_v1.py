import streamlit as st
import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import logging
from urllib.parse import quote_plus, urljoin
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    relevance_score: float
    source_type: str  # 'web', 'wikipedia', 'academic', 'news'
    timestamp: datetime
    domain: str = ""
    
    def __post_init__(self):
        if self.url:
            from urllib.parse import urlparse
            self.domain = urlparse(self.url).netloc

@dataclass
class ResearchInsight:
    category: str
    insight: str
    confidence: float
    supporting_sources: List[str]

@dataclass
class ResearchResponse:
    query: str
    synthesized_answer: str
    sources: List[SearchResult]
    claude_knowledge: str
    combined_analysis: str
    insights: List[ResearchInsight]
    search_time: float
    analysis_time: float
    timestamp: datetime
    total_sources_found: int
    unique_domains: int

class SimplifiedInternetSearcher:
    """Simplified internet search without aiohttp dependency"""
    
    def __init__(self, serper_api_key: str = None):
        self.serper_api_key = serper_api_key
        self.cache = {}
        self.cache_ttl = timedelta(minutes=30)
        
    def _get_cache_key(self, query: str, search_type: str) -> str:
        """Generate cache key for query"""
        return hashlib.md5(f"{query}_{search_type}".encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        return datetime.now() - cache_entry['timestamp'] < self.cache_ttl
    
    def search_serper(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search using Serper API (synchronous version)"""
        if not self.serper_api_key:
            return []
            
        cache_key = self._get_cache_key(query, "serper")
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            return self.cache[cache_key]['data']
        
        try:
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'q': query,
                'num': num_results,
                'gl': 'us',
                'hl': 'en'
            }
            
            response = requests.post(
                'https://google.serper.dev/search',
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Parse organic results
                for item in data.get('organic', []):
                    results.append(SearchResult(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        relevance_score=0.8,
                        source_type='web',
                        timestamp=datetime.now()
                    ))
                
                # Parse news results if available
                for item in data.get('news', []):
                    results.append(SearchResult(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        relevance_score=0.85,
                        source_type='news',
                        timestamp=datetime.now()
                    ))
                
                # Cache results
                self.cache[cache_key] = {
                    'data': results,
                    'timestamp': datetime.now()
                }
                
                return results
            else:
                logger.error(f"Serper API error: {response.status_code}")
                return []
                        
        except Exception as e:
            logger.error(f"Serper search error: {e}")
            return []
    
    def search_duckduckgo_enhanced(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Enhanced DuckDuckGo search with better parsing"""
        cache_key = self._get_cache_key(query, "duckduckgo")
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            return self.cache[cache_key]['data']
        
        try:
            # Use DuckDuckGo instant answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            results = []
            
            # Extract instant answer
            if data.get('AbstractText'):
                results.append(SearchResult(
                    title=data.get('AbstractSource', 'DuckDuckGo'),
                    url=data.get('AbstractURL', ''),
                    snippet=data.get('AbstractText', ''),
                    relevance_score=0.9,
                    source_type='web',
                    timestamp=datetime.now()
                ))
            
            # Extract related topics
            for topic in data.get('RelatedTopics', [])[:num_results-1]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append(SearchResult(
                        title=topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else 'Related',
                        url=topic.get('FirstURL', ''),
                        snippet=topic.get('Text', ''),
                        relevance_score=0.7,
                        source_type='web',
                        timestamp=datetime.now()
                    ))
            
            # Cache results
            self.cache[cache_key] = {
                'data': results,
                'timestamp': datetime.now()
            }
            
            return results[:num_results]
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def search_wikipedia_enhanced(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Enhanced Wikipedia search with multiple endpoints"""
        cache_key = self._get_cache_key(query, "wikipedia")
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            return self.cache[cache_key]['data']
        
        results = []
        
        try:
            # Search for related articles
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': num_results,
                'srprop': 'snippet|titlesnippet|size'
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            data = response.json()
            
            for item in data.get('query', {}).get('search', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=f"https://en.wikipedia.org/wiki/{quote_plus(item.get('title', ''))}",
                    snippet=re.sub(r'<[^>]+>', '', item.get('snippet', '')),
                    relevance_score=0.75,
                    source_type='wikipedia',
                    timestamp=datetime.now()
                ))
            
            # Cache results
            self.cache[cache_key] = {
                'data': results,
                'timestamp': datetime.now()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Wikipedia enhanced search error: {e}")
            return results
    
    def search_arxiv(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search arXiv for academic papers"""
        try:
            import xml.etree.ElementTree as ET
            
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': num_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                results = []
                
                # Parse XML response
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                    summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                    id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                    
                    if title_elem is not None and summary_elem is not None:
                        results.append(SearchResult(
                            title=title_elem.text.strip(),
                            url=id_elem.text if id_elem is not None else '',
                            snippet=summary_elem.text.strip()[:300] + '...',
                            relevance_score=0.85,
                            source_type='academic',
                            timestamp=datetime.now()
                        ))
                
                return results
                
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []
    
    def search_comprehensive(self, query: str, num_results: int = 20) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """Comprehensive search across all available sources"""
        start_time = time.time()
        all_results = []
        search_stats = {
            'serper_count': 0,
            'duckduckgo_count': 0,
            'wikipedia_count': 0,
            'arxiv_count': 0,
            'total_time': 0
        }
        
        # Sequential search execution (simplified)
        try:
            # Serper search
            if self.serper_api_key:
                serper_results = self.search_serper(query, num_results // 2)
                all_results.extend(serper_results)
                search_stats['serper_count'] = len(serper_results)
            
            # DuckDuckGo search
            ddg_results = self.search_duckduckgo_enhanced(query, num_results // 3)
            all_results.extend(ddg_results)
            search_stats['duckduckgo_count'] = len(ddg_results)
            
            # Wikipedia search
            wiki_results = self.search_wikipedia_enhanced(query, 5)
            all_results.extend(wiki_results)
            search_stats['wikipedia_count'] = len(wiki_results)
            
            # arXiv search for academic content
            arxiv_results = self.search_arxiv(query, 3)
            all_results.extend(arxiv_results)
            search_stats['arxiv_count'] = len(arxiv_results)
            
        except Exception as e:
            logger.error(f"Search task error: {e}")
        
        # Remove duplicates and rank results
        unique_results = self._deduplicate_and_rank(all_results)
        
        # Update search stats
        search_stats['total_time'] = time.time() - start_time
        
        return unique_results[:num_results], search_stats
    
    def _deduplicate_and_rank(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicates and rank results by relevance"""
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url and result.url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result.url)
        
        # Sort by relevance score and source type priority
        source_priority = {'academic': 3, 'news': 2, 'wikipedia': 1, 'web': 0}
        
        unique_results.sort(
            key=lambda x: (source_priority.get(x.source_type, 0), x.relevance_score),
            reverse=True
        )
        
        return unique_results

class EnhancedBedrockClient:
    """Enhanced AWS Bedrock client with retry logic and error handling"""
    
    def __init__(self, region_name: str = 'us-east-1'):
        self.region_name = region_name
        self.bedrock_runtime = None
        self.max_retries = 3
        self.retry_delay = 1
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Bedrock client with error handling"""
        try:
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region_name
            )
            logger.info("Bedrock client initialized successfully")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            st.error("AWS credentials not configured. Please check your AWS setup.")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            st.error(f"Failed to connect to AWS Bedrock: {str(e)}")
    
    def invoke_claude_with_retry(self, prompt: str, model_id: str, max_tokens: int = 4000) -> str:
        """Invoke Claude with retry logic"""
        if not self.bedrock_runtime:
            return "Error: Bedrock client not initialized"
        
        for attempt in range(self.max_retries):
            try:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "top_p": 0.9
                })
                
                response = self.bedrock_runtime.invoke_model(
                    body=body,
                    modelId=model_id,
                    accept='application/json',
                    contentType='application/json'
                )
                
                response_body = json.loads(response.get('body').read())
                return response_body.get('content')[0].get('text')
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ThrottlingException' and attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Throttling detected, waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Bedrock client error: {e}")
                    return f"Error: {str(e)}"
            except Exception as e:
                logger.error(f"Unexpected error in Bedrock invocation: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return f"Error invoking Claude: {str(e)}"
        
        return "Error: Max retries exceeded"

class SimplifiedResearchAgent:
    """Simplified research agent without async dependencies"""
    
    def __init__(self, bedrock_client: EnhancedBedrockClient, searcher: SimplifiedInternetSearcher):
        self.bedrock_client = bedrock_client
        self.searcher = searcher
    
    def extract_insights(self, research_content: str, sources: List[SearchResult]) -> List[ResearchInsight]:
        """Extract key insights from research content"""
        insights_prompt = f"""
        Analyze the following research content and extract 5-7 key insights. For each insight, provide:
        1. Category (e.g., "Trends", "Challenges", "Opportunities", "Technology", "Market", "Social Impact")
        2. The insight itself (concise but informative)
        3. Confidence level (0.0-1.0)
        
        Research Content:
        {research_content[:3000]}  # Limit content to avoid token limits
        
        Format your response as JSON:
        [
            {{"category": "Category", "insight": "Insight text", "confidence": 0.8}},
            ...
        ]
        """
        
        try:
            insights_response = self.bedrock_client.invoke_claude_with_retry(
                insights_prompt, 
                "anthropic.claude-3-haiku-20240307-v1:0",  # Use faster model for insights
                max_tokens=2000
            )
            
            # Parse JSON response
            insights_data = json.loads(insights_response)
            insights = []
            
            for item in insights_data:
                # Find supporting sources
                supporting_sources = [
                    source.url for source in sources[:5]  # Top 5 sources
                    if any(word in source.snippet.lower() for word in item['insight'].lower().split()[:3])
                ]
                
                insights.append(ResearchInsight(
                    category=item['category'],
                    insight=item['insight'],
                    confidence=item['confidence'],
                    supporting_sources=supporting_sources[:3]
                ))
                
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return []
    
    def conduct_advanced_research(self, query: str, model_id: str, search_limit: int = 20) -> ResearchResponse:
        """Conduct advanced research with comprehensive analysis"""
        start_time = time.time()
        
        # Step 1: Get Claude's initial knowledge
        claude_prompt = f"""
        Provide a comprehensive analysis of: {query}
        
        Include:
        1. Core concepts and definitions
        2. Current state and context
        3. Key stakeholders and entities involved
        4. Recent developments (based on training data)
        5. Critical questions that need current information
        6. Potential areas of rapid change
        
        Be thorough and analytical. Highlight areas where recent information would be most valuable.
        """
        
        analysis_start = time.time()
        claude_knowledge = self.bedrock_client.invoke_claude_with_retry(claude_prompt, model_id)
        
        # Step 2: Comprehensive internet search
        search_start = time.time()
        search_results, search_stats = self.searcher.search_comprehensive(query, search_limit)
        search_time = time.time() - search_start
        
        # Step 3: Advanced synthesis
        if search_results:
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results[:15]):  # Limit to top 15 for context
                context_parts.append(f"""
                Source {i+1} ({result.source_type}):
                Title: {result.title}
                URL: {result.url}
                Content: {result.snippet}
                Domain: {result.domain}
                Relevance: {result.relevance_score:.2f}
                """)
            
            internet_context = "\n".join(context_parts)
            
            synthesis_prompt = f"""
            Research Topic: {query}
            
            My Knowledge Base Analysis:
            {claude_knowledge}
            
            Current Internet Sources:
            {internet_context}
            
            Provide a comprehensive research report with:
            
            1. EXECUTIVE SUMMARY (2-3 paragraphs)
            
            2. KEY FINDINGS
            - Synthesize the most important discoveries
            - Highlight new developments not in my training data
            - Note any contradictions between sources
            
            3. DETAILED ANALYSIS
            - Integrate knowledge base and current information
            - Provide context and implications
            - Discuss trends and patterns
            
            4. SOURCE RELIABILITY ASSESSMENT
            - Evaluate the quality and credibility of sources
            - Note any potential biases or limitations
            
            5. RESEARCH GAPS AND LIMITATIONS
            - Identify areas needing further investigation
            - Note information that couldn't be verified
            
            6. CONCLUSIONS AND IMPLICATIONS
            - Synthesize overall findings
            - Discuss broader implications
            - Suggest actionable insights
            
            Use clear headings and maintain academic rigor while being accessible.
            """
            
            synthesized_answer = self.bedrock_client.invoke_claude_with_retry(
                synthesis_prompt, model_id, max_tokens=4000
            )
        else:
            synthesized_answer = f"""
            # Research Report: {query}
            
            ## Executive Summary
            This analysis is based solely on my training data as no additional internet sources were found.
            
            ## Analysis
            {claude_knowledge}
            
            ## Limitations
            This research is limited to information available in my training data. Current developments, recent news, or real-time information may not be reflected in this analysis.
            """
        
        # Step 4: Extract insights
        insights = self.extract_insights(synthesized_answer, search_results)
        
        # Step 5: Generate combined analysis with metrics
        analysis_time = time.time() - analysis_start
        
        combined_analysis_prompt = f"""
        Based on the comprehensive research conducted on "{query}", provide:
        
        1. RESEARCH QUALITY ASSESSMENT
        - Overall confidence level (High/Medium/Low)
        - Source diversity score
        - Information recency evaluation
        
        2. KEY INSIGHTS SUMMARY
        - Top 3 most important findings
        - Emerging trends identified
        - Critical knowledge gaps
        
        3. ACTIONABLE RECOMMENDATIONS
        - Next steps for further research
        - Key areas to monitor
        - Practical applications
        
        4. METHODOLOGY NOTES
        - Search strategy effectiveness
        - Source types utilized
        - Analysis limitations
        
        Research conducted with {len(search_results)} sources across {len(set(r.source_type for r in search_results))} different source types.
        """
        
        combined_analysis = self.bedrock_client.invoke_claude_with_retry(
            combined_analysis_prompt, model_id, max_tokens=2000
        )
        
        total_time = time.time() - start_time
        
        return ResearchResponse(
            query=query,
            synthesized_answer=synthesized_answer,
            sources=search_results,
            claude_knowledge=claude_knowledge,
            combined_analysis=combined_analysis,
            insights=insights,
            search_time=search_time,
            analysis_time=analysis_time,
            timestamp=datetime.now(),
            total_sources_found=len(search_results),
            unique_domains=len(set(r.domain for r in search_results if r.domain))
        )

# Role-based authentication system
class RoleBasedAuth:
    ROLES = {
        'admin': {
            'bedrock_models': ['claude-3-5-sonnet', 'claude-3-opus', 'claude-3-haiku'],
            'search_limit': 30,
            'api_access': True
        },
        'researcher': {
            'bedrock_models': ['claude-3-5-sonnet', 'claude-3-haiku'],
            'search_limit': 20,
            'api_access': False
        },
        'basic': {
            'bedrock_models': ['claude-3-haiku'],
            'search_limit': 10,
            'api_access': False
        }
    }
    
    @staticmethod
    def authenticate(username: str, password: str) -> Optional[str]:
        credentials = {
            'admin_user': ('admin123', 'admin'),
            'researcher_user': ('research123', 'researcher'),
            'basic_user': ('basic123', 'basic')
        }
        
        for user, (pwd, role) in credentials.items():
            if username == user and password == pwd:
                return role
        return None
    
    @staticmethod
    def get_permissions(role: str) -> Dict[str, Any]:
        return RoleBasedAuth.ROLES.get(role, RoleBasedAuth.ROLES['basic'])

class PDFReportGenerator:
    """Generate comprehensive PDF reports from research data"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        custom_styles = {}
        
        # Title style
        custom_styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1f4e79')
        )
        
        # Heading style
        custom_styles['CustomHeading'] = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#2d5aa0')
        )
        
        # Subheading style
        custom_styles['CustomSubheading'] = ParagraphStyle(
            'CustomSubheading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#4472c4')
        )
        
        # Body text style
        custom_styles['CustomBody'] = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            leftIndent=0,
            rightIndent=0
        )
        
        # Quote style
        custom_styles['Quote'] = ParagraphStyle(
            'Quote',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=8,
            spaceAfter=8,
            leftIndent=20,
            rightIndent=20,
            fontName='Helvetica-Oblique',
            textColor=colors.HexColor('#666666')
        )
        
        return custom_styles
    
    def _create_matplotlib_chart(self, research_response: ResearchResponse, chart_type: str) -> BytesIO:
        """Create matplotlib charts for the PDF"""
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if chart_type == 'source_distribution':
            source_types = [r.source_type for r in research_response.sources]
            source_counts = pd.Series(source_types).value_counts()
            
            colors_list = ['#1f4e79', '#2d5aa0', '#4472c4', '#5b85d6', '#7da7e8']
            ax.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', 
                   colors=colors_list[:len(source_counts)])
            ax.set_title('Source Type Distribution', fontsize=14, fontweight='bold')
            
        elif chart_type == 'relevance_scores':
            relevance_scores = [r.relevance_score for r in research_response.sources]
            ax.hist(relevance_scores, bins=10, color='#4472c4', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Relevance Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Source Relevance Score Distribution', fontsize=14, fontweight='bold')
            
        elif chart_type == 'insights_confidence':
            if research_response.insights:
                categories = [i.category for i in research_response.insights]
                confidences = [i.confidence for i in research_response.insights]
                
                ax.barh(categories, confidences, color='#2d5aa0', alpha=0.7)
                ax.set_xlabel('Confidence Level')
                ax.set_title('Research Insights by Category and Confidence', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No insights available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title('Research Insights', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    
    def _clean_text_for_pdf(self, text: str) -> str:
        """Clean and format text for PDF generation"""
        # Remove or replace problematic characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()
        
        # Handle markdown-like formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)  # Italic
        text = re.sub(r'#{1,6}\s*(.*)', r'<b>\1</b>', text)  # Headers
        
        return text
    
# Continuation of the PDFReportGenerator class and remaining application code

    def generate_pdf_report(self, research_response: ResearchResponse, user_role: str) -> BytesIO:
        """Generate comprehensive PDF report from research data"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Build the story (content) for the PDF
        story = []
        
        # Title Page
        title = Paragraph(f"Research Report: {research_response.query}", 
                         self.custom_styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 30))
        
        # Executive Summary
        exec_summary_data = [
            ['Report Generated:', research_response.timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Total Sources Found:', str(research_response.total_sources_found)],
            ['Unique Domains:', str(research_response.unique_domains)],
            ['Search Time:', f"{research_response.search_time:.2f} seconds"],
            ['Analysis Time:', f"{research_response.analysis_time:.2f} seconds"],
            ['User Role:', user_role.title()]
        ]
        
        exec_table = Table(exec_summary_data, colWidths=[2*inch, 3*inch])
        exec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
        ]))
        
        story.append(exec_table)
        story.append(Spacer(1, 20))
        
        # Main Research Content
        story.append(Paragraph("Research Analysis", self.custom_styles['CustomHeading']))
        
        # Split the synthesized answer into paragraphs and clean them
        content_paragraphs = research_response.synthesized_answer.split('\n\n')
        for para in content_paragraphs:
            if para.strip():
                cleaned_para = self._clean_text_for_pdf(para.strip())
                if cleaned_para:
                    story.append(Paragraph(cleaned_para, self.custom_styles['CustomBody']))
                    story.append(Spacer(1, 6))
        
        story.append(PageBreak())
        
        # Research Insights Section
        if research_response.insights:
            story.append(Paragraph("Key Research Insights", self.custom_styles['CustomHeading']))
            
            insights_data = [['Category', 'Insight', 'Confidence']]
            for insight in research_response.insights:
                insights_data.append([
                    insight.category,
                    insight.insight[:100] + "..." if len(insight.insight) > 100 else insight.insight,
                    f"{insight.confidence:.2f}"
                ])
            
            insights_table = Table(insights_data, colWidths=[1.5*inch, 3.5*inch, 1*inch])
            insights_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            
            story.append(insights_table)
            story.append(Spacer(1, 20))
        
        # Add charts if sources are available
        if research_response.sources:
            try:
                # Source distribution chart
                chart_buffer = self._create_matplotlib_chart(research_response, 'source_distribution')
                chart_img = Image(chart_buffer, width=4*inch, height=3*inch)
                story.append(Paragraph("Source Analysis", self.custom_styles['CustomHeading']))
                story.append(chart_img)
                story.append(Spacer(1, 10))
                
                # Relevance scores chart
                rel_chart_buffer = self._create_matplotlib_chart(research_response, 'relevance_scores')
                rel_chart_img = Image(rel_chart_buffer, width=4*inch, height=3*inch)
                story.append(rel_chart_img)
                story.append(Spacer(1, 20))
                
            except Exception as e:
                logger.warning(f"Could not generate charts for PDF: {e}")
        
        # Sources Section
        if research_response.sources:
            story.append(PageBreak())
            story.append(Paragraph("Research Sources", self.custom_styles['CustomHeading']))
            
            for i, source in enumerate(research_response.sources[:20], 1):  # Limit to top 20
                story.append(Paragraph(f"{i}. {source.title}", self.custom_styles['CustomSubheading']))
                story.append(Paragraph(f"Source: {source.domain} ({source.source_type})", 
                                     self.custom_styles['CustomBody']))
                story.append(Paragraph(f"URL: {source.url}", self.custom_styles['CustomBody']))
                
                cleaned_snippet = self._clean_text_for_pdf(source.snippet)
                story.append(Paragraph(cleaned_snippet, self.custom_styles['Quote']))
                story.append(Spacer(1, 10))
        
        # Combined Analysis Section
        if research_response.combined_analysis:
            story.append(PageBreak())
            story.append(Paragraph("Research Quality Assessment", self.custom_styles['CustomHeading']))
            
            analysis_paragraphs = research_response.combined_analysis.split('\n\n')
            for para in analysis_paragraphs:
                if para.strip():
                    cleaned_para = self._clean_text_for_pdf(para.strip())
                    if cleaned_para:
                        story.append(Paragraph(cleaned_para, self.custom_styles['CustomBody']))
                        story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

class StreamlitUI:
    """Main Streamlit user interface"""
    
    def __init__(self):
        self.setup_page_config()
        self.setup_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Advanced Research Agent",
            page_icon="??",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        if 'research_history' not in st.session_state:
            st.session_state.research_history = []
        if 'bedrock_client' not in st.session_state:
            st.session_state.bedrock_client = None
        if 'searcher' not in st.session_state:
            st.session_state.searcher = None
    
    def render_login_form(self):
        """Render the login form"""
        st.title("?? Advanced Research Agent")
        st.markdown("---")
        
        with st.form("login_form"):
            st.subheader("Authentication Required")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                role = RoleBasedAuth.authenticate(username, password)
                if role:
                    st.session_state.authenticated = True
                    st.session_state.user_role = role
                    st.success(f"Successfully logged in as {role.title()}")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        # Display demo credentials
        with st.expander("Demo Credentials"):
            st.code("""
            Admin User: admin_user / admin123
            Researcher: researcher_user / research123
            Basic User: basic_user / basic123
            """)
    
    def render_sidebar_config(self):
        """Render sidebar configuration"""
        with st.sidebar:
            st.header("Configuration")
            
            # User info
            st.info(f"Logged in as: **{st.session_state.user_role.title()}**")
            
            # AWS Configuration
            st.subheader("AWS Bedrock Setup")
            aws_region = st.selectbox(
                "AWS Region",
                ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
                index=0
            )
            
            # API Keys
            st.subheader("API Configuration")
            serper_key = st.text_input(
                "Serper API Key (Optional)",
                type="password",
                help="Get your free API key from serper.dev"
            )
            
            # Model Selection
            permissions = RoleBasedAuth.get_permissions(st.session_state.user_role)
            available_models = {
                'claude-3-5-sonnet': 'anthropic.claude-3-5-sonnet-20241022-v2:0',
                'claude-3-opus': 'anthropic.claude-3-opus-20240229-v1:0',
                'claude-3-haiku': 'anthropic.claude-3-haiku-20240307-v1:0'
            }
            
            allowed_models = {k: v for k, v in available_models.items() 
                            if k in permissions['bedrock_models']}
            
            selected_model = st.selectbox(
                "Claude Model",
                options=list(allowed_models.keys()),
                format_func=lambda x: x.replace('-', ' ').title()
            )
            
            # Search Settings
            search_limit = st.slider(
                "Search Results Limit",
                min_value=5,
                max_value=permissions['search_limit'],
                value=min(15, permissions['search_limit']),
                help=f"Maximum {permissions['search_limit']} for your role"
            )
            
            # Initialize clients
            if st.button("Initialize Connections", type="primary"):
                with st.spinner("Initializing connections..."):
                    try:
                        # Initialize Bedrock client
                        st.session_state.bedrock_client = EnhancedBedrockClient(aws_region)
                        
                        # Initialize searcher
                        st.session_state.searcher = SimplifiedInternetSearcher(serper_key or None)
                        
                        st.success("Connections initialized successfully!")
                        
                        # Store configuration
                        st.session_state.model_id = allowed_models[selected_model]
                        st.session_state.search_limit = search_limit
                        
                    except Exception as e:
                        st.error(f"Initialization failed: {str(e)}")
            
            # Connection status
            if st.session_state.bedrock_client and st.session_state.searcher:
                st.success("? Ready for research")
            else:
                st.warning("?? Initialize connections first")
    
    def render_research_interface(self):
        """Render the main research interface"""
        st.title("?? Advanced Research Agent")
        st.markdown("Conduct comprehensive research using AI and internet sources")
        
        # Research query input
        query = st.text_area(
            "Research Query",
            height=100,
            placeholder="Enter your research question or topic here...",
            help="Be specific and detailed for better results"
        )
        
        # Research options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            include_analysis = st.checkbox("Include Deep Analysis", value=True)
        with col2:
            include_insights = st.checkbox("Extract Key Insights", value=True)
        with col3:
            generate_pdf = st.checkbox("Generate PDF Report", value=False)
        
        # Conduct research
        if st.button("?? Conduct Research", type="primary", disabled=not query):
            if not (st.session_state.bedrock_client and st.session_state.searcher):
                st.error("Please initialize connections in the sidebar first")
                return
            
            with st.spinner("Conducting comprehensive research..."):
                try:
                    # Create research agent
                    agent = SimplifiedResearchAgent(
                        st.session_state.bedrock_client,
                        st.session_state.searcher
                    )
                    
                    # Conduct research
                    research_response = agent.conduct_advanced_research(
                        query,
                        st.session_state.model_id,
                        st.session_state.search_limit
                    )
                    
                    # Store in history
                    st.session_state.research_history.append(research_response)
                    
                    # Display results
                    self.display_research_results(research_response, generate_pdf)
                    
                except Exception as e:
                    st.error(f"Research failed: {str(e)}")
                    logger.error(f"Research error: {e}")
    
    def display_research_results(self, research_response: ResearchResponse, generate_pdf: bool = False):
        """Display comprehensive research results"""
        
        # Research summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sources Found", research_response.total_sources_found)
        with col2:
            st.metric("Unique Domains", research_response.unique_domains)
        with col3:
            st.metric("Search Time", f"{research_response.search_time:.1f}s")
        with col4:
            st.metric("Analysis Time", f"{research_response.analysis_time:.1f}s")
        
        # Main research content
        st.markdown("## Research Analysis")
        st.markdown(research_response.synthesized_answer)
        
        # Research insights
        if research_response.insights:
            st.markdown("## Key Insights")
            
            for insight in research_response.insights:
                with st.expander(f"?? {insight.category} (Confidence: {insight.confidence:.2f})"):
                    st.write(insight.insight)
                    if insight.supporting_sources:
                        st.write("**Supporting Sources:**")
                        for source in insight.supporting_sources:
                            st.write(f"- {source}")
        
        # Combined analysis
        if research_response.combined_analysis:
            st.markdown("## Quality Assessment")
            with st.expander("View Research Quality Assessment"):
                st.markdown(research_response.combined_analysis)
        
        # Source visualization
        if research_response.sources:
            st.markdown("## Source Analysis")
            
            # Create source type distribution chart
            source_types = [r.source_type for r in research_response.sources]
            source_df = pd.DataFrame({'source_type': source_types})
            source_counts = source_df['source_type'].value_counts()
            
            fig = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Source Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Relevance scores distribution
            relevance_scores = [r.relevance_score for r in research_response.sources]
            fig2 = px.histogram(
                x=relevance_scores,
                title="Source Relevance Score Distribution",
                labels={'x': 'Relevance Score', 'y': 'Count'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed sources
        with st.expander(f"View All Sources ({len(research_response.sources)})"):
            for i, source in enumerate(research_response.sources, 1):
                st.markdown(f"**{i}. {source.title}**")
                st.markdown(f"*{source.domain}* ({source.source_type}) - Relevance: {source.relevance_score:.2f}")
                st.markdown(f"[?? Visit Source]({source.url})")
                st.markdown(f"> {source.snippet}")
                st.markdown("---")
        
        # PDF Generation
        if generate_pdf:
            try:
                pdf_generator = PDFReportGenerator()
                pdf_buffer = pdf_generator.generate_pdf_report(
                    research_response, 
                    st.session_state.user_role
                )
                
                st.download_button(
                    label="?? Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"PDF generation failed: {str(e)}")
    
    def render_research_history(self):
        """Render research history tab"""
        st.header("Research History")
        
        if not st.session_state.research_history:
            st.info("No research conducted yet.")
            return
        
        for i, research in enumerate(reversed(st.session_state.research_history)):
            with st.expander(f"?? {research.query} - {research.timestamp.strftime('%Y-%m-%d %H:%M')}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Query:** {research.query}")
                    st.write(f"**Sources:** {research.total_sources_found}")
                    st.write(f"**Domains:** {research.unique_domains}")
                    st.write(f"**Search Time:** {research.search_time:.2f}s")
                
                with col2:
                    if st.button(f"View Details", key=f"view_{i}"):
                        st.session_state.selected_research = research
                        st.rerun()
                    
                    # PDF download for history item
                    try:
                        pdf_generator = PDFReportGenerator()
                        pdf_buffer = pdf_generator.generate_pdf_report(
                            research, 
                            st.session_state.user_role
                        )
                        
                        st.download_button(
                            label="?? PDF",
                            data=pdf_buffer,
                            file_name=f"research_{i}_{research.timestamp.strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            key=f"pdf_{i}"
                        )
                    except Exception as e:
                        st.error(f"PDF error: {str(e)}")
    
    def run(self):
        """Main application runner"""
        # Authentication check
        if not st.session_state.authenticated:
            self.render_login_form()
            return
        
        # Sidebar configuration
        self.render_sidebar_config()
        
        # Main interface tabs
        tab1, tab2, tab3 = st.tabs(["?? Research", "?? History", "?? About"])
        
        with tab1:
            self.render_research_interface()
        
        with tab2:
            self.render_research_history()
        
        with tab3:
            st.header("About Advanced Research Agent")
            st.markdown("""
            This application provides comprehensive AI-powered research capabilities by combining:
            
            **?? AI Analysis**
            - Claude 3.5 Sonnet, Opus, and Haiku models via AWS Bedrock
            - Deep analysis and synthesis of information
            - Intelligent insight extraction
            
            **?? Internet Search**
            - Multiple search sources (Serper, DuckDuckGo, Wikipedia, arXiv)
            - Comprehensive source aggregation
            - Real-time information gathering
            
            **?? Advanced Features**
            - Role-based access control
            - Comprehensive PDF reports
            - Source quality analysis
            - Research history tracking
            - Interactive visualizations
            
            **?? User Roles**
            - **Admin**: Full access to all models and features
            - **Researcher**: Access to Sonnet and Haiku models
            - **Basic**: Access to Haiku model with limited search results
            
            **?? Setup Requirements**
            - AWS Bedrock access with Claude models enabled
            - Optional: Serper API key for enhanced search
            - Proper AWS credentials configuration
            """)
            
            # Logout button
            if st.button("?? Logout"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

def main():
    """Main application entry point"""
    try:
        app = StreamlitUI()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Main application error: {e}")

if __name__ == "__main__":
    main()        