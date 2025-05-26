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

def create_visualizations(research_response: ResearchResponse):
    """Create visualizations for research results"""
    
    # Source type distribution
    source_types = [r.source_type for r in research_response.sources]
    source_df = pd.DataFrame({'source_type': source_types})
    source_counts = source_df['source_type'].value_counts()
    
    fig1 = px.pie(
        values=source_counts.values,
        names=source_counts.index,
        title="Source Type Distribution"
    )
    
    # Relevance score distribution
    relevance_scores = [r.relevance_score for r in research_response.sources]
    fig2 = px.histogram(
        x=relevance_scores,
        nbins=10,
        title="Source Relevance Score Distribution"
    )
    
    # Timeline of sources (if timestamps vary)
    timestamps = [r.timestamp for r in research_response.sources]
    if len(set(timestamps)) > 1:
        fig3 = px.scatter(
            x=timestamps,
            y=relevance_scores,
            title="Source Timeline vs Relevance"
        )
    else:
        fig3 = None
    
    # Insights confidence levels
    if research_response.insights:
        insights_df = pd.DataFrame([
            {'category': i.category, 'confidence': i.confidence}
            for i in research_response.insights
        ])
        fig4 = px.bar(
            insights_df,
            x='category',
            y='confidence',
            title="Research Insights by Category and Confidence"
        )
    else:
        fig4 = None
    
    return fig1, fig2, fig3, fig4

def export_research_report(research_response: ResearchResponse, format_type: str = 'markdown'):
    """Export research report in various formats"""
    
    if format_type == 'markdown':
        report = f"""# Research Report: {research_response.query}

## Executive Summary
Generated on: {research_response.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Sources Found: {research_response.total_sources_found}
Unique Domains: {research_response.unique_domains}
Search Time: {research_response.search_time:.2f}s
Analysis Time: {research_response.analysis_time:.2f}s

## Synthesized Analysis
{research_response.synthesized_answer}

## Claude's Knowledge Base
{research_response.claude_knowledge}

## Combined Analysis
{research_response.combined_analysis}

## Key Insights
"""
        for insight in research_response.insights:
            report += f"\n### {insight.category} (Confidence: {insight.confidence:.2f})\n"
            report += f"{insight.insight}\n"
            if insight.supporting_sources:
                report += f"**Supporting Sources:** {', '.join(insight.supporting_sources[:3])}\n"
        
        report += "\n## Sources\n"
        for i, source in enumerate(research_response.sources, 1):
            report += f"{i}. **{source.title}** ({source.source_type})\n"
            report += f"   - URL: {source.url}\n"
            report += f"   - Domain: {source.domain}\n"
            report += f"   - Relevance: {source.relevance_score:.2f}\n"
            report += f"   - Snippet: {source.snippet[:200]}...\n\n"
        
        return report
    
    elif format_type == 'json':
        return json.dumps(asdict(research_response), default=str, indent=2)
    
    else:
        return "Unsupported format type"

def main():
    st.set_page_config(
        page_title="Simplified Deep Research Agent",
        page_icon="??",
        layout="wide"
    )
    
    st.title("?? Simplified Deep Research Agent")
    st.subheader("AI-powered comprehensive research with multi-source intelligence")
    
    # Authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_role = None
    
    if not st.session_state.authenticated:
        st.header("?? Authentication")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                role = RoleBasedAuth.authenticate(username, password)
                if role:
                    st.session_state.authenticated = True
                    st.session_state.user_role = role
                    st.success(f"Logged in as {role}")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        # Display demo credentials
        st.info("Demo credentials: admin_user/admin123, researcher_user/research123, basic_user/basic123")
        return
    
    # Get user permissions
    permissions = RoleBasedAuth.get_permissions(st.session_state.user_role)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("?? Configuration")
        st.write(f"**Role:** {st.session_state.user_role}")
        st.write(f"**Search Limit:** {permissions['search_limit']}")
        
        # API Configuration
        st.subheader("?? API Keys")
        serper_api_key = st.text_input("Serper API Key", type="password", help="Get from serper.dev")
        
        # AWS Configuration
        st.subheader("?? AWS Configuration")
        aws_region = st.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1"], index=0)
        
        # Model selection based on role
        available_models = {
            'anthropic.claude-3-5-sonnet-20241022-v2:0': 'Claude 3.5 Sonnet',
            'anthropic.claude-3-opus-20240229-v1:0': 'Claude 3 Opus',
            'anthropic.claude-3-haiku-20240307-v1:0': 'Claude 3 Haiku'
        }
        
        allowed_models = {k: v for k, v in available_models.items() 
                         if any(allowed in k for allowed in permissions['bedrock_models'])}
        
        selected_model = st.selectbox(
            "Claude Model",
            options=list(allowed_models.keys()),
            format_func=lambda x: allowed_models[x]
        )
        
        if st.button("?? Logout"):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.rerun()
    
    # Initialize clients
    if 'research_agent' not in st.session_state:
        bedrock_client = EnhancedBedrockClient(region_name=aws_region)
        searcher = SimplifiedInternetSearcher(serper_api_key=serper_api_key)
        st.session_state.research_agent = SimplifiedResearchAgent(bedrock_client, searcher)
    
    # Main interface
    st.header("?? Research Query")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_area(
            "Enter your research topic or complex question:",
            placeholder="e.g., 'Latest developments in quantum computing and their impact on cryptography'",
            height=100
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        research_button = st.button("?? Start Research", type="primary", use_container_width=True)
    
    if query and research_button:
        with st.spinner("?? Conducting comprehensive research..."):
            try:
                research_response = st.session_state.research_agent.conduct_advanced_research(
                    query, selected_model, permissions['search_limit']
                )
                
                # Store in session state
                st.session_state.last_research = research_response
                
                # Display results
                st.success(f"? Research completed in {research_response.search_time + research_response.analysis_time:.2f} seconds")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "?? Summary", 
                    "?? Full Report", 
                    "?? Sources", 
                    "?? Insights",
                    "?? Analytics"
                ])
                
                with tab1:
                    st.header("?? Research Summary")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Sources Found", research_response.total_sources_found)
                    with col2:
                        st.metric("Unique Domains", research_response.unique_domains)
                    with col3:
                        st.metric("Search Time", f"{research_response.search_time:.2f}s")
                    with col4:
                        st.metric("Analysis Time", f"{research_response.analysis_time:.2f}s")
                    
                    st.subheader("Combined Analysis")
                    st.write(research_response.combined_analysis)
                
                with tab2:
                    st.header("?? Full Research Report")
                    st.markdown(research_response.synthesized_answer)
                    
                    with st.expander("Claude's Knowledge Base Analysis"):
                        st.write(research_response.claude_knowledge)
                
                with tab3:
                    st.header("?? Sources")
                    
                    # Source filtering
                    source_types = list(set(s.source_type for s in research_response.sources))
                    selected_types = st.multiselect("Filter by source type:", source_types, default=source_types)
                    
                    filtered_sources = [s for s in research_response.sources if s.source_type in selected_types]
                    
                    for i, source in enumerate(filtered_sources, 1):
                        with st.expander(f"{i}. {source.title} ({source.source_type}) - Score: {source.relevance_score:.2f}"):
                            st.write(f"**URL:** {source.url}")
                            st.write(f"**Domain:** {source.domain}")
                            st.write(f"**Snippet:** {source.snippet}")
                            st.write(f"**Timestamp:** {source.timestamp}")
                
                with tab4:
                    st.header("?? Key Insights")
                    
                    if research_response.insights:
                        for insight in research_response.insights:
                            with st.expander(f"{insight.category} - Confidence: {insight.confidence:.2f}"):
                                st.write(insight.insight)
                                if insight.supporting_sources:
                                    st.write("**Supporting Sources:**")
                                    for source_url in insight.supporting_sources:
                                        st.write(f"- {source_url}")
                    else:
                        st.write("No specific insights were extracted from this research.")
                
                with tab5:
                    st.header("?? Research Analytics")
                    
                    if research_response.sources:
                        fig1, fig2, fig3, fig4 = create_visualizations(research_response)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig1, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        if fig3:
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        if fig4:
                            st.plotly_chart(fig4, use_container_width=True)
                    else:
                        st.write("No sources found for visualization.")
                
            except Exception as e:
                st.error(f"Research failed: {str(e)}")
                logger.error(f"Research error: {e}")
    
    # Export functionality
    if 'last_research' in st.session_state:
        st.header("?? Export Research")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            export_format = st.selectbox("Export Format", ["markdown", "json"])
        
        with col2:
            if st.button("?? Generate Export"):
                export_content = export_research_report(st.session_state.last_research, export_format)
                st.session_state.export_content = export_content
        
        with col3:
            if 'export_content' in st.session_state:
                filename = f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
                st.download_button(
                    label="?? Download Report",
                    data=st.session_state.export_content,
                    file_name=filename,
                    mime="text/markdown" if export_format == "markdown" else "application/json"
                )
    
    # Research history (if admin or researcher)
    if st.session_state.user_role in ['admin', 'researcher']:
        st.header("?? Research History")
        
        if 'research_history' not in st.session_state:
            st.session_state.research_history = []
        
        if 'last_research' in st.session_state:
            # Add to history if not already there
            if not any(r.query == st.session_state.last_research.query and 
                      r.timestamp == st.session_state.last_research.timestamp 
                      for r in st.session_state.research_history):
                st.session_state.research_history.append(st.session_state.last_research)
        
        if st.session_state.research_history:
            for i, research in enumerate(reversed(st.session_state.research_history[-10:])):  # Show last 10
                with st.expander(f"{research.query[:50]}... - {research.timestamp.strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"**Sources:** {research.total_sources_found}")
                    st.write(f"**Domains:** {research.unique_domains}")
                    st.write(f"**Time:** {research.search_time + research.analysis_time:.2f}s")
                    if st.button(f"View Full Report", key=f"view_{i}"):
                        st.session_state.last_research = research
                        st.rerun()
        else:
            st.write("No research history available.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>?? Simplified Deep Research Agent v2.0</p>
        <p>Powered by AWS Bedrock, Claude AI, and Multi-Source Intelligence</p>
        <p>Built with Streamlit | Role-based Access Control | Export Capabilities</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()