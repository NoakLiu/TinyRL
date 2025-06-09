"""
Web Agent - External Information Retrieval for MCP Agent Sandbox Framework
Implements web search and browsing capabilities with minimal predefinition
"""

import asyncio
import requests
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from urllib.parse import urlparse, quote
import json
import re
from bs4 import BeautifulSoup

from models.llm_interface import MultiModelManager

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result"""
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0

@dataclass
class WebPageContent:
    """Represents web page content"""
    url: str
    title: str
    content: str
    links: List[str]
    metadata: Dict[str, Any]

class SimpleTextBrowser:
    """Simple text-based web browser interface"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.current_url = None
        self.current_content = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Alita-Framework/1.0 (Text Browser)'
        })
    
    def visit(self, url: str) -> WebPageContent:
        """Visit a webpage and extract content"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "No Title"
            
            # Extract main content
            content = self._extract_text_content(soup)
            
            # Extract links
            links = [link.get('href') for link in soup.find_all('a', href=True)]
            links = [self._resolve_url(link, url) for link in links if link]
            
            # Metadata
            metadata = {
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(content)
            }
            
            self.current_url = url
            self.current_content = WebPageContent(
                url=url,
                title=title,
                content=content,
                links=links,
                metadata=metadata
            )
            
            return self.current_content
            
        except Exception as e:
            logger.error(f"Failed to visit {url}: {e}")
            raise
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract readable text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _resolve_url(self, url: str, base_url: str) -> str:
        """Resolve relative URLs"""
        if url.startswith('http'):
            return url
        elif url.startswith('//'):
            return f"https:{url}"
        elif url.startswith('/'):
            parsed = urlparse(base_url)
            return f"{parsed.scheme}://{parsed.netloc}{url}"
        else:
            return f"{base_url.rstrip('/')}/{url}"
    
    def page_up(self, lines: int = 20) -> str:
        """Simulate page up navigation"""
        if not self.current_content:
            return "No page loaded"
        
        lines_list = self.current_content.content.split('\n')
        start_idx = max(0, len(lines_list) - lines * 2)
        end_idx = max(lines, len(lines_list) - lines)
        
        return '\n'.join(lines_list[start_idx:end_idx])
    
    def page_down(self, lines: int = 20) -> str:
        """Simulate page down navigation"""
        if not self.current_content:
            return "No page loaded"
        
        lines_list = self.current_content.content.split('\n')
        start_idx = min(lines, len(lines_list))
        end_idx = min(lines * 2, len(lines_list))
        
        return '\n'.join(lines_list[start_idx:end_idx])

class GoogleSearchTool:
    """Google search interface"""
    
    def __init__(self, api_key: str = None, search_engine_id: str = None):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Perform Google search"""
        if not self.api_key or not self.search_engine_id:
            # Fallback to simulated search
            return await self._fallback_search(query, num_results)
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10)
            }
            
            response = await asyncio.to_thread(
                requests.get, self.base_url, params=params
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    relevance_score=1.0
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return await self._fallback_search(query, num_results)
    
    async def _fallback_search(self, query: str, num_results: int) -> List[SearchResult]:
        """Fallback search using DuckDuckGo or other free services"""
        try:
            # Simple DuckDuckGo instant answer API
            encoded_query = quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1"
            
            response = await asyncio.to_thread(requests.get, url)
            data = response.json()
            
            results = []
            
            # Add abstract if available
            if data.get('Abstract'):
                results.append(SearchResult(
                    title=data.get('AbstractSource', 'DuckDuckGo'),
                    url=data.get('AbstractURL', ''),
                    snippet=data.get('Abstract', ''),
                    relevance_score=0.9
                ))
            
            # Add related topics
            for topic in data.get('RelatedTopics', [])[:num_results-1]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append(SearchResult(
                        title=topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                        url=topic.get('FirstURL', ''),
                        snippet=topic.get('Text', ''),
                        relevance_score=0.7
                    ))
            
            return results[:num_results]
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

class GithubSearchTool:
    """GitHub search interface"""
    
    def __init__(self, token: str = None):
        self.token = token
        self.base_url = "https://api.github.com/search"
        self.session = requests.Session()
        if token:
            self.session.headers.update({'Authorization': f'token {token}'})
    
    async def search_repositories(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search GitHub repositories"""
        try:
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': min(num_results, 30)
            }
            
            response = await asyncio.to_thread(
                self.session.get, f"{self.base_url}/repositories", params=params
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for repo in data.get('items', []):
                result = SearchResult(
                    title=repo.get('full_name', ''),
                    url=repo.get('html_url', ''),
                    snippet=repo.get('description', '') or 'No description',
                    relevance_score=repo.get('stargazers_count', 0) / 1000.0
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []
    
    async def get_repository_content(self, repo_url: str, path: str = "") -> Dict[str, Any]:
        """Get repository content (README, code files, etc.)"""
        try:
            # Extract owner and repo from URL
            parsed = urlparse(repo_url)
            path_parts = parsed.path.strip('/').split('/')
            
            if len(path_parts) < 2:
                raise ValueError("Invalid repository URL")
            
            owner, repo = path_parts[0], path_parts[1]
            
            # Get repository contents
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
            
            response = await asyncio.to_thread(self.session.get, api_url)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, list):
                # Directory listing
                return {
                    'type': 'directory',
                    'contents': [
                        {
                            'name': item['name'],
                            'type': item['type'],
                            'download_url': item.get('download_url')
                        }
                        for item in data
                    ]
                }
            else:
                # Single file
                if data.get('download_url'):
                    content_response = await asyncio.to_thread(
                        requests.get, data['download_url']
                    )
                    content = content_response.text
                else:
                    # Base64 encoded content
                    import base64
                    content = base64.b64decode(data['content']).decode('utf-8')
                
                return {
                    'type': 'file',
                    'name': data['name'],
                    'content': content,
                    'size': data['size']
                }
                
        except Exception as e:
            logger.error(f"Failed to get repository content: {e}")
            return {}

class WebAgent:
    """Web agent for external information retrieval"""
    
    def __init__(self, model_manager: MultiModelManager):
        self.model_manager = model_manager
        self.browser = SimpleTextBrowser()
        self.google_search = GoogleSearchTool()
        self.github_search = GithubSearchTool()
        
        # Navigation state
        self.current_page = None
        self.search_history = []
    
    async def search(self, query: str, search_type: str = "general") -> List[SearchResult]:
        """Perform web search based on type"""
        
        self.search_history.append({
            'query': query,
            'type': search_type,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        if search_type == "github" or "github" in query.lower():
            results = await self.github_search.search_repositories(query)
        else:
            results = await self.google_search.search(query)
        
        logger.info(f"Found {len(results)} results for: {query}")
        return results
    
    async def visit_page(self, url: str) -> WebPageContent:
        """Visit a webpage and return content"""
        try:
            content = self.browser.visit(url)
            self.current_page = content
            return content
        except Exception as e:
            logger.error(f"Failed to visit {url}: {e}")
            raise
    
    async def extract_relevant_content(self, url: str, context: str) -> str:
        """Visit page and extract content relevant to context"""
        try:
            content = await self.visit_page(url)
            
            # Use LLM to extract relevant information
            extraction_prompt = f"""
            Extract information relevant to the following context from the webpage content:
            
            Context: {context}
            
            Webpage Title: {content.title}
            Webpage Content: {content.content[:3000]}...
            
            Please extract and summarize the most relevant information for the given context.
            """
            
            relevant_info = await self.model_manager.generate(
                extraction_prompt,
                system_prompt="You are an expert information extractor. Focus on relevance and accuracy."
            )
            
            return relevant_info
            
        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {e}")
            return f"Error extracting content: {str(e)}"
    
    async def intelligent_search(self, task_description: str) -> Dict[str, Any]:
        """Perform intelligent search based on task requirements"""
        
        # Use LLM to determine search strategy
        strategy_prompt = f"""
        Given the following task, determine the best search strategy:
        
        Task: {task_description}
        
        Provide:
        1. Search queries to use
        2. Types of sources to look for (documentation, code examples, tutorials, etc.)
        3. Specific platforms to search (GitHub, Stack Overflow, official docs, etc.)
        
        Format as JSON with keys: queries, source_types, platforms
        """
        
        strategy_response = await self.model_manager.generate(
            strategy_prompt,
            system_prompt="You are a search strategist. Optimize for finding the most useful information."
        )
        
        try:
            strategy = json.loads(strategy_response)
        except json.JSONDecodeError:
            # Fallback strategy
            strategy = {
                "queries": [task_description],
                "source_types": ["documentation", "examples"],
                "platforms": ["general", "github"]
            }
        
        # Execute search strategy
        all_results = []
        
        for query in strategy.get("queries", [task_description]):
            for platform in strategy.get("platforms", ["general"]):
                results = await self.search(query, platform)
                all_results.extend(results)
        
        # Rank and filter results
        ranked_results = await self._rank_results(all_results, task_description)
        
        return {
            "strategy": strategy,
            "total_results": len(all_results),
            "ranked_results": ranked_results[:10],  # Top 10
            "search_queries": strategy.get("queries", [])
        }
    
    async def _rank_results(self, results: List[SearchResult], context: str) -> List[SearchResult]:
        """Rank search results by relevance to context"""
        
        if not results:
            return []
        
        # Use LLM to score relevance
        ranking_prompt = f"""
        Rank the following search results by relevance to the context: {context}
        
        Results:
        """
        
        for i, result in enumerate(results[:20]):  # Limit for prompt size
            ranking_prompt += f"\n{i+1}. {result.title}\n   {result.snippet}\n   URL: {result.url}\n"
        
        ranking_prompt += """
        
        Provide a ranked list of result numbers (1-indexed) from most to least relevant.
        Format as a JSON array of numbers: [3, 1, 7, ...]
        """
        
        try:
            ranking_response = await self.model_manager.generate(
                ranking_prompt,
                system_prompt="You are a relevance evaluator. Rank results by their usefulness for the given context."
            )
            
            ranking = json.loads(ranking_response)
            
            # Reorder results based on ranking
            ranked_results = []
            for rank in ranking:
                if 1 <= rank <= len(results):
                    ranked_results.append(results[rank - 1])
            
            # Add any remaining results
            ranked_indices = [r - 1 for r in ranking if 1 <= r <= len(results)]
            for i, result in enumerate(results):
                if i not in ranked_indices:
                    ranked_results.append(result)
            
            return ranked_results
            
        except (json.JSONDecodeError, IndexError):
            # Fallback to original order with relevance score
            return sorted(results, key=lambda r: r.relevance_score, reverse=True)
    
    def get_navigation_tools(self) -> Dict[str, callable]:
        """Get navigation tools for the browser"""
        return {
            "visit": self.visit_page,
            "page_up": self.browser.page_up,
            "page_down": self.browser.page_down,
            "search": self.search,
            "extract_content": self.extract_relevant_content
        }
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get search history"""
        return self.search_history
    
    def clear_search_history(self):
        """Clear search history"""
        self.search_history.clear() 