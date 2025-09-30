"""
Perplexity API client for LLM interactions (replacing OpenRouter).
"""
import os
import requests
import logging
from typing import Dict, Any, Optional, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class PerplexityClient:
    """Client for interacting with Perplexity API (OpenRouter replacement)."""
    
    def __init__(self):
        """Initialize Perplexity client."""
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.base_url = "https://api.perplexity.ai"
        self.default_model = os.getenv("DEFAULT_MODEL", "sonar-pro")
        
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable is required")
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,  # Allow 3 retries
            backoff_factor=1,  # Exponential backoff: 1s, 2s, 4s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],  # Only retry POST requests
            raise_on_status=False,  # Don't raise on retry-able status codes
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to Perplexity API with error handling.
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response data
            
        Raises:
            RuntimeError: If API request fails
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.post(url, json=data, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 429:
                logger.warning("Perplexity API rate limit exceeded")
                raise RuntimeError("Rate limit exceeded. Please try again later.")
            
            # Handle other HTTP errors
            if response.status_code >= 400:
                logger.error(f"Perplexity API error {response.status_code}: {response.text}")
                raise RuntimeError(f"API request failed with status {response.status_code}")
            
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error("Perplexity API request timed out")
            raise RuntimeError("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Perplexity API")
            raise RuntimeError("Failed to connect to API. Please check your internet connection.")
        except Exception as e:
            logger.error(f"Unexpected error in Perplexity API request: {e}")
            raise RuntimeError(f"API request failed: {str(e)}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Get chat completion from Perplexity API.
        
        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to configured model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            API response data
        """
        model = model or self.default_model
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        return self._make_request("chat/completions", data)
    
    def chat_completion_with_tools(self, messages: List[Dict[str, str]], tools: List[Any] = None) -> Dict[str, Any]:
        """
        Get chat completion with tool calling support.
        Note: Perplexity doesn't support tools, so we fall back to regular chat completion.
        
        Args:
            messages: List of message dictionaries
            tools: List of tools (ignored for Perplexity)
            
        Returns:
            API response data
        """
        # Perplexity doesn't support tool calling, so we use regular chat completion
        logger.info("Perplexity doesn't support tool calling, using regular chat completion")
        return self.chat_completion(messages)
    
    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Any:
        """
        Stream chat completion from Perplexity API.
        
        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to configured model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Streaming response
        """
        model = model or self.default_model
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        url = f"{self.base_url}/chat/completions"
        
        try:
            response = self.session.post(url, json=data, stream=True, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {e}")
            raise RuntimeError(f"Streaming request failed: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """
        Get available models from Perplexity.
        
        Returns:
            List of available model names
        """
        # Perplexity doesn't have a models endpoint, return common models
        return [
            "sonar",
            "sonar-pro"
        ]
    
    def check_api_health(self) -> bool:
        """
        Check if the Perplexity API is accessible.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            self.chat_completion(test_messages, max_tokens=10)
            return True
        except Exception as e:
            logger.error(f"Perplexity API health check failed: {e}")
            return False
    
    def search_recipes(self, query: str, cuisine: str = None) -> str:
        """Search for recipes using Perplexity API (legacy method)."""
        try:
            search_query = f"Find detailed recipes for {query}"
            if cuisine:
                search_query += f" in {cuisine} cuisine"
            search_query += ". Include ingredients, preparation time, and nutritional information."
            
            messages = [{"role": "user", "content": search_query}]
            response = self.chat_completion(messages, max_tokens=1000)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error searching recipes: {str(e)}"