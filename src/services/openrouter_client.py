"""
OpenRouter API client for LLM interactions.
"""
import os
import json
import time
import logging
from typing import Optional, Dict, Any, Iterator, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self):
        """Initialize OpenRouter client."""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_model = os.getenv("DEFAULT_MODEL", "anthropic/claude-3-haiku")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
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
            "Content-Type": "application/json",
            "HTTP-Referer": "https://meal-ai-agent.com",
            "X-Title": "Meal AI Agent"
        })
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to OpenRouter API with error handling.
        
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
            
            # Let the retry mechanism handle 429 errors by raising an exception
            if response.status_code == 429:
                logger.warning("Rate limit exceeded, retry mechanism will handle this")
                response.raise_for_status()  # This will trigger the retry mechanism
            
            # If we get here, the request was successful
            response.raise_for_status()
            return response.json()
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limit exceeded after all retries")
                return {
                    "choices": [{
                        "message": {
                            "content": "I'm currently experiencing high demand. Please try again in a moment."
                        }
                    }]
                }
            else:
                logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                raise RuntimeError(f"API request failed: {e.response.status_code}")
        except requests.exceptions.Timeout:
            logger.error("OpenRouter API request timed out")
            raise RuntimeError("API request timed out")
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to OpenRouter API")
            raise RuntimeError("Failed to connect to API")
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            raise RuntimeError(f"API request failed: {e}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Get chat completion from OpenRouter API.
        
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

    # Add this method after the existing chat_completion method (around line 100)
    def chat_completion_with_tools(self, messages: List[Dict[str, str]], tools: List[Any] = None) -> Dict[str, Any]:
        """Get chat completion with tool calling support."""
        data = {
            "model": self.default_model,
            "messages": messages,
            "temperature": 0.7
        }
        
        if tools:
            tool_definitions = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_schema.schema()
                }
            } for tool in tools]
            data["tools"] = tool_definitions
            data["tool_choice"] = "auto"
        
        return self._make_request("chat/completions", data)
    
    
    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """
        Stream chat completion from OpenRouter API.
        
        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to configured model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Chunks of the streaming response
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
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {e}")
            raise RuntimeError(f"Streaming failed: {e}")
    
    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get available models from OpenRouter.
        
        Returns:
            List of available models
        """
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information or None if not found
        """
        models = self.get_models()
        for model in models:
            if model.get('id') == model_id:
                return model
        return None
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate the cost of a request.
        
        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        model_info = self.get_model_info(model)
        if not model_info:
            return 0.0
        
        pricing = model_info.get('pricing', {})
        input_cost = pricing.get('prompt', 0) * input_tokens / 1000000
        output_cost = pricing.get('completion', 0) * output_tokens / 1000000
        
        return input_cost + output_cost
    
    def health_check(self) -> bool:
        """
        Check if the OpenRouter API is accessible.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            models = self.get_models()
            return len(models) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
