import os
from typing import Optional, Dict, List
import json
import requests
import pandas as pd

from mindsdb.utilities import log
from mindsdb.integrations.libs.base import BaseMLEngine
from mindsdb.integrations.utilities.handler_utils import get_api_key

logger = log.getLogger(__name__)

class DeepSeekHandler(BaseMLEngine):
    """
    Handler for DeepSeek LLM API
    """
    name = 'deepseek'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generative = True
        self.default_model = "deepseek-chat"  # DeepSeek default model
        self.supported_models = [
            "deepseek-chat",
            "deepseek-coder",
            "deepseek-math"
        ]
        self.default_mode = 'default'
        self.supported_modes = ['default', 'conversational']
        self.rate_limit = 60  # requests per minute
        self.max_batch_size = 20
        self.default_max_tokens = 2048
        self.api_base = None
        self._session = requests.Session()

    def create_engine(self, connection_args: Dict, integration_id: int = None) -> None:
        """
        Set up DeepSeek API credentials
        Args:
            connection_args (Dict): Dictionary containing API credentials
            integration_id (int): Integration ID
        """
        connection_args = {k.lower(): v for k, v in connection_args.items()}
        api_key = connection_args.get('deepseek_api_key') or connection_args.get('api_key')
        if api_key is None:
            api_key = get_api_key('deepseek_api_key', connection_args, self.engine_storage)
            
        if api_key is None:
            raise Exception("API key is required for DeepSeek integration")
            
        self.api_base = connection_args.get('api_base', 'https://api.deepseek.com/v1')
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
            
        # Save API key and base URL to engine storage
        self.engine_storage.json_set('connection_args', {
            'api_key': api_key,
            'api_base': self.api_base
        })

    def create(self, target: str, df: Optional[pd.DataFrame] = None, args: Optional[Dict] = None) -> None:
        """
        Create and save model configuration
        Args:
            target (str): Target column name
            df (pd.DataFrame, optional): Training data
            args (Dict, optional): Additional parameters
        """
        args = args or {}
        if 'using' in args:
            args = args['using']
        args['target'] = target
        
        if not args.get('mode'):
            args['mode'] = self.default_mode
        elif args['mode'] not in self.supported_modes:
            raise Exception(f"Invalid operation mode. Please use one of {self.supported_modes}")
            
        if not args.get('model_name'):
            args['model_name'] = self.default_model
        elif args['model_name'] not in self.supported_models:
            raise Exception(f"Model {args['model_name']} is not supported. Supported models: {self.supported_models}")
            
        args['max_tokens'] = args.get('max_tokens', self.default_max_tokens)
        args['temperature'] = args.get('temperature', 0.7)
        args['top_p'] = args.get('top_p', 0.7)
        args['presence_penalty'] = args.get('presence_penalty', 0.0)
        args['frequency_penalty'] = args.get('frequency_penalty', 0.0)
        
        self.model_storage.json_set('args', args)

    def predict(self, df: pd.DataFrame, args: Optional[Dict] = None) -> pd.DataFrame:
        """
        Use DeepSeek model for predictions
        Args:
            df (pd.DataFrame): Input data
            args (Dict, optional): Prediction parameters
        Returns:
            pd.DataFrame: DataFrame containing prediction results
        """
        args = args or {}
        pred_args = args.get('predict_params', {})
        stored_args = self.model_storage.json_get('args')
        
        # Get API key and base URL
        connection_args = self.engine_storage.json_get('connection_args')
        if connection_args is None:
            raise Exception("Missing API key. Either re-create this ML_ENGINE specifying the 'deepseek_api_key' parameter, or re-create this model and pass the API key with `USING` syntax.")
            
        api_key = connection_args.get('api_key')
        if api_key is None:
            raise Exception("Missing API key. Either re-create this ML_ENGINE specifying the 'deepseek_api_key' parameter, or re-create this model and pass the API key with `USING` syntax.")
            
        self.api_base = connection_args.get('api_base', 'https://api.deepseek.com/v1')
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        
        model_name = pred_args.get('model_name', stored_args.get('model_name', self.default_model))
        mode = pred_args.get('mode', stored_args.get('mode', self.default_mode))
        max_tokens = pred_args.get('max_tokens', stored_args.get('max_tokens', self.default_max_tokens))
        temperature = pred_args.get('temperature', stored_args.get('temperature', 0.7))
        top_p = pred_args.get('top_p', stored_args.get('top_p', 0.7))
        presence_penalty = pred_args.get('presence_penalty', stored_args.get('presence_penalty', 0.0))
        frequency_penalty = pred_args.get('frequency_penalty', stored_args.get('frequency_penalty', 0.0))
        
        target = stored_args.get('target')
        if not target:
            raise Exception("Target column not found in model storage")
            
        results = []
        
        for _, row in df.iterrows():
            # Use text column as input
            prompt = str(row.get('text', ''))
            
            if mode == 'conversational':
                # Handle conversation history
                history = pred_args.get('history', [])
                messages = []
                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": prompt})
            else:
                # Default mode
                messages = [{"role": "user", "content": prompt}]
            
            try:
                response = self._session.post(
                    f"{self.api_base}/chat/completions",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "presence_penalty": presence_penalty,
                        "frequency_penalty": frequency_penalty
                    }
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    result = response_data['choices'][0]['message']['content']
                else:
                    response_data = response.json()
                    if 'error' in response_data and response_data['error'].get('message') == 'Insufficient Balance':
                        result = "Error: DeepSeek API account has insufficient balance. Please visit https://platform.deepseek.com/ to check your account balance or get a new API key."
                    else:
                        result = f"Error: {response.text}"
            except Exception as e:
                result = f"Error: {str(e)}"
            
            results.append(result)
        
        # Create output DataFrame
        output_df = pd.DataFrame()
        output_df[target] = results  # Use target column name as output column name
        return output_df

    def describe(self, attribute: Optional[str] = None) -> pd.DataFrame:
        """
        Return model metadata information
        Args:
            attribute: Optional, specify which attribute to return
        Returns:
            pd.DataFrame: DataFrame containing model metadata
        """
        if attribute == "args":
            args = self.model_storage.json_get('args')
            if args is None:
                args = {}
            return pd.DataFrame([args])
            
        if attribute == "metadata":
            metadata = {
                'name': 'DeepSeek',
                'version': '1.0',
                'type': 'language model',
                'description': 'DeepSeek Language Model',
                'model_name': self.model_storage.json_get('args', {}).get('model_name', self.default_model),
                'mode': self.model_storage.json_get('args', {}).get('mode', self.default_mode),
                'supported_modes': self.supported_modes,
                'supported_models': self.supported_models,
                'api_base': self.api_base
            }
            return pd.DataFrame([metadata])
            
        tables = ['args', 'metadata']
        return pd.DataFrame(tables, columns=['tables'])

    def _get_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Get text embeddings
        Args:
            texts (List[str]): List of texts to embed
            model (str, optional): Embedding model name
        Returns:
            List[List[float]]: List of embedding vectors
        """
        model = model or "deepseek-embedding"
        
        embeddings = []
        for text in texts:
            try:
                response = self._session.post(
                    f"{self.api_base}/embeddings",
                    json={
                        "model": model,
                        "input": text
                    }
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    embedding = response_data['data'][0]['embedding']
                    embeddings.append(embedding)
                else:
                    raise Exception(f"Failed to get embedding: {response.text}")
            except Exception as e:
                logger.error(f"Error getting embedding: {str(e)}")
                embeddings.append([0.0] * 1024)  # Return zero vector as fallback
                
        return embeddings
