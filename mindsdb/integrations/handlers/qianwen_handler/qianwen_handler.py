import os
from typing import Optional, Dict, List
import dashscope
from dashscope import Generation
import pandas as pd

from mindsdb.utilities import log
from mindsdb.integrations.libs.base import BaseMLEngine
from mindsdb.integrations.utilities.handler_utils import get_api_key

logger = log.getLogger(__name__)

class QianwenHandler(BaseMLEngine):
    """
    Handler for Qianwen (通义千问) LLM API
    """
    name = 'qianwen'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import dashscope
        except ImportError:
            raise Exception(
                "dashscope is not installed. Please install it with: pip install dashscope>=1.13.6"
            )
        self.generative = True
        self.default_model = "qwen-turbo"
        self.supported_models = ["qwen-turbo", "qwen-plus", "qwen-max"]
        self.default_mode = 'default'
        self.supported_modes = ['default', 'conversational']
        self.rate_limit = 60
        self.max_batch_size = 20
        self.default_max_tokens = 1024

    @staticmethod
    def _validate_connection(api_key: str):
        """验证API连接"""
        try:
            response = Generation.call(
                model="qwen-turbo",
                prompt='测试连接',
                max_tokens=10,
                api_key=api_key
            )
            if response.status_code != 200:
                raise Exception(f"Connection test failed: {response.message}")
        except Exception as e:
            raise Exception(f"Failed to connect to Qianwen API: {str(e)}")

    def create_engine(self, connection_args: Dict, integration_id: int = None) -> None:
        """
        设置通义千问 API 凭证
        Args:
            connection_args (Dict): 包含API凭证的参数字典
            integration_id (int): 集成ID
        """
        connection_args = {k.lower(): v for k, v in connection_args.items()}
        api_key = connection_args.get('qianwen_api_key') or connection_args.get('api_key')
            
        if api_key is None:
            raise Exception("API key is required for Qianwen integration")
            
        self._validate_connection(api_key)
        dashscope.api_key = api_key
        
        # 保存 API key 到 engine storage
        self.engine_storage.json_set('connection_args', {'api_key': api_key})

    def create(self, target: str, df: Optional[pd.DataFrame] = None, args: Optional[Dict] = None) -> None:
        """
        创建并保存模型配置
        Args:
            target (str): 目标列名
            df (pd.DataFrame, optional): 训练数据
            args (Dict, optional): 其他参数
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
            
        self.model_storage.json_set('args', args)

    def predict(self, df: pd.DataFrame, args: Optional[Dict] = None) -> pd.DataFrame:
        """
        使用通义千问模型进行预测
        Args:
            df (pd.DataFrame): 输入数据
            args (Dict, optional): 预测参数
        Returns:
            pd.DataFrame: 包含预测结果的数据框
        """
        args = args or {}
        pred_args = args.get('predict_params', {})
        stored_args = self.model_storage.json_get('args')
        
        # 获取 API key
        connection_args = self.engine_storage.json_get('connection_args')
        if connection_args is None:
            raise Exception("Missing API key. Either re-create this ML_ENGINE specifying the 'qianwen_api_key' parameter, or re-create this model and pass the API key with `USING` syntax.")
            
        api_key = connection_args.get('api_key')
        if api_key is None:
            raise Exception("Missing API key. Either re-create this ML_ENGINE specifying the 'qianwen_api_key' parameter, or re-create this model and pass the API key with `USING` syntax.")
            
        dashscope.api_key = api_key
        
        model_name = pred_args.get('model_name', stored_args.get('model_name', self.default_model))
        mode = pred_args.get('mode', stored_args.get('mode', self.default_mode))
        max_tokens = pred_args.get('max_tokens', stored_args.get('max_tokens', self.default_max_tokens))
        temperature = pred_args.get('temperature', stored_args.get('temperature', 0.7))
        top_p = pred_args.get('top_p', stored_args.get('top_p', 0.7))
        
        target = stored_args.get('target')
        if not target:
            raise Exception("Target column not found in model storage")
            
        results = []
        
        for _, row in df.iterrows():
            # 使用 text 列作为输入
            prompt = str(row.get('text', ''))
            
            if mode == 'conversational':
                # 处理对话历史
                history = pred_args.get('history', [])
                messages = []
                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": prompt})
                
                response = Generation.call(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    result_format='message'  # 返回消息格式
                )
                
                if response.status_code == 200:
                    result = response.output.choices[0]['message']['content']
                else:
                    result = f"Error: {response.message}"
            else:
                # 默认模式
                response = Generation.call(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                if response.status_code == 200:
                    result = response.output.text
                else:
                    result = f"Error: {response.message}"
            
            results.append(result)
        
        # 创建输出数据框
        output_df = pd.DataFrame()
        output_df[target] = results  # 使用目标列名作为输出列名
        return output_df

    def describe(self, attribute: Optional[str] = None) -> pd.DataFrame:
        """
        返回模型的元数据信息
        Args:
            attribute: 可选，指定要返回的属性
        Returns:
            pd.DataFrame: 包含模型元数据的数据框
        """
        if attribute == "args":
            args = self.model_storage.json_get('args')
            if args is None:
                args = {}
            return pd.DataFrame([args])
            
        if attribute == "metadata":
            metadata = {
                'name': 'Qianwen',
                'version': '1.0',
                'type': 'language model',
                'description': '通义千问大语言模型',
                'model_name': self.model_storage.json_get('args', {}).get('model_name', self.default_model),
                'mode': self.model_storage.json_get('args', {}).get('mode', self.default_mode),
                'supported_modes': self.supported_modes,
                'supported_models': self.supported_models
            }
            return pd.DataFrame([metadata])
            
        tables = ['args', 'metadata']
        return pd.DataFrame(tables, columns=['tables'])
