from collections import OrderedDict
from mindsdb.integrations.libs.const import HANDLER_CONNECTION_ARG_TYPE as ARG_TYPE

creation_args = OrderedDict(
    api_key={
        'type': ARG_TYPE.STR,
        'description': 'API key for DeepSeek',
        'required': True,
        'label': 'DeepSeek API key',
        'secret': True
    },
    api_base={
        'type': ARG_TYPE.STR,
        'description': 'API base URL for DeepSeek (optional)',
        'required': False,
        'label': 'DeepSeek API base URL',
        'default': 'https://api.deepseek.com/v1'
    }
)
