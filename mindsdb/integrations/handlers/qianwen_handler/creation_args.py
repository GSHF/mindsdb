from collections import OrderedDict
from mindsdb.integrations.libs.const import HANDLER_CONNECTION_ARG_TYPE as ARG_TYPE

creation_args = OrderedDict(
    api_key={
        'type': ARG_TYPE.STR,
        'description': 'API key for Qianwen (通义千问)',
        'required': True,
        'label': 'Qianwen API key',
        'secret': True
    }
)
