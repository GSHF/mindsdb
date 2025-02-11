# -*- coding: utf-8 -*-
from mindsdb.integrations.libs.const import HANDLER_TYPE

from .__about__ import __version__ as version, __description__ as description
from .creation_args import creation_args

try:
    from .qianwen_handler import QianwenHandler as Handler
    import_error = None
except Exception as e:
    Handler = None
    import_error = e

title = 'Qianwen'
name = 'qianwen'
type = HANDLER_TYPE.ML
icon_path = 'icon.svg'
permanent = False

__all__ = [
    'Handler', 'version', 'name', 'type', 'title', 'description',
    'import_error', 'icon_path', 'creation_args'
]
