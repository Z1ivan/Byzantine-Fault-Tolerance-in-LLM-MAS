
from .base_model import BaseModel, NetworkClient
from .api_model import APIModel
from .local_model import LocalModel

__all__ = [
    'BaseModel',
    'NetworkClient', 
    'APIModel',
    'LocalModel',
    'create_model'
]

def create_model(model_type: str = 'api', **kwargs):

    if model_type.lower() == 'api':
        return APIModel(**kwargs)
    elif model_type.lower() == 'local':
        return LocalModel(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}。支持的类型: 'api', 'local'")

__version__ = "2.0.0" 