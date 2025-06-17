import os
from typing import Dict, Any
TOGETHER_CONFIG = {
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "max_tokens": 100,
    "temperature": 0.7
}

UPLOAD_CONFIG = {
    "max_file_size": 200, 
    "allowed_extensions": ['.txt', '.csv', '.pdf', '.png', '.jpg', '.jpeg'],
    "chunk_size": 1024 * 1024 
}
VIZ_CONFIG = {
    "default_theme": "plotly_white",
    "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "figure_size": (10, 6)
}
def get_env_var(key: str, default: str = None) -> str:
    return os.getenv(key, default)
def validate_config() -> Dict[str, Any]:
  
    status = {
        "together_api_key": bool(get_env_var("TOGETHER_API_KEY")),
        "config_loaded": True
    }
    return status
