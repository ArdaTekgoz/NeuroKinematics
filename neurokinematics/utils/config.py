import yaml
import os
from pathlib import Path

class ConfigManager:
    """Merkezi Konfigürasyon Yöneticisi"""
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        keys = key.split('.')
        val = self.config
        try:
            for k in keys:
                val = val[k]
            return val
        except KeyError:
            return default

# Singleton instance
config_manager = None

def get_config(config_path="config.yaml"):
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager(config_path)
    return config_manager
