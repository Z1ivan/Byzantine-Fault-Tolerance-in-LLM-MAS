#!/usr/bin/env python3

import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_probe_test import SimpleProbe

logger = logging.getLogger(__name__)

class LCDConfidenceExtractor:

    def __init__(self, model_path: str):

        self.model_path = Path(model_path)
        self.probe = None
        self.pca = None
        self.scaler = None
        self.config = None
        self.load_model()

    def load_model(self):

        import joblib

        safety_path = self.model_path / 'safety_lcd_model.pkl'
        if self.model_path.suffix == '.pkl' and self.model_path.name.endswith('safety_lcd_model.pkl'):
            safety_path = self.model_path
        if safety_path.exists():
            model_data = joblib.load(safety_path)
            self.scaler = model_data.get('scaler')
            self.pca = model_data.get('pca')
            class _ProbeShim:
                def __init__(self, clf):
                    self.method = 'logistic'
                    self.classifier = clf
                    self.device = 'cpu'
            self.probe = _ProbeShim(model_data.get('model'))
            self.config = model_data.get('config', {})
            self._loaded_mode = 'safe_joblib'
            logger.info(f"LCD模型加载成功(SAFE joblib): {safety_path}")
            return

        simple_probe_ok = False
        try:
            self.probe, self.pca, self.scaler = SimpleProbe.load_model(self.model_path)
            model_data_path = self.model_path / 'lcd_model.pkl'
            if model_data_path.exists():
                model_data = joblib.load(model_data_path)
                self.config = model_data.get('config', {})
            logger.info(f"LCD模型加载成功(SimpleProbe): {self.model_path}")
            self._loaded_mode = 'simple_probe'
            simple_probe_ok = True
        except Exception as e:
            logger.error(f"未找到 LCD 模型: {self.model_path} (lcd_model.pkl 或 safety_lcd_model.pkl)，错误: {e}")
            raise

    def extract_confidence(self, hidden_states: np.ndarray, model_name: str = None) -> float:

        try:

            if hidden_states.ndim == 1:
                features = hidden_states.reshape(1, -1)
            else:

                if model_name == "llama31":
                    target_layer = 12                  
                elif model_name == "llama3":
                    target_layer = 16                
                else:

                    target_layer = self.config.get('layer', 16) if self.config else 16

                if hidden_states.ndim == 2 and hidden_states.shape[0] > target_layer:
                    features = hidden_states[target_layer].reshape(1, -1)
                else:
                    features = hidden_states.reshape(1, -1)

            loaded_mode = getattr(self, '_loaded_mode', 'simple_probe')
            if loaded_mode == 'safe_joblib':
                if self.scaler is not None:
                    features = self.scaler.transform(features)
                if self.pca is not None:
                    features = self.pca.transform(features)
            else:
                if self.pca is not None:
                    features = self.pca.transform(features)
                if self.scaler is not None:
                    features = self.scaler.transform(features)

            if self.probe.method == 'logistic':

                probs = self.probe.classifier.predict_proba(features)
                confidence = probs[0, 1]          
            else:

                self.probe.model.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).to(self.probe.device)
                    outputs = self.probe.model(features_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence = probs[0, 1].item()          

            return float(confidence)

        except Exception as e:
            logger.error(f"置信度提取失败: {e}")
            return 0.5         

    def extract_confidence_from_llm_output(self, llm_outputs: Dict[str, Any], 
                                         probe_type: str = "pooled", model_name: str = None) -> float:

        try:
            hidden_states = llm_outputs.get('hidden_states')
            if hidden_states is None:
                logger.warning("LLM输出中没有hidden_states")
                return 0.5

            if isinstance(hidden_states, torch.Tensor):
                hidden_states = hidden_states.cpu().numpy()

            if probe_type == "query":

                if hidden_states.ndim == 3:                                 
                    features = hidden_states[-1, -1, :]                  
                elif hidden_states.ndim == 2:                        
                    features = hidden_states[-1]        
                else:
                    features = hidden_states

            elif probe_type == "answer":

                if hidden_states.ndim == 3:                                 
                    features = hidden_states[-1, -1, :]                  
                elif hidden_states.ndim == 2:                        
                    features = hidden_states[-1]        
                else:
                    features = hidden_states

            elif probe_type == "pooled":

                if hidden_states.ndim == 3:                                 

                    features = np.mean(hidden_states[-1], axis=0)                  
                elif hidden_states.ndim == 2:                                  

                    if model_name == "llama31":
                        target_layer = 12                  
                    else:
                        target_layer = 16                    

                    if hidden_states.shape[0] > target_layer:
                        features = hidden_states[target_layer]          
                    else:

                        features = hidden_states[-1]
                        logger.warning(f"Hidden states层数不足，期望第{target_layer}层，但只有{hidden_states.shape[0]}层")
                else:
                    features = hidden_states
            else:
                logger.warning(f"未知的probe_type: {probe_type}")
                return 0.5

            return self.extract_confidence(features, model_name)

        except Exception as e:
            logger.error(f"从LLM输出提取置信度失败: {e}")
            return 0.5

    def get_model_info(self) -> Dict[str, Any]:

        return {
            'model_path': str(self.model_path),
            'method': self.probe.method if self.probe else None,
            'config': self.config,
            'loaded': self.probe is not None
        }

def load_best_lcd_model(models_dir: str = "../models") -> LCDConfidenceExtractor:

    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"模型目录不存在: {models_dir}")

    model_dirs = [d for d in models_path.iterdir() if d.is_dir()]

    if not model_dirs:
        raise FileNotFoundError(f"在 {models_dir} 中没有找到任何模型")

    best_model_path = model_dirs[0]
    logger.info(f"选择模型: {best_model_path}")

    return LCDConfidenceExtractor(str(best_model_path))

if __name__ == "__main__":

    try:
        extractor = load_best_lcd_model()
        print(f"模型信息: {extractor.get_model_info()}")

        dummy_features = np.random.randn(256)               
        confidence = extractor.extract_confidence(dummy_features)
        print(f"测试置信度: {confidence:.4f}")

    except Exception as e:
        print(f"测试失败: {e}") 