#!/usr/bin/env python3

import random
import numpy as np
import time
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SeedManager:

    def __init__(self, master_seed: Optional[int] = None):

        self.master_seed = master_seed or int(time.time() * 1000) % 2**32
        self.component_seeds = {}
        self.is_reproducible = master_seed is not None

        random.seed(self.master_seed)
        np.random.seed(self.master_seed)

        logger.info(f"种子管理器初始化: master_seed={self.master_seed}, reproducible={self.is_reproducible}")

    def get_component_seed(self, component_name: str) -> int:

        if component_name not in self.component_seeds:

            import hashlib
            seed_string = f"{self.master_seed}_{component_name}"
            stable_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
            self.component_seeds[component_name] = stable_hash
            logger.debug(f"为组件 '{component_name}' 生成稳定种子: {self.component_seeds[component_name]}")

        return self.component_seeds[component_name]

    def set_component_seed(self, component_name: str, temp_random: bool = False):

        if temp_random and not self.is_reproducible:

            seed = int(time.time() * 1000) % 2**32
            random.seed(seed)
            logger.debug(f"为组件 '{component_name}' 设置临时随机种子: {seed}")
        else:

            seed = self.get_component_seed(component_name)
            random.seed(seed)
            logger.debug(f"为组件 '{component_name}' 设置确定性种子: {seed}")

    def get_summary(self) -> Dict[str, Any]:

        return {
            "master_seed": self.master_seed,
            "is_reproducible": self.is_reproducible,
            "component_seeds": self.component_seeds.copy(),
            "num_components": len(self.component_seeds)
        }

_seed_manager: Optional[SeedManager] = None

def initialize_seed_manager(master_seed: Optional[int] = None) -> SeedManager:

    global _seed_manager
    _seed_manager = SeedManager(master_seed)
    return _seed_manager

def get_seed_manager() -> SeedManager:

    global _seed_manager
    if _seed_manager is None:
        _seed_manager = SeedManager()
    return _seed_manager

class Components:

    TOPOLOGY = "topology"                   
    POSITION = "position"                       
    LLM_SELECTION = "llm_selection"            
    CONSENSUS = "consensus"                     
    MALICIOUS_BEHAVIOR = "malicious_behavior"          
    EXPERIMENT = "experiment"                  