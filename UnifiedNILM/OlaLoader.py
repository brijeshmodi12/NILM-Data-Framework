# Wrapper for Ola Dataset
import pandas as pd
import os
import json
from UnifiedNILM import TimeSeriesNILMDataset, Channel

class OlaLoader(TimeSeriesNILMDataset):
    def load_metadata(self):
        pass
