import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

class Alpha:
    def __init__(self, data):
        self.data = data

    def calculate_efg(self, time):
        return 0