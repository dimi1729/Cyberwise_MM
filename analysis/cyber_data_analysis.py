import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from typing import List, Dict, Optional, Any, Tuple

class CyberDataDate:
    def __init__(self, attack_date: str):
        # Dates are stored dd/mm/yyyy 0:00
        self.date_string = attack_date.split(" ")[0] # time is always stored as 0:00 so remove it
        self.date_object = datetime.strptime(self.date_string, '%d/%m/%Y')
    
    def linearize(self) -> int:
        # Earliest day is 10/11/2022 (dd/mm/yyyy)
        reference_date = datetime(2022, 10, 11)
        diff = self.date_object - reference_date
        return diff.days


if __name__ == "__main__":
    df = pd.read_csv("data/cyber_data.csv")
    cyber_data_days = df["AttackDate"].apply(lambda x: CyberDataDate(x).linearize())

    plt.figure(figsize=(10, 6))
    plt.hist(cyber_data_days, bins=30)
    plt.xlabel('Days since 10/11/2022')
    plt.ylabel('Number of Attacks')
    plt.grid(False)
    plt.show()



        