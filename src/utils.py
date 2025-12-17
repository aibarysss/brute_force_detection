# src/utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


def setup_plotting():
    """Setup plotting parameters"""
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14


def print_separator(title=""):
    """Print separator with title"""
    if title:
        print(f"\n{'=' * 60}")
        print(f"{title.center(60)}")
        print(f"{'=' * 60}\n")
    else:
        print(f"\n{'=' * 60}\n")


# Initialize plotting setup
setup_plotting()