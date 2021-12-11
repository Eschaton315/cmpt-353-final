The main Code is project.py and the parameter configuration is the following:
interventions.csv month.csv monthlyChange.csv dates.csv datesPredict.csv

Thus, a command such as:

python3 project.py interventions.csv month.csv monthlyChange.csv dates.csv datesPredict.csv

will run the code as intended

The only imports we used that we dont remember in any of the excercises are:

pandas.plotting


The rest are import such as:

import numpy as np
import pandas as pd
import sys
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
