import sklearn.datasets
import numpy as np
breast_cancer = sklearn.datasets.load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target
import pandas as pd
data = pd.DataFrame(breast_cancer.data, columns= breast_cancer.feature_names)
data['class'] = Y
