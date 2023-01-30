import pandas as pd
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import numpy as np

from work import Prospector
from mkl.acquisition import GreedyNRanking


RAND = np.random.RandomState(1)

df = pd.read_csv('COF_pct_deliverablecapacityvSTPv.csv')
subsample = RAND.choice(len(df), size=1000, replace=False)
X, y = df.iloc[subsample, :5].values, df.iloc[subsample, -1].values
X = scale(X)

n_top = 100
y_top = np.argsort(y)[-n_top:]

# ----------------------------------------------------
tested = list(RAND.choice(len(X), size=50))
untested = [i for i in range(len(X)) if i not in tested]
y_tested = y[tested].reshape(-1)

model = Prospector(X=X)
acqu = GreedyNRanking()

for itr in range(50, 100):
        
    untested = [i for i in range(len(X)) if i not in tested]
    ytested = y[tested].reshape(-1)

    model.fit(tested, untested, ytested)
    posterior = model.samples(nsamples=1)
    alpha = acqu.score_points(posterior)
    alpha_ranked = np.argsort(alpha)[::-1]
    to_sample = [i for i in alpha_ranked if i not in tested][0]
    
    tested.append(to_sample)    
    n_top_found = sum([1 for i in tested if i in y_top])
    
    print(F'[{itr} / 100] : {n_top_found}/{n_top} found : y_max_tested={max(y):.3f} : y_sampled={y[to_sample]}')
    
    
    