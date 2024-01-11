from pdict import *
import numpy as np


templist = [1,1,4,4,6,3,
            5,0,0,3,2,
            2,1,1,0,1,
            1,2,2,3,0,
            0,7,4,7,8,
            8,6,5,8,7,]

p=11
templist[p] = templist[p] + np.random.choice([100,-100])
templist[p+(2*(14-p))] = templist[p]


mutations = [np.random.uniform(0, 100), np.random.uniform(100, 1000), np.random.uniform(1000,10000), np.random.uniform(10000, 100000), 1e20]
multiplier = np.random.choice([1,-1])
temp= round(multiplier*np.random.choice(mutations), 10)
print(temp)


# for i in [46,59,72,85]:
#     print(list(pdict.keys())[i])
