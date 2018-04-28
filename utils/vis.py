# -*- coding: utf-8 -*-
"""
@Time    : 2018/3/23 23:49
@Author  : Elvis
"""
"""
 vis.py
  
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# superclass = "vehicles"
superclass = "hairstyles"
# superclass = "vehicles"
# superclass = "fruits"
w_attr = np.load("data/%s_attr.npy" % superclass)
f = plt.figure(figsize=(20, 10))
sns.heatmap(w_attr, cmap="YlGnBu")
plt.xlabel('%d attributes' % w_attr.shape[1])
plt.ylabel('%d classes' % w_attr.shape[0])
plt.tight_layout()
f.savefig("data/%s_attr.pdf" % superclass)

# <img src="data/zsl_demo.png" width = "80%" />