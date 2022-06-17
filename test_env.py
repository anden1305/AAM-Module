from matplotlib.pyplot import pie
from AAM import AA
import numpy as np

AAM = AA()

AAM.load_analysis("K=4 - RBOAA", "RBOAA",False)
AAM._results["RBOAA"][0].N = len(AAM._results["RBOAA"][0].X[0,:])
AAM.plot("RBOAA","barplot_all")


for i in range(4):
    AAM.plot("RBOAA","barplot",archetype_number=i+1)
    