from json import load
from matplotlib.pyplot import pie
from sklearn.datasets import load_sample_image
from AAM import AA
import numpy as np

AAM = AA()

AAM.load_csv("ESS8_data.csv",np.arange(12,32))
# AAM.analyse(5,6,2000,True,"RBOAA",0.01,False,False,True)
AAM.load_analysis("K=4 - RBOAA","RBOAA",False)
AAM._results["RBOAA"][0].N = len(AAM._results["RBOAA"][0].X[0,:])
AAM.create_dataframe("RBOAA",0,False,0,False,False)

AAM.plot("RBOAA","pie_chart","Denmark - education high - Male - Liberal",attribute_indexes=[("Country","=","DE"),("education","=",7),("gender","=",2),("left_right", ">", 5),("left_right", "<", 11)])
AAM.plot("RBOAA","pie_chart","Russia - education low - Female - Social",attribute_indexes=[("Country","=","RU"),("education","=",1),("gender","=",1),("left_right", "<", 5)])

# AAM.plot("RBOAA","circular_typal_barplot",types={"Type 1": [1,2,3],"Type 2": [4,5,6],"Type 3": [7,8,9],"Type 4": [10,11,12]},archetype_number=1)