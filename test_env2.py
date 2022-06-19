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

# AAM.plot("RBOAA","pie_chart","Denmark - education high - Male - Liberal",attribute_indexes=[("Country","=","DE"),("education","=",7),("gender","=",2),("left_right", ">", 5),("left_right", "<", 11)])
# AAM.plot("RBOAA","pie_chart","Russia - education low - Female - Social",attribute_indexes=[("Country","=","RU"),("education","=",1),("gender","=",1),("left_right", "<", 5)])

# AAM.plot("RBOAA","circular_typal_barplot",types={"Type 1": [1,2,3],"Type 2": [4,5,6],"Type 3": [7,8,9],"Type 4": [10,11,12]},archetype_number=1)
# AAM.plot("RBOAA","circular_typal_barplot",types={"Type 1": [1,2,3],"Type 2": [4,5,6],"Type 3": [7,8,9],"Type 4": [10,11,12]},archetype_number=2)
# AAM.plot("RBOAA","circular_typal_barplot",types={"Type 1": [1,2,3],"Type 2": [4,5,6],"Type 3": [7,8,9],"Type 4": [10,11,12]},archetype_number=3)
# AAM.plot("RBOAA","circular_typal_barplot",types={"Type 1": [1,2,3],"Type 2": [4,5,6],"Type 3": [7,8,9],"Type 4": [10,11,12]},archetype_number=4)

# AAM.plot("RBOAA","barplot_all")
# AAM.plot("RBOAA","pie_chart",attribute_indexes=[("Country","=","DE")])
# AAM.plot("RBOAA","attribute_distribution",attribute_indexes=[("Country","=","DE")])

types4 = {'Openness to Change': [6,15,1,11,10,21],
         'Self-Trancendence'  : [3,8,12,18],
         'Conservation'       : [5,14,7,16,20,9],
         'Self-Enhancement'   : [2,4,13,17,10,21]}

types10 = {'Self-Direction'   : [1,11],
           'Stimulation'      : [6,15],
           'Hedonism'         : [10,21],
           'Achievement'      : [4,13],
           'Power'            : [2,17],
           'Security'         : [5,14],
           'Conformity'       : [7,16],
           'Tradition'        : [9,20],
           'Benevolence'      : [12,18],
           'Universalism'     : [3,8,19]
           }

AAM.plot("RBOAA","circular_typal_barplot",archetype_number=2,types=types10)