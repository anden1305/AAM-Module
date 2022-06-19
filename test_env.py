from AAM import AA
import numpy as np

AAM = AA()

AAM.load_csv("ESS8_data.csv",np.arange(13,25),10000)
AAM.analyse(AA_type = "RBOAA", with_synthetic_data=False,K=5, n_iter=3000)
print(AAM._results["RBOAA"][0].A)
AAM.plot(model_type = "RBOAA", plot_type = "mixture_plot", with_synthetic_data=False)
