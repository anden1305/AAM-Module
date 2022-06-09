from AAM import AA

import numpy as np
def calcMI(A1, A2):
    P = A1@A2.T
    PXY = P/sum(sum(P))
    PXPY = np.outer(np.sum(PXY,axis=1), np.sum(PXY,axis=0))
    ind = np.where(PXY>0)
    MI = sum(PXY[ind]*np.log(PXY[ind]/PXPY[ind]))
    return MI


# Normalized mutual information function
def NMI(A1, A2):
    #krav at værdierne i række summer til 1 ???
    NMI = (2*calcMI(A1,A2)) / (calcMI(A1,A1) + calcMI(A2,A2))
    return NMI

AAM = AA()
#AAM.load_csv("ESS8_data.csv",range(12,33), rows = 1000)
# AAM.analyse(AA_type = "OAA")
# AAM.plot(model_type="OAA")

AAM.create_synthetic_data(N = 2000, M=21,K=5, sigma = -4, a_param = 1, b_param = 1000, rb = False)
AAM.analyse(AA_type = "RBOAA", with_synthetic_data=True,K=5, n_iter=3000)
AAM.plot(model_type = "RBOAA", plot_type = "loss_plot", with_synthetic_data=True)
AAM.plot(model_type = "RBOAA", plot_type = "barplot_all", with_synthetic_data=True)


A_anal = AAM.synthetic_results["RBOAA"][0].A
A_true = AAM._synthetic_data.A

NMI_score = NMI(A_anal, A_true)
print(NMI_score)

