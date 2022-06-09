from AAM import AA

AAM = AA()
#AAM.load_csv("ESS8_data.csv",range(12,33), rows = 1000)
# AAM.analyse(AA_type = "OAA")
# AAM.plot(model_type="OAA")

AAM.create_synthetic_data(N = 1000, M=21,K=5)
AAM.analyse(AA_type = "RBOAA", with_synthetic_data=True,K=5, n_iter=3000)
AAM.plot(model_type = "RBOAA", plot_type = "loss_plot", with_synthetic_data=True)
AAM.plot(model_type = "RBOAA", plot_type = "barplot_all", with_synthetic_data=True)

