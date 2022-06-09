from AAM import AA

AAM = AA()
AAM.load_csv("ESS8_data.csv",range(12,33), rows = 10000)
# AAM.analyse(AA_type = "OAA")
# AAM.plot(model_type="OAA")

# AAM.create_synthetic_data(N = 100000, M=15,K=5)
AAM.analyse(AA_type = "RBOAA", with_synthetic_data=False,K=5)
AAM.plot(model_type = "RBOAA",n_iter = 3000, plot_type = "barplot_all", with_synthetic_data=False)
