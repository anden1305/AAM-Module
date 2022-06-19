########## IMPORT ##########
from matplotlib.pyplot import plot
from plots_class import _plots
import pickle


########## CONVENTIONAL ARCHETYPAL ANALYSIS RESULT ##########
class _CAA_result:

    plots = _plots()
    
    def __init__(self, A, B, X, X_hat, n_iter, RSS, Z, K, p, time, columns,type, with_synthetic_data = False):
        self.A = A
        self.B = B
        self.X = X
        self.X_hat  = X_hat
        self.n_iter = len(RSS)
        self.loss = RSS
        self.Z = Z
        self.K = K
        self.p = p
        self.time = time
        self.columns = columns
        self.type = type
        self.with_synthetic_data = with_synthetic_data
        self.N = len(self.X[0,:])

    def _print(self):
        if self.type == "CAA":
            type_name = "Conventional Archetypal"
        else:
            type_name = "Two Step Archetypal"
        print("/////////////// INFORMATION ABOUT " + type_name.upper() + " ANALYSIS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
        print(f"▣ The " + type_name + " Analysis was computed using " + str(self.K) + " archetypes.")
        print(f"▣ The " + type_name + " Analysis was computed on " + str(len(self.X)) + " attributes.")
        print(f"▣ The " + type_name + " Analysis was computed on " + str(self.N) + " subjects.")
        print(f"▣ The " + type_name + " Analysis ran for " + str(self.n_iter) + " iterations.")
        print(f"▣ The " + type_name + " Analysis took " + str(self.time) + " seconds to complete.")
        print(f"▣ The final RSS was: {self.loss[-1]}.")

    def _plot(self,plot_type, attributes, archetype_number, types, weighted, subject_indexes, attribute_indexes, archetype_dataframe, save_figure, filename,title):
        
        if plot_type == "PCA_scatter_plot":
            self.plots._PCA_scatter_plot(self.Z,self.X_hat,self.type, save_figure, filename,title)
        elif plot_type == "attribute_scatter_plot":
            self.plots._attribute_scatter_plot(self.Z,self.X_hat,attributes,self.type,self.p, save_figure, filename,title)
        elif plot_type == "loss_plot":
            self.plots._loss_plot(self.loss,self.type,  save_figure, filename,title)
        elif plot_type == "mixture_plot":
            self.plots._mixture_plot(self.Z,self.A,self.type,  save_figure, filename,title)
        elif plot_type == "barplot":
            self.plots._barplot(self.Z,self.columns,archetype_number,self.type, self.p,  save_figure, filename,title)
        elif plot_type == "barplot_all":
            self.plots._barplot_all(self.Z,self.columns, self.type, self.p,  save_figure, filename,title)
        elif plot_type == "typal_plot":
            self.plots._typal_plot(self.Z,types,weighted,  save_figure, filename,title)
        elif plot_type == "pie_chart":
            self.plots._pie_chart(self.A, subject_indexes, attribute_indexes, archetype_dataframe,  save_figure, filename,title)
        elif plot_type == "attribute_distribution":
            self.plots._attribute_distribution(self.A,self.Z,subject_indexes,self.columns,self.p,self.type, attribute_indexes, archetype_dataframe,  save_figure, filename,title)
        elif plot_type == "circular_typal_barplot":
            self.plots._circular_typal_barplot(self.type,self.Z, types, archetype_number,self.columns,self.p, save_figure, filename, title)

    def _save(self,filename):
        if not self.with_synthetic_data:
            file = open("results/" + self.type + "_" + filename + '.obj','wb')
        else:
            file = open("synthetic_results/" + self.type + "_" + filename + '.obj','wb')
        pickle.dump(self, file)
        file.close()


########## ORDINAL ARCHETYPAL ANALYSIS RESULT ##########
class _OAA_result:

    plots = _plots()
    
    def __init__(self, A, B, X, n_iter, b, Z, X_tilde, Z_tilde, X_hat, loss, K, p, time, columns,type,sigma, with_synthetic_data = False):
        self.A = A
        self.B = B
        self.X = X
        self.n_iter = len(loss)
        self.b = b
        self.sigma = sigma
        self.X_tilde = X_tilde
        self.Z_tilde = Z_tilde
        self.X_hat = X_hat
        self.loss = loss
        self.Z = Z
        self.K = K
        self.p = p
        self.time = time
        self.columns = columns
        self.type = type
        self.with_synthetic_data = with_synthetic_data
        self.N = len(self.X[0,:])

    def _print(self):
        if self.type == "RBOAA":
            type_name = "Response Bias Ordinal Archetypal"
        else:
            type_name = "Ordinal Archetypal"
        
        print("/////////////// INFORMATION ABOUT " + type_name.upper() + " ANALYSIS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
        print(f"▣ The " + type_name + " Analysis was computed using " + str(self.K) + " archetypes.")
        print(f"▣ The " + type_name + " Analysis was computed on " + str(len(self.X)) + " attributes.")
        print(f"▣ The " + type_name + " Analysis was computed on " + str(self.N) + " subjects.")
        print(f"▣ The " + type_name + " Analysis ran for " + str(self.n_iter) + " iterations.")
        print(f"▣ The " + type_name + " Analysis took " + str(self.time) + " seconds to complete.")
        print(f"▣ The final loss was: {self.loss[-1]}.")
    
    def _plot(self,plot_type, attributes, archetype_number, types, weighted, subject_indexes, attribute_indexes, archetype_dataframe, save_figure, filename,title):
        
        if plot_type == "PCA_scatter_plot":
            self.plots._PCA_scatter_plot(self.Z,self.X_hat,self.type, save_figure, filename,title)
        elif plot_type == "attribute_scatter_plot":
            self.plots._attribute_scatter_plot(self.Z,self.X_hat,attributes,self.type, self.p, save_figure, filename,title)
        elif plot_type == "loss_plot":
            self.plots._loss_plot(self.loss,self.type,  save_figure, filename,title)
        elif plot_type == "mixture_plot":
            self.plots._mixture_plot(self.Z,self.A,self.type,  save_figure, filename,title)
        elif plot_type == "barplot":
            self.plots._barplot(self.Z,self.columns,archetype_number,self.type,self.p,  save_figure, filename,title)
        elif plot_type == "barplot_all":
            self.plots._barplot_all(self.Z,self.columns, self.type, self.p,  save_figure, filename,title)
        elif plot_type == "typal_plot":
            self.plots._typal_plot(self.Z,types,weighted,  save_figure, filename,title)
        elif plot_type == "pie_chart":
            self.plots._pie_chart(self.A,subject_indexes, attribute_indexes, archetype_dataframe,  save_figure, filename,title)
        elif plot_type == "attribute_distribution":
            self.plots._attribute_distribution(self.A,self.Z,subject_indexes,self.columns,self.p,self.type,attribute_indexes, archetype_dataframe,  save_figure, filename,title)
        elif plot_type == "circular_typal_barplot":
            self.plots._circular_typal_barplot(self.type,self.Z, types, archetype_number,self.columns,self.p,save_figure, filename, title)

    def _save(self,filename):
        if not self.with_synthetic_data:
            file = open("results/" + self.type + "_" + filename + '.obj','wb')
        else:
            file = open("synthetic_results/" + self.type + "_" + filename + '.obj','wb')
        pickle.dump(self, file)
        file.close()