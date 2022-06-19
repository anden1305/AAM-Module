########## IMPORTS ##########
from CAA_class import _CAA
from OAA_class import _OAA
from RBOAA_class import _RBOAA
from TSAA_class import _TSAA
from synthetic_data_class import _synthetic_data
import pandas as pd
import numpy as np
import pickle
from os import path


########## ARCHETYPAL ANALYSIS MODULE CLASS ##########
class AA:
    
    def __init__(self):
        
        self._CAA = _CAA()
        self._OAA = _OAA()
        self._RBOAA = _RBOAA()
        self._TSAA = _TSAA()
        self._results = {"CAA": [], "OAA": [], "RBOAA": [], "TSAA": []}
        self._synthetic_results = {"CAA": [], "OAA": [], "RBOAA": [], "TSAA": []}
        self._has_data = False
        self.has_synthetic_data = False
        self.has_dataframe = False
        self.has_archetype_dataframe = False
        self.archetype_dataframe = pd.DataFrame()
        self.has_ranked_archetype_dataframe = False
        self.ranked_archetype_dataframe = pd.DataFrame()


    def load_data(self, X: np.ndarray, columns: list()):
        self.columns = columns
        self.X = X
        self.N, self.M = X.shape
        self._has_data = True
        if self.N<self.M:
            print("Your data has more attributes than subjects.")
            print(f"Your data has {self.M} attributes and {self.N} subjects.")
            print("This is highly unusual for this type of data.")
            print("Please try loading transposed data instead.")
        else:
            print(f"\nThe data was loaded successfully!\n")


    def load_csv(self, filename: str, columns: list(), rows: int = None, mute: bool = False):
        self.columns, self.M, self.N, self.X = self._clean_data(filename, columns, rows)
        self._has_data = True
        if not mute:
            print(f"\nThe data of \'{filename}\' was loaded successfully!\n")

    
    def _clean_data(self, filename, columns, rows):
        df = pd.read_csv(filename)

        column_names = df.columns.to_numpy()
        
        if not columns is None:
            column_names = column_names[columns]
            X = df[column_names]
        else:
            X = df[column_names]
        if not rows is None:
            X = X.iloc[range(rows),:]
            self.has_dataframe = True
            self.dataframe = df.iloc[range(rows),:]
        else:
            self.has_dataframe = True
            self.dataframe = df

        X = X.to_numpy().T
        M, N = X.shape

        return column_names, M, N, X
    

    def create_synthetic_data(self, N: int = 1000, M: int = 10, K: int = 3, p: int = 6, sigma: float = -20.0, rb: bool = False, b_param: float = 100, a_param: float = 1, sigma_dev: float = 0, mute = False):
        if N < 2:
            print("The value of N can't be less than 2. The value specified was {0}".format(N))
        elif M < 2:
            print("The value of M can't be less than 2. The value specified was {0}".format(M))
        elif K < 2:
            print("The value of K can't be less than 2. The value specified was {0}".format(K))
        elif p < 2:
            print("The value of p can't be less than 2. The value specified was {0}".format(p))
        else:
            self._synthetic_data = _synthetic_data(N, M, K, p, sigma, rb, a_param, b_param, sigma_std = sigma_dev)
            self.has_synthetic_data = True
            self._synthetic_results = {"CAA": [], "OAA": [], "RBOAA": [], "TSAA": []}
            if not mute:
                print("\nThe synthetic data was successfully created! To use the data in an analysis, specificy the with_synthetic_data parameter as True.\n")


    def analyse(self, K: int = 3, p: int = 6, n_iter: int = 1000, early_stopping: bool = True, model_type = "all", lr: float = 0.01, mute: bool = False, with_synthetic_data: bool = False, with_hot_start: bool = False):
        
        success = True

        if model_type == "TSAA":
            print("The model_type TSAA has been deprecated, due to errors in the method.")
            success = False

        elif self._has_data and not with_synthetic_data:
            if model_type == "all" or model_type == "CAA":
                self._results["CAA"].insert(0,self._CAA._compute_archetypes(self.X, K, p, n_iter, lr, mute,self.columns,early_stopping=early_stopping))
            elif model_type == "all" or model_type == "OAA":
                self._results["OAA"].insert(0,self._OAA._compute_archetypes(self.X, K, p, n_iter, lr, mute,self.columns,with_synthetic_data=False,early_stopping=early_stopping))
            elif model_type == "all" or model_type == "RBOAA":
                self._results["RBOAA"].insert(0,self._RBOAA._compute_archetypes(self.X, K, p, n_iter, lr, mute,self.columns, with_synthetic_data=False, early_stopping=early_stopping, with_OAA_initialization = with_hot_start))
            elif model_type == "all" or model_type == "TSAA":
                self._results["TSAA"].insert(0,self._TSAA._compute_archetypes(self.X, K, p, n_iter, lr, mute,self.columns,early_stopping=early_stopping))
            else:
                print("The model_type \"{0}\" specified, does not match any of the possible AA_types.".format(model_type))
                success = False
    
        elif self.has_synthetic_data and with_synthetic_data:
            if model_type == "all" or model_type == "CAA":
                self._synthetic_results ["CAA"].insert(0,self._CAA._compute_archetypes(self._synthetic_data.X, K, p, n_iter, lr, mute, self._synthetic_data.columns, with_synthetic_data=True,early_stopping=early_stopping))
            elif model_type == "all" or model_type == "OAA":
                self._synthetic_results["OAA"].insert(0,self._OAA._compute_archetypes(self._synthetic_data.X, K, p, n_iter, lr, mute, self._synthetic_data.columns,with_synthetic_data=True,early_stopping=early_stopping,for_hotstart_usage=False))
            elif model_type == "all" or model_type == "RBOAA":
                self._synthetic_results["RBOAA"].insert(0,self._RBOAA._compute_archetypes(self._synthetic_data.X, K, p, n_iter, lr, mute, self._synthetic_data.columns,with_synthetic_data=True,early_stopping=early_stopping,with_OAA_initialization = with_hot_start))
            elif model_type == "all" or model_type == "TSAA":
                self._synthetic_results["TSAA"].insert(0,self._TSAA._compute_archetypes(self._synthetic_data.X, K, p, n_iter, lr, mute, self._synthetic_data.columns,with_synthetic_data=True,early_stopping=early_stopping))
            else:
                print("The model_type \"{0}\" specified, does not match any of the possible AA_types.".format(model_type))
                success = False
        
        else:
            print("\nYou have not loaded any data yet! \nPlease load data through the \'load_data\' or \'load_csv\' methods and try again.\n")
            success = False

        if success and self.has_dataframe:
            self.create_dataframe(model_type=model_type,with_synthetic_data=with_synthetic_data,mute=True)
            self.create_dataframe(model_type=model_type,with_synthetic_data=with_synthetic_data,archetype_rank=3,mute=True)


    def plot(self, 
            model_type: str = "CAA", 
            plot_type: str = "PCA_scatter_plot", 
            title: str = "",
            save_figure: bool = False,
            filename: str = "figure",
            result_number: int = 0, 
            attributes: list() = [1,2], 
            archetype_number: int = 1, 
            types: dict = {"type 1": [1],"type 2": [2]},
            weighted: str = "equal_norm",
            subject_indexes: list() = [1],
            attribute_indexes: list() = [],
            with_synthetic_data: bool = False):
        
        if not model_type in ["CAA", "OAA", "RBOAA", "TSAA"]:
            print("\nThe model type you have specified can not be recognized. Please try again.")
        elif not plot_type in ["PCA_scatter_plot","attribute_scatter_plot","loss_plot","mixture_plot","barplot","barplot_all","typal_plot","pie_chart","attribute_distribution","circular_typal_barplot"]:
            print("\nThe plot type you have specified can not be recognized. Please try again.\n")
        elif not weighted in ["none","equal_norm","equal","norm"]:
            print(f"\nThe \'weighted\' parameter received an unexpected value of {weighted}.\n")
        elif not attribute_indexes == [] and not self.has_archetype_dataframe:
            print(f"\nYou have not created any dataframe to plot w.r.t..\n")
        
        elif not with_synthetic_data:
            if result_number < 0 or not result_number < len(self._results[model_type]):
                print("\nThe result you are requesting to plot is not availabe.\n Please make sure you have specified the input correctly.\n")
            elif archetype_number < 1 or archetype_number > self._results[model_type][result_number].K:
                print(f"\nThe \'archetype_number\' parameter received an unexpected value of {archetype_number}.\n")
            elif any(np.array(attributes) < 0) or any(np.array(attributes) > len(self._results[model_type][result_number].columns)):
                print(f"\nThe \'attributes\' parameter received an unexpected value of {attributes}.\n")
            elif any(np.array(subject_indexes) < 0) or any(np.array(subject_indexes) > self._results[model_type][result_number].N-1):
                print(f"\nThe \'subject_indexes\' parameter received an unexpected value of {subject_indexes}.\n")
            else:
                result = self._results[model_type][result_number]
                result._plot(plot_type,attributes,archetype_number,types,weighted,subject_indexes,attribute_indexes, self.archetype_dataframe ,save_figure,filename,title)
                if save_figure:
                    print("\nThe requested plot was successfully saved to your device!\n")
                else:
                    print("\nThe requested plot was successfully plotted!\n")
        
        else:
            if result_number < 0 or not result_number < len(self._synthetic_results[model_type]):
                print("\nThe result you are requesting to plot is not available.\n Please make sure you have specified the input correctly.\n")
            elif archetype_number < 0 or archetype_number > self._synthetic_results[model_type][result_number].K:
                print(f"\nThe \'archetype_number\' parameter received an unexpected value of {archetype_number}.\n")
            elif any(np.array(attributes) < 0) or any(np.array(attributes) > len(self._synthetic_results[model_type][result_number].columns)):
                print(f"\nThe \'attributes\' parameter received an unexpected value of {attributes}.\n")
            elif any(np.array(subject_indexes) < 0) or any(np.array(subject_indexes) > self._synthetic_results[model_type][result_number].N-1):
                print(f"\nThe \'subject_indexes\' parameter received an unexpected value of {subject_indexes}.\n")
            else:
                result = self._synthetic_results[model_type][result_number]
                result._plot(plot_type,attributes,archetype_number,types,weighted,subject_indexes,attribute_indexes, self.archetype_dataframe,save_figure,filename,title)
                print("\nThe requested synthetic data result plot was successfully plotted!\n")


    def save_analysis(self,filename: str = "analysis",model_type: str = "CAA", result_number: int = 0, with_synthetic_data: bool = False, save_synthetic_data: bool = True):

        if not model_type in ["CAA", "OAA", "RBOAA", "TSAA"]:
            print("\nThe model type you have specified can not be recognized. Please try again.\n")
        
        if not with_synthetic_data:
            if not result_number < len(self._results[model_type]):
                print("\nThe analysis you are requesting to save is not available.\n Please make sure you have specified the input correctly.\n")
            
            
            else:
                self._results[model_type][result_number]._save(filename)
                print("\nThe analysis was successfully saved!\n")
        else:
            if not result_number < len(self._synthetic_results[model_type]):
                print("\nThe analysis with synthetic data, which you are requesting to save is not available.\n Please make sure you have specified the input correctly.\n")
            
            else:
                self._synthetic_results[model_type][result_number]._save(filename)
                if save_synthetic_data:
                    self._synthetic_data._save(model_type,filename)
                print("\nThe analysis was successfully saved!\n")

    
    def load_analysis(self, filename: str = "analysis", model_type: str = "CAA", with_synthetic_data: bool = False):
        if not model_type in ["CAA", "OAA", "RBOAA", "TSAA"]:
            print("\nThe model type you have specified can not be recognized. Please try again.\n")

        elif not with_synthetic_data:
            if not path.exists("results/" + model_type + "_" + filename + '.obj'):
                print(f"The analysis {filename} of type {model_type} does not exist on your device.")
            
            
            else:
                file = open("results/" + model_type + "_" + filename + '.obj','rb')
                result = pickle.load(file)
                file.close()
                self._results[model_type].append(result)
                print("\nThe analysis was successfully loaded!\n")

        else:
            if not path.exists("synthetic_results/" + model_type + "_" + filename + '.obj'):
                print(f"The analysis {filename} with synthetic data of type {model_type} does not exist on your device.")
            
            else:
                file = open("synthetic_results/" + model_type + "_" + filename + '.obj','rb')
                result = pickle.load(file)
                file.close()
                self._synthetic_results[model_type].append(result)

                file = open("synthetic_results/" + model_type + "_" + filename + '_metadata' + '.obj','rb')
                result = pickle.load(file)
                file.close()
                self._synthetic_data = result

                print("\nThe analysis with synthetic data was successfully loaded!\n")
                self.has_synthetic_data = True


    def create_dataframe(self, model_type: str = "CAA", result_number: int = 0, with_synthetic_data: bool = False, archetype_rank: int = 0, return_dataframe: bool = False, mute = False):
        
        if not model_type in ["CAA", "OAA", "RBOAA", "TSAA"]:
            print("\nThe model type you have specified can not be recognized. Please try again.\n")
            return

        if (with_synthetic_data and len(self._synthetic_results[model_type]) <= result_number) or (not with_synthetic_data and len(self._results[model_type]) <= result_number):
            print("\nThe result you have specified to create the dataframe can not be recognized. Please try again.\n")
            return

        if with_synthetic_data:
            result = self._synthetic_results[model_type][result_number]
        else:
            result = self._results[model_type][result_number]

        if archetype_rank == False:
            if self.has_dataframe:
                self.archetype_dataframe = self.dataframe.copy()
                for archetype in range(result.K):
                    self.archetype_dataframe["Archetype {0}".format(archetype+1)] = result.A[archetype,:]
                if not mute:
                    print("\nThe dataframe was successfully created from a copy of your imported dataframe.\n")
                
            else:
                dict = {}
                for archetype in range(result.K):
                    dict["Archetype {0}".format(archetype+1)] = result.A[archetype,:]
                archetype_dataframe = pd.DataFrame.from_dict(dict)
                self.archetype_dataframe = archetype_dataframe
                if not mute:
                    print("\nThe dataframe was successfully created.\n")

        else:
            if self.has_dataframe:
                self.ranked_archetype_dataframe = self.dataframe.copy()
                for rank in range(archetype_rank):
                    rank_list = []
                    for n in range(result.N):
                        rank_list.append(np.where(result.A[:,n] == np.sort(result.A[:,n])[::-1][rank])[0][0]+1)
                    self.ranked_archetype_dataframe["Archetype Rank {0}".format(rank+1)] = rank_list
                if not mute:
                    print("\nThe dataframe was successfully created from a copy of your imported dataframe.\n")

        if archetype_rank == False:
            self.has_archetype_dataframe = True
            if return_dataframe:
                return self.archetype_dataframe
        else:
            self.has_ranked_archetype_dataframe = True
            if return_dataframe:
                return self.ranked_archetype_dataframe


    def get_dataframe(self,ranked_dataframe: bool = False):

        if not ranked_dataframe:
            if not self.has_archetype_dataframe:
                print("\nThe dataframe which you have requested, does not exist yet.\n")
            
            return self.archetype_dataframe
        
        else:
            if not self.has_ranked_archetype_dataframe:
                print("\nThe dataframe which you have requested, does not exist yet.\n")
            
            return self.ranked_archetype_dataframe


    def get_analysis(self, model_type: str = "CAA", result_number: int = 0, with_synthetic_data: bool = False):

        if not model_type in ["CAA", "OAA", "RBOAA", "TSAA"]:
            print("\nThe model type you have specified can not be recognized. Please try again.\n")
            return

        if (with_synthetic_data and len(self._synthetic_results[model_type]) <= result_number) or (not with_synthetic_data and len(self._results[model_type]) <= result_number):
            print("\nThe result you have specified to create the dataframe can not be recognized. Please try again.\n")
            return

        if with_synthetic_data:
            result = self._synthetic_results[model_type][result_number]
        else:
            result = self._results[model_type][result_number]
        
        return result
