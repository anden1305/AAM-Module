########## IMPORT ##########
#from turtle import color
from cProfile import label
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


########## PLOTS CLASS ##########
class _plots:

    def _PCA_scatter_plot(self,Z,X,type,save_fig,filename, title):
        
        pca = PCA(n_components=2)
        pca.fit(Z.T)
        
        Z_pca = pca.transform(Z.T)
        X_pca = pca.transform(X.T)
        
        plt.rcParams["figure.figsize"] = (10,10)
        plt.scatter(X_pca[:,0], X_pca[:,1], c ="black", s = 1)
        for a in range(len(Z[0,:])):
            plt.scatter(Z_pca[a,0], Z_pca[a,1], marker ="^", s = 500, label="Archetype {0}".format(a+1))
        plt.xlabel("Principal Component 1", fontsize=15)
        plt.ylabel("Principal Component 2", fontsize=15)
        if title == "":
            plt.title(f"PCA Scatter Plot of {type}", fontsize = 20)
        else:
            plt.title(title, fontsize = 20)
        plt.legend(prop={'size': 15})

        if not save_fig:
            plt.show()
        else:
            plt.savefig("{0}.png".format(filename),dpi=300)


    def _attribute_scatter_plot(self,Z,X,attributes,type,p, save_fig, filename,title):
        
        plt.rcParams["figure.figsize"] = (10,10)
        plt.scatter(X[attributes[0]-1,:]*p, X[attributes[1]-1,:]*p, c ="black", s = 1)
        for a in range(len(Z[0,:])):
            plt.scatter(Z[attributes[0]-1,a]*p, Z[attributes[1]-1,a]*p, marker ="^", s = 500, label="Archetype {0}".format(a+1))
        plt.xlabel(f"Attribute {attributes[0]}", fontsize=15)
        plt.ylabel(f"Attribute {attributes[1]}", fontsize=15)
        plt.legend(prop={'size': 15})
        if title == "":
            plt.title(f"Attribute Scatter Plot of {type}", fontsize = 20)
        else:
            plt.title(title, fontsize = 20)

        if not save_fig:
            plt.show()
        else:
            plt.savefig("{0}.png".format(filename),dpi=300)


    def _loss_plot(self,loss,type, save_fig, filename,title):
        plt.plot(loss, c="#2c6c8c")
        plt.xlabel(f"Iteration of {type}")
        plt.ylabel(f"Loss of {type}")
        if title == "":
            plt.title(f"Loss w.r.t. Itteration of {type}")
        else:
            plt.title(title)
        
        if not save_fig:
            plt.show()
        else:
            plt.savefig("{0}.png".format(filename),dpi=300)


    def _mixture_plot(self,Z,A,type, save_fig, filename,title):

        plt.rcParams["figure.figsize"] = (10,10)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        K = len(Z.T)
        corners = []
        for k in range(K):
            corners.append([np.cos(((2*np.pi)/K)*(k)), np.sin(((2*np.pi)/K)*(k))])
            plt.plot(
                np.cos(((2*np.pi)/K)*(k)), 
                np.sin(((2*np.pi)/K)*(k)), 
                marker="o", markersize=12, 
                markeredgecolor="black", 
                zorder=10,
                label = "Archetype {0}".format(k+1))

        points_x = []
        points_y = []
        for p in A.T:
            x = 0
            y = 0
            for k in range(K):
                x += p[k] * np.cos(((2*np.pi)/K)*(k))
                y += p[k] * np.sin(((2*np.pi)/K)*(k))
            points_x.append(x)
            points_y.append(y)
        
        p = Polygon(corners, closed=False,zorder=0)
        ax.add_patch(p)
        ax.set_xlim(-1.1,1.1)
        ax.set_ylim(-1.1,1.1)
        ax.set_aspect('equal')
        if title == "":
            plt.title(f"Mixture Plot of {type}", fontsize = 20)
        else:
            plt.title(title, fontsize = 20)
        plt.scatter(points_x, points_y, c ="black", s = 1, zorder=5)
        plt.legend()
        
        if not save_fig:
            plt.show()
        else:
            plt.savefig("{0}.png".format(filename),dpi=300)


    def _barplot(self,Z,columns, archetype_number,type,p, save_fig, filename,title):
        
        plt.rcParams["figure.figsize"] = (10,10)
        archetype = Z.T[archetype_number-1]
        if type in ["OAA","RBOAA"]:
            archetype *=p
        fig, ax = plt.subplots()
        ax.set_ylabel('Value')
        plt.xlabel('Attributes')
        if title == "":
            ax.set_title(f"Value-Distribution of Archeype {archetype_number}")
        else:
            ax.set_title(title)
        ax.bar(np.arange(len(archetype)),archetype)
        ax.set_xticks(np.arange(len(archetype)))
        ax.set_xticklabels(labels=columns)
        plt.ylim(0, p+0.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.set_size_inches(10, 10)
        
        if not save_fig:
            plt.show()
        else:
            plt.savefig("{0}.png".format(filename),dpi=300)


    def _barplot_all(self,Z,columns,type, p,  save_fig, filename,title):
        plt.rcParams["figure.figsize"] = (10,10)
        data = []
        names = ["Attributes"]
        
        for (arch, column) in zip(Z,columns):
            current_data = [column]
            for value in arch:
                if type in ["OAA","RBOAA"]:
                    value *=p
                current_data.append(value)
                
            data.append(current_data)
        
        for i in range(len(Z.T)):
            names.append("Archetype {0}".format(i+1))

        df=pd.DataFrame(data,columns=names)
        df.plot(x="Attributes", y=names[1:], kind="bar",figsize=(10,10))
        plt.ylim(0.0, p+0.5)
        plt.ylabel(f"Value")
        if title == "":
            plt.title(f"Value-Distribution over All Archetypes")
        else:
            plt.title(title)
        
        if not save_fig:
            plt.show()
        else:
            plt.savefig("{0}.png".format(filename),dpi=300)


    def _typal_plot(self, Z, types, weighted,  save_fig, filename,title):
        plt.rcParams["figure.figsize"] = (10,10)
        fig, ax = plt.subplots()
        type_names = types.keys()
        type_names_display = list(types.keys())
        labels = [f"Archetype {i}" for i in range(len(Z.T))]
        width = 0.5
        bottoms = []
        bottoms.append([0 for i in range(len(Z.T))])
        values = []
        for label in type_names:
            label_values = []
            for archetype in Z.T:
                archetype_value = 0
                for i in types[label-1]:
                    archetype_value += archetype[i]
                if weighted in ["equal","equal_norm"]:
                    archetype_value = archetype_value / len(types[label])
                label_values.append(archetype_value)
            values.append(label_values)
        
        values_new = np.array(values)

        if weighted in ["norm","equal_norm"]:
            for i in range(len(values)):
                values_new[i] = values_new[i] / np.sum(values,0)

        for i in range(len(values)-1):
            bottoms.append([b + l for (b,l) in zip(bottoms[-1],values_new[i])])

        for i in range(len(values)):
            ax.bar(labels, values_new[i], width, bottom=bottoms[i], label=type_names_display[i])
        ax.set_ylabel('Value')
        if title == "":
            ax.set_title('Typal Composition of Archetypes')
        else:
            ax.set_title(title)
        ax.legend()

        if not save_fig:
            plt.show()
        else:
            plt.savefig("{0}.png".format(filename),dpi=300)
        

    def _pie_chart(self, A, indexes, attribute_indexes, archetype_dataframe,  save_fig, filename,title):

        data = []

        if attribute_indexes == []:

            for i in range(len(indexes)):
                datapoint = A.T[indexes[i]]
                if len(data) == 0:
                    data = datapoint
                else:
                    data = data + datapoint / 2

        else:
            data_subset = archetype_dataframe.copy()

            for pair in attribute_indexes:
                if pair[1] == "=":
                    data_subset = data_subset.loc[data_subset[pair[0]] == pair[2]]
                elif pair[1] == "<":
                    data_subset = data_subset.loc[data_subset[pair[0]] < pair[2]]
                elif pair[1] == ">":
                    data_subset = data_subset.loc[data_subset[pair[0]] > pair[2]]
            
            if data_subset.shape[0] < 1:
                print("\nThere are no datapoints with the value(s) given by the 'attribute_indexes' parameter.")
                return

            for i in range(data_subset.shape[0]):
                datapoint = A.T[i]
                if len(data) == 0:
                    data = datapoint
                else:
                    data = data + datapoint / 2


        labels = []
        explode = []
        for i in range(len(A.T[0])):
            labels.append("Archetype {0}".format(i+1))
            if data[i] == np.max(data):
                explode.append(0.1)
            else:
                explode.append(0.0)

        plt.pie(data,explode=tuple(explode), labels = labels, shadow=True, startangle=90, autopct='%1.1f%%')
        if title == "":
            plt.title("Pie Chart of Archetype Distribution on Given Subset of Data")
        else:
            plt.title(title)
        
        if not save_fig:
            plt.show()
        else:
            plt.savefig("{0}.png".format(filename),dpi=300)
        

    def _attribute_distribution(self, A, Z, indexes, columns, p, type, attribute_indexes, archetype_dataframe,  save_fig, filename, title):
        
        archetype_distribution = []

        if attribute_indexes == []:

            for i in range(len(indexes)):
                datapoint = A.T[indexes[i]]
                if len(archetype_distribution) == 0:
                    archetype_distribution = datapoint
                else:
                    archetype_distribution = archetype_distribution + datapoint / 2

        else:
            data_subset = archetype_dataframe.copy()

            for pair in attribute_indexes:
                if pair[1] == "=":
                    data_subset = data_subset.loc[data_subset[pair[0]] == pair[2]]
                elif pair[1] == "<":
                    data_subset = data_subset.loc[data_subset[pair[0]] < pair[2]]
                elif pair[1] == ">":
                    data_subset = data_subset.loc[data_subset[pair[0]] > pair[2]]
            
            if data_subset.shape[0] < 1:
                print("\nThere are no datapoints with the value {0} of attribute {1}.\n".format(attribute_indexes[1],attribute_indexes[0]))
                return

            for i in range(data_subset.shape[0]):
                datapoint = A.T[i]
                if len(archetype_distribution) == 0:
                    archetype_distribution = datapoint
                else:
                    archetype_distribution = archetype_distribution + datapoint / 2
        
        archetype_distribution = archetype_distribution/np.sum(archetype_distribution)

        attribute_distribution = []

        for a in range(len(archetype_distribution)):
            if len(attribute_distribution) == 0:
                attribute_distribution = Z.T[a]*archetype_distribution[a]
            else:
                attribute_distribution += Z.T[a]*archetype_distribution[a]

        
        plt.rcParams["figure.figsize"] = (10,10)
        if type in ["OAA","RBOAA"]:
            attribute_distribution *=p
        fig, ax = plt.subplots()
        ax.set_ylabel('Value')
        plt.xlabel('Attributes')
        if title == "":
            ax.set_title(f"Value-Distribution of Archetype")
        else:
            ax.set_title(title)
        ax.bar(np.arange(len(attribute_distribution)),attribute_distribution)
        ax.set_xticks(np.arange(len(attribute_distribution)))
        ax.set_xticklabels(labels=columns)
        plt.ylim(0, p+0.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.set_size_inches(10, 10)
        
        if not save_fig:
            plt.show()
        else:
            plt.savefig("{0}.png".format(filename),dpi=300)
        
    
    def _circular_typal_barplot(self, type, Z, types, archetype_number,columns,p, save_fig, filename, title):

        archetype = Z.T[archetype_number-1]
        if type in ["OAA","RBOAA"]:
            archetype *=p

        type_values = []
        type_names = types.keys()
        for type in type_names:
            type_value = 0
            for attribute in types[type]:
                type_value += archetype[attribute-1]
            type_values.append(type_value/len(types[type]))
        
        
        archetype = type_values

        ANGLES = np.linspace(0.05, 2 * np.pi - 0.05, len(archetype), endpoint=False)
        width = 1/(len(archetype)/6)

        fig, ax = plt.subplots(figsize=(9, 12.6), subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#dae5eb")

        ax.set_theta_offset(1.2 * np.pi / 2)
        ax.set_ylim(0, p)

        ax.bar(ANGLES, archetype, alpha=0.9, width=width, zorder=10)
        ax.vlines(ANGLES, 0, p, ls=(0, (4, 4)), zorder=11)

        ax.set_xticks(ANGLES)
        ax.set_xticklabels(type_names, size=12)

        if title == "":
            ax.set_title("Circular Typal Barplot of Archetype {0}".format(archetype_number),size = 25)
        else:
            ax.set_title(title,size = 25)

        if not save_fig:
            plt.show()
        else:
            plt.savefig("{0}.png".format(filename),dpi=300)