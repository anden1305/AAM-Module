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

    def _PCA_scatter_plot(self,Z,X,type):
        
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
        plt.title(f"PCA Scatter Plot of {type}", fontsize = 20)
        plt.legend(prop={'size': 15})
        plt.show()


    def _attribute_scatter_plot(self,Z,X,attributes,type,p):
        
        plt.rcParams["figure.figsize"] = (10,10)
        plt.scatter(X[attributes[0]-1,:]*p, X[attributes[1]-1,:]*p, c ="black", s = 1)
        for a in range(len(Z[0,:])):
            plt.scatter(Z[attributes[0]-1,a]*p, Z[attributes[1]-1,a]*p, marker ="^", s = 500, label="Archetype {0}".format(a+1))
        plt.xlabel(f"Attribute {attributes[0]}", fontsize=15)
        plt.ylabel(f"Attribute {attributes[1]}", fontsize=15)
        plt.legend(prop={'size': 15})
        plt.title(f"Attribute Scatter Plot of {type}", fontsize = 20)
        plt.show()


    def _loss_plot(self,loss,type):
        plt.plot(loss, c="#2c6c8c")
        plt.xlabel(f"Iteration of {type}")
        plt.ylabel(f"Loss of {type}")
        plt.title(f"Loss w.r.t. Itteration of {type}")
        plt.show()


    def _mixture_plot(self,Z,A,type):

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
        plt.title(f"Mixture Plot of {type}", fontsize = 20)
        plt.scatter(points_x, points_y, c ="black", s = 3, zorder=5)
        plt.legend()
        plt.show()


    def _barplot(self,Z,columns,archetype_number,type,p):
        
        plt.rcParams["figure.figsize"] = (10,10)
        archetype = Z.T[archetype_number-1]
        if type in ["OAA","RBOAA"]:
            archetype *=p
        fig, ax = plt.subplots()
        ax.set_ylabel('Value')
        plt.xlabel('Attributes')
        ax.set_title(f"Value-Distribution of Archeype {archetype_number}")
        ax.bar(np.arange(len(archetype)),archetype)
        ax.set_xticks(np.arange(len(archetype)))
        ax.set_xticklabels(labels=columns)
        plt.ylim(0, p+0.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.set_size_inches(10, 10)
        plt.show()


    def _barplot_all(self,Z,columns,type, p):
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
        plt.title(f"Value-Distribution over All Archetypes")
        plt.show()


    def _typal_plot(self, Z, types, weighted):
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
                for i in types[label]:
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
        ax.set_title('Typal Composition of Archetypes')
        ax.legend()

        plt.show()
        

    def _pie_chart(self, A, indexes):

        data = []

        for i in range(len(indexes)):
            datapoint = A.T[indexes[i]]
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
        plt.title("Pie Chart of Archetype Distribution on Given Subset of Data")
        plt.show()
    

    def _attribute_distribution(self, A, Z, indexes, columns, p, type):

        archetype_distribution = []

        for i in range(len(indexes)):
            datapoint = A.T[indexes[i]]
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
        ax.set_title(f"Value-Distribution of Archeype")
        ax.bar(np.arange(len(attribute_distribution)),attribute_distribution)
        ax.set_xticks(np.arange(len(attribute_distribution)))
        ax.set_xticklabels(labels=columns)
        plt.ylim(0, p+0.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.set_size_inches(10, 10)
        plt.show()