import csv
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import numpy as np
from scipy.special import zeta

def read_graph():
    G=nx.Graph()
    with open('net_2') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter='\b')
        for row in csv_reader:
            a,b=row[0].split(' ')
            G.add_edge(a,b)

    return G


def nodo_con_grado_max(graph):
    max=0
    for a in graph.nodes():
        deg=graph.degree(a)
        if deg >max:
            max=deg
    return max

def check_if_power_law(graph,grado_max,r_start, k_start, q_start):
    distribution=dict()#In distribution[0] è presente il numero di nodi che ha grado 0
                        #In distribution[1] è presente il numero di nodi che ha grado 1

    #Initialize dict
    for i in range(0,grado_max+1) :
        distribution[i]=0

    for a in graph.nodes():
        deg = graph.degree(a)
        distribution[deg]+=1

    return (distribution,r_start,k_start,q_start)


def check_if_power_law_s(graph,grado_max):
    distribution=dict()

    #Initialize dict
    for i in range(0,grado_max+1) :
        distribution[i]=0

    for a in graph.nodes():
        deg = graph.degree(a)
        distribution[deg]+=1

    return distribution



def check_if_power_law_analysis(graph,grado_max):
    distribution=dict()#In distribution[0] è presente il numero di nodi che ha grado 0
                        #In distribution[1] è presente il numero di nodi che ha grado 1

    #Initialize dict
    for i in range(0,grado_max+1) :
        distribution[i]=0

    for a in graph.nodes():
        deg = graph.degree(a)
        distribution[deg]+=1

    lista_x=list(distribution.keys())
    lista_y=list(distribution.values())
    plt.plot(lista_x,lista_y)
    plt.show()
def configurationG(deg):
    G = nx.Graph()
    # The following list will contain all nodes for which there is at least another neighbor to add
    nodes=list(range(len(deg)))
    # We consider '> 1' and not '>0' because when len(nodes) = 1 only loops are possible
    # Hence, the degree sequence of the resulting graph may not be exactly the same as the one
    #in input for the single remaining node.
    # However, note that this single "outlier" does not alter the degree sequence distribution.
    while len(nodes)>1:
        edge=random.sample(nodes,2)#ritorna una lista di due elementi casuali presi dalla lista nodes
        if not G.has_edge(edge[0],edge[1]):
            G.add_edge(edge[0],edge[1])
            deg[edge[0]]-=1
            if deg[edge[0]] == 0:
                nodes.remove(edge[0])
            deg[edge[1]]-=1
            if deg[edge[1]] == 0:
                nodes.remove(edge[1])
    return G

def convert_into_gaussian_for_configurationG(distribution):
    cello=list()
    for key in distribution.keys():
      i=distribution[key]
      for j in range(i):
        cello.append(key)
    return cello

def degreelist(G):
    degree_list=[]
    for node in G.nodes():
        degree_list.append(G.degree(node))
    return degree_list

G=read_graph()
distribution=check_if_power_law_s(G,nodo_con_grado_max(G))
ipotetic_gaussian=convert_into_gaussian_for_configurationG(distribution)
print(len(ipotetic_gaussian))
print(ipotetic_gaussian)
Grafo=configurationG(ipotetic_gaussian)
print("fine della generazione del grafo")
n=Grafo.number_of_nodes()
m=Grafo.number_of_edges()
cc=nx.number_connected_components(Grafo)
clustering_C=nx.algorithms.average_clustering(Grafo)
#distribution=check_if_power_law_s(Grafo,nodo_con_grado_max(Grafo))
print("Numero di nodi",str(n))
print("Numero di archi",str(m))
print("Numero di componenti connesse",str(cc))
print("Coefficiente di clustering medio",str(clustering_C))