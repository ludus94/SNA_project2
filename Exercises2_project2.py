import csv
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import numpy

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





# Generalized Watts-Strogatz (EK 20)
# n is the number of nodes (we assume n is a perfect square or it will be rounded to the closest perfect square)
# r is the radius of each node (a node u is connected with each other node at distance at most r) - strong ties
# k is the number of random edges for each node u - weak ties
#
# Here, the weak ties are still proportional to distance
# q is a term that evaluate how much the distance matters.
# Specifically, the probability of an edge between u and v is proportional to 1/dist(u,v)**q
# Hence, q=0 means that weak ties are placed uniformly at random, q=infinity only place weak ties towards neighbors.
#
# Next implementation of Watts-Strogatz graphs assumes that nodes are on a two-dimensional space (similar implementation can be given on larger dimensions).
# Here, distance between nodes will be set to be the Euclidean distance.
# This approach allows us a more fine-grained and realistic placing of nodes (i.e., they not need to be all at same distance as in the grid)
def GenWS2DG(n, r, k, q):
    G = nx.Graph()

    # We assume that the 2D area is sqrt(n) x sqrt(n) for sake of comparison with the grid implementation.
    # Anyway, one may consider a larger or a smaller area.
    # However, recall that the radius r given in input must be in the same order of magnitude as the size of the area
    # (e.g., you cannot consider the area as being a unit square, and consider a radius 2, otherwise there will be an edge between each pair of nodes)
    line=int(math.sqrt(n))
    nodes=dict() #This will be used to associate to each node its coordinates
    prob=dict() #Keeps for each pair of nodes (u,v) the term 1/dist(u,v)**q

    # The following for loop creates n nodes and place them randomly in the 2D area.
    # If one want to consider a different placement, e.g., for modeling communities, one only need to change this part.
    for i in range(n):
        x=random.random()
        y=random.random()
        nodes[i]=(x*line,y*line)
        prob[i]=dict()

    for i in range(n):
        # Strong-ties
        for j in range(i+1,n): #we add edge only towards next nodes, since edge to previous nodes have been added when these nodes have been processed
            dist=math.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2) #Euclidean Distance
            prob[i][j]=1/(dist**q)
            prob[j][i]=prob[i][j]
            if dist <= r:
                G.add_edge(i,j)

        # Terms 1/dist(u,v)**q are not probabilities since their sum can be different from 1.
        # To translate them in probabilities we normalize them, i.e, we divide each of them for their sum
        norm=sum(prob[i].values())
        # Weak ties
        # They are not exactly h, since the random choice can return a node s such that edge (i, s) already exists
        for h in range(k):
            # Next instruction allows to choice from the list given as first argument according to the probability distribution given as second argument
            s=numpy.random.choice([x for x in range(n) if x != i],p=[prob[i][x]/norm for x in range(n) if x!= i])
            G.add_edge(i,s)

    return G




def variazione_parametri_watts(n, r_start, k_start, q_start,n_iter,passo):

    lista=list()
    for i in range(n_iter):
        Grafo=GenWS2DG(n, r_start, k_start, q_start)
        print("Parametri n: "+str(n)+" r: "+str(r_start)+" k:"+str(k_start)+" q:"+str(q_start))
        print("Average clustering coefficient:",nx.algorithms.average_clustering(Grafo))
        print("Numero di componenti connesse:",nx.number_connected_components(Grafo))
        #print("Diametro della rete",nx.diameter(Grafo))
        lista.append(check_if_power_law(Grafo,nodo_con_grado_max(Grafo),r_start, k_start, q_start))
        r_start=r_start+passo
        k_start=k_start+passo
        q_start=q_start+passo

    fig,axs = plt.subplots(n_iter)
    count=0

    for el in lista:
        axs[count].set(xlabel="Parameters r:"+str(el[1])+" k:"+str(el[2])+" q:"+str(el[3]),ylabel="ma")
        axs[count].plot(el[0].keys(),el[0].values())
        count+=1
    plt.show()






variazione_parametri_watts(100,4,3,2,2,1)
#G=read_graph()
#check_if_power_law(G,nodo_con_grado_max(G))#check if the distribution of degree is a power law
#Gcc = sorted(nx.connected_components(G), key=len, reverse=True)#computation of the giant component
#G0 = G.subgraph(Gcc[0])
#now we calculate the number of nodes of the giant component with respect to G
#print("Number of connect commponent:",G0.number_of_nodes()/G.number_of_nodes())
#print("Avarage clustering coefficient:",nx.algorithms.average_clustering(G))
#print("Diametro della rete",nx.diameter(G)) #Dall' esecuzione si è visto che il dimatro è pari a 4
#print("Radius :",nx.algorithms.radius(G)) #il radius è pari a 4
#--------Insert a model experiments---------
#G1=randomG(G.number_of_nodes(),0.2)
#check_if_power_law(G1,nodo_con_grado_max(G1))
#print("Avarage clustering coefficient:",nx.algorithms.average_clustering(G1))
