import random
import math
import networkx as nx

def load_graph():
    Data = open('musae_facebook_edges.csv', "r")
    next(Data, None)  # skip the first line in the input file
    Graphtype = nx.Graph()
    G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                      nodetype=str)
    return G

def shapley_degree_ip(graph):
    SV=dict()
    for v in graph.nodes():
        SV[v]=1/(1+graph.degree(v))
        for u in graph[v]:
            SV[v]+=1/(1+graph.degree(u))
    return SV

def shapley_threshold_ip(graph,k):
    SV = dict()
    for v in graph.nodes():
        SV[v] =min(1,k / (1 + graph.degree(v)))
        for u in graph[v]:
            SV[v] += max(0,(graph.degree(u)-k+1)/(graph.degree(u)*(1+graph.degree(u))))
    return SV

def shapley_closeness_ip(graph):
    SV = dict()
    for v in graph.nodes():
        SV[v]=0
    for v in graph.nodes():
        d,w = nx.single_source_dijkstra(graph, v)
        '''end_p=list()
        for curr in w:
            end_p.append(curr(len(curr)-1))#sto prelevando da ogni path di dijlstra l'endpoint'''
        sum=0
        prev_dist=-1
        prev_sv=-1
        index=len(graph.nodes())-1
        d_key=list()
        d_value=list()
        for key in d.keys():
            d_key.append(key)
        for value in d.values():
            d_value.append(value)
        #print(d_key)
        #print(type(d_value))
        while index>0:
            #print(index)
            #print(d_value[index])
            if d_value[index] == prev_dist:
                curr_sv=prev_sv
            else:
                curr_sv=(d_value[index]/(1+index))-sum
            SV[d_key[index]]+=curr_sv
            sum+=(d_value[index])/(index*(1+index))
            prev_dist=d_value[index]
            prev_sv=curr_sv
            index-=1
        SV[v]-=sum
    return SV


def set_attributes(G):
    attrs = dict()
    bs = dict()

    for node in G.nodes():
        bs["belief"] = round(random.random(), 5)
        bs["stub"] = round(random.random(), 5)
        attrs[node] = bs
        bs = dict()
    #print(attrs)
    nx.set_node_attributes(G, attrs)

    # Stampa
    '''for act in G.nodes():
        print("Nodo:",act)
        print("Attributi:",G.nodes[act]["belief"],G.nodes[act]["stub"])'''


def FJ_dynamics(graph):
    set_attributes(graph)
    ts=1
    x_u=dict()
    x_u_prec=dict()
    for v in graph.nodes():#instant 0
        x_u[v]=round(graph.nodes[v]["belief"],5)
    while not check_dict(x_u_prec,x_u):#condizione
        x_u_prec=x_u.copy()
        for node in graph.nodes():
            s_u=graph.nodes[node]['stub']
            neigh=graph.degree(node)
            x_u[node]=round(s_u*graph.nodes[node]['belief']+(1-s_u)*(1/neigh)*sum(x_u_prec[t] for t in graph[node]),5)

        ts+=1
    print("Ts ",ts)
    return x_u


def check_dict(dict_prec,dict_succ):
    list_prec=list(dict_prec.values())
    list_succ=list(dict_succ.values())
    list_prec.sort()
    list_succ.sort()
    if list_prec==list_succ:
        return True
    return False
"""
G=nx.Graph()
G.add_edge('A', 'B')
G.add_edge('A', 'C')
G.add_edge('B', 'C')
G.add_edge('B', 'D')
G.add_edge('D', 'E')
G.add_edge('D', 'F')
G.add_edge('D', 'G')
G.add_edge('E', 'F')
G.add_edge('F', 'G')
"""
G=load_graph()


G_W=nx.Graph()
G_W.add_edge('A', 'B',weight=1)
G_W.add_edge('A', 'C',weight=2)
G_W.add_edge('B', 'C',weight=3)
G_W.add_edge('B', 'D',weight=4)
G_W.add_edge('D', 'E',weight=5)
G_W.add_edge('D', 'F',weight=6)
G_W.add_edge('D', 'G',weight=7)
G_W.add_edge('E', 'F',weight=8)
G_W.add_edge('F', 'G',weight=9)

print("Shapley degree:")
print(shapley_degree_ip(G))
print("Shapley threshold:")
print(shapley_threshold_ip(G,2))
print("Shapley closeness:")
#print(shapley_closeness_ip(G_W))
print("FJ dynamics:")
#print(FJ_dynamics(G))
