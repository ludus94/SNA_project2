import networkx as nx
import random
import numpy as np
from priorityq import PriorityQueue

def shapley_degree_ip(graph):
    SV=dict()
    for v in graph.nodes():
        SV[v]=1/(1+graph.degree(v))
        for u in graph[v]:
            SV[v]+=1/(1+graph.degree(u))
    return SV

def preference_by_dist(G,p,u):#LE PREFERENZE LE METTO IN ORDINE DI DISTANZA DAL MIO PICCO
    #ipotizzo che un nodo abbia come attributi xu(t) e si
    x_u=G.nodes[u]["belief"]

    lista=PriorityQueue()

    for candidate in range(len(p)):
        lista.add(candidate,p[candidate]-x_u)#abbiamo previsto che in base alla distanza tra il picco del nodo u e il picco di ogni candidato p, io inserisca il candidato che ha come indice p con una certa priorità nella pq

    preferences=list()
    for el in range(len(p)):
        preferences.append(lista.pop())
    return preferences#preferences ritorna l'indice dei candidati in ordine di prefererenza rispetto a p_i

def randomG(n, p):
    G = nx.Graph()
    for i in range(n):
        for j in range (i+1, n):
            r=random.random()
            if r <= p:
                G.add_edge(i,j)
    return G

def set_attributes(G,b,S):
    attrs = dict()#dizionario globale di tutti i nodi
    bs = dict()
    for node in G.nodes():
        bs["belief"] = b[node]
        if node not in S:
            bs["stub"] = 0.5
        else:
            bs["stub"] = 1
        attrs[node] = bs
        bs = dict() #inizializzo per un altro nodo
    #print(attrs)
    nx.set_node_attributes(G, attrs)#assegno il dizionario globale al grafo G

def check_dict(dict_prec,dict_succ):
    list_prec=list(dict_prec.values())
    list_succ=list(dict_succ.values())
    list_prec.sort()
    list_succ.sort()
    if list_prec==list_succ:
        return True
    return False

def FJ_dynamics(graph):
   # set_attributes(graph)
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

def manipulation(G,p,c,B,b):

    #ricavare S
    set_attributes(G,b,set())#elezione senza manipulation
    x_u=FJ_dynamics(G)
    sum_votes_c=0
    for el in x_u.keys():
        lista_pref=preference_by_dist(G, p, el)#passo il picco di ogni votante (x_u)
        if(lista_pref[0]==c):#se il voto spetta al canddidato c
            sum_votes_c+=1

    print("Group number 2; number of votes for candidate c before the manipulation:"+str(sum_votes_c)+";")
    #Flag=True
    #while Flag:
    #dobbiamo fare un algoritmo che setta i seed (max B) [scegliere un algoritmo di centralità e selezionare i primi B elementi ] e scegliere i loro peak [pari al partito cui c appartiene, p_c]
    diction=shapley_degree_ip(G)
    sort_orders = sorted(diction.items(), key=lambda x: x[1], reverse=True)
    S=list()
    price=0
    peak_s = list()
    for el in  sort_orders.keys():
        if(price<=B):
           S.append(el)
           price+=sort_orders[el]
           peak_s.append(p[c])
    for u in range(len(S)):
        G[S[u]]['belief'] = peak_s[u]

    set_attributes(G,b,S)  # elezione con manipulation
    x_u = FJ_dynamics(G)
    sum_votes_c = 0

    #return



def main():
    print("Inserisci il numero dei candidati politici:")
    m=int(input())
    print("Inserisci il numero dei votanti:")
    n = int(input())
    print("Inserisci budget:")
    B=int(input())
    p=list()
    for i in range(m):#Generiamo le pi che vanno da 1 a m nel range [0,1] indicavano la corrente politica del candidato
        rand = round(random.random(), 5)
        p.append(rand)
    # b is a Python list such that len(b) = len(G.nodes())
    print("Inserisci il numero del candidato speciale:")
    c = int(input())
    while(c<0 or c>len(p)-1):
        print("Inserisci il numero del candidato speciale:")
        c = int(input())
    b=list()
    G=randomG(n,0.45)#n,p
    for i in range(len(G.nodes())):
        rand = round(random.random(), 5)#random [0,1] per settare l' attributo bu di ogni votante
        b.append(rand)

    set_s,peak_s=manipulation(G, p, c, B, b)#ritorna i seed e i picchi per cui  il delta è massimo

if __name__ == "__main__":
     main()
