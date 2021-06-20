from priorityq import PriorityQueue
import networkx as nx
import random
import numpy as np
import itertools as it
from joblib import Parallel, delayed
import math
import operator


def load_graph():
    Data = open('musae_facebook_edges.csv', "r")
    next(Data, None)  # skip the first line in the input file
    Graphtype = nx.Graph()
    G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                          nodetype=str)
    return G

def affiliationG(n, m, q, c, p, s):
    G = nx.Graph()
    community=dict() #It keeps for each community the nodes that it contains
    for i in range(m):
        community[i]=set()
    comm_inv=dict() #It keeps for each node the communities to which is affiliated
    for i in range(n):
        comm_inv[i]=set()
    # Keeps each node as many times as the number of communities in which is contained
    # It serves for the preferential affiliation to communities (view below)
    communities=[]
    #Keeps each node as many times as its degree
    #It serves for the preferential attachment of weak ties (view below)
    nodes=[]

    for i in range(n):
        # Preferential Affiliation to communities
        r=random.random()
        # Preferential Affiliation is done only with probability q (view else block).
        # With remaining probability (or if i is the first node to be processed and thus preferential attachment is not possible),
        # i is affiliated to at most c randomly chosen communities
        if len(communities) == 0 or r > q:
            num_com=random.randint(1,c) #number of communities is chosen at random among 1 and c
            for k in range(num_com):
                comm=random.choice([x for x in range(m)])
                community[comm].add(i)
                if comm not in comm_inv[i]:
                    comm_inv[i].add(comm)
                    communities.append(i)
        else:
            #Here, we make preferential affiliation: a node is chosen proportionally to the number of communities in which it is contained
            #and is copied (i.e., i is affilated to the same communities containing the chosen node).
            #Observe that the probability that i is affilated to a given community increases when the number of nodes in that community is large,
            #since the probability of selecting a node from a large community is larger than from a small community
            prot=random.choice(communities) #Choose a prototype to copy
            for comm in comm_inv[prot]:
                community[comm].add(i)
                if comm not in comm_inv[i]:
                    comm_inv[i].add(comm)
                    communities.append(i)

        # Strong ties (edge within communities)
        # For each community and each node within that community we add an edge with probability p
        for comm in comm_inv[i]:
            for j in community[comm]:
                if j != i and not G.has_edge(i,j):
                    r=random.random()
                    if r <= p:
                        G.add_edge(i,j)
                        nodes.append(i)
                        nodes.append(j)

        # Preferential Attachment of weak ties
        # We choose s nodes with a probability that is proportional to their degree and we add an edge to these nodes
        if len(nodes) == 0:  #if i is the first node to be processed (and thus preferential attachment is impossible), then the s neighbors are selected at random
            for k in range(s):
                v = random.choice([x for x in range(n) if x!=i])
                if not G.has_edge(i,v):
                    G.add_edge(i,v)
                    nodes.append(i)
                    nodes.append(v)
        else:
            for k in range(s):
                v = random.choice(nodes)
                if not G.has_edge(i,v):
                    G.add_edge(i,v)
                    nodes.append(i)
                    nodes.append(v)

    return G


def preference_by_dist(p, x_u):
    diz = dict()
    for candidate in range(len(p)):
        diz[candidate] = abs(p[candidate] - x_u)

    sort_diz = sorted(diz.items(), key=lambda x: x[1], reverse=False)
    preferences = list()

    for el in sort_diz:
        preferences.append(el[0])

    return preferences


def set_attributes(G, b, S):
    attrs = dict()
    bs = dict()
    for node in G.nodes():
        bs["belief"] = b[int(node)]
        if node not in S:
            bs["stub"] = 0.5
        else:
            bs["stub"] = 1
        attrs[node] = bs
        bs = dict()
    nx.set_node_attributes(G, attrs)


def FJ_dynamics(graph, max_iter):
    ts = 1
    x_u = dict()
    x_u_prec = dict()
    for v in graph.nodes():
        x_u[v] = round(graph.nodes[v]["belief"], 5)
    while True:
        x_u_prec = x_u.copy()
        for node in graph.nodes():
            s_u = graph.nodes[node]['stub']
            neigh = graph.degree(node)
            x_u[node] = round(
                s_u * graph.nodes[node]['belief'] + (1 - s_u) * (1 / neigh) * sum(x_u_prec[t] for t in graph[node]), 5)
        if check_update(x_u_prec, x_u):
            break
        if ts == max_iter:
            print("Numero massimo di iterazioni raggiunto")
            break
        ts += 1

    return x_u, ts


def check_update(dict_prec, dict_succ):
    flag = 0
    for el in dict_prec.keys():
        if abs(dict_succ[el] - dict_prec[el]) > 0.00001:
            flag = 1
            break
    if flag == 1:
        return False
    else:
        return True


def randomG(n, p):
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            r = random.random()
            if r <= p:
                G.add_edge(i, j)
    return G

def shapley_degree(graph):
    SV=dict()
    for v in graph.nodes():
        SV[v]=1/(1+graph.degree(v))
        for u in graph[v]:
            SV[v]+=1/(1+graph.degree(u))
    return SV

def shapley_threshold(graph,k):
    SV = dict()
    for v in graph.nodes():
        SV[v] =min(1,k / (1 + graph.degree(v)))
        for u in graph[v]:
            SV[v] += max(0,(graph.degree(u)-k+1)/(graph.degree(u)*(1+graph.degree(u))))
    return SV

def chunks(data,size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

def top(G,measure,k):
    pq = PriorityQueue()
    cen=measure(G)
    for u in G.nodes():
        pq.add(u, -cen[u])
    out=[]
    for i in range(k):
        out.append(pq.pop())
    return out

def degree(G):
    cen=dict()
    for u in G.nodes():
        cen[u] = G.degree(u)
    return cen


def closeness_par(G,sample=None):
    cen=dict()
    if sample is None:
        sample = G.nodes()
    subgraph=nx.subgraph(G,sample)
    for u in sample:
        visited=set()
        visited.add(u)
        queue=[u]
        dist=dict()
        dist[u]=0

        while len(queue) > 0:
            v = queue.pop(0)
            for w in subgraph[v]:
                if w not in visited:
                    visited.add(w)
                    queue.append(w)
                    dist[w] = dist[v]+1#contiene per ogni nodo la lunghezza del path minimo da esso alla radice

        cen[u]=sum(dist.values())
    return cen


def top_parallel(G,k,j):
    pq = PriorityQueue()
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        result=parallel(delayed(closeness_par)(G,X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))

    for u in result:#u is a dict
        for el in u.keys():
            pq.add(el, -u[el])
      # We use negative value because PriorityQueue returns first values whose priority value is lower
    out=[]
    for i in range(k):
        out.append(pq.pop())
    return out



def betweenness_par(G,sample=None):

    if sample is None:
        sample=G.nodes()
    subgraph=nx.subgraph(G,sample)
    edge_btw={frozenset(e):0 for e in subgraph.edges()}
    node_btw={i:0 for i in subgraph.nodes()}

    for s in subgraph.nodes():
        # Compute the number of shortest paths from s to every other node
        tree = []  # it lists the nodes in the order in which they are visited
        spnum = {i: 0 for i in subgraph.nodes()}  # it saves the number of shortest paths from s to i
        parents = {i: [] for i in subgraph.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
        distance = {i: -1 for i in subgraph.nodes()}  # the number of shortest paths starting from s that use the edge e
        eflow = {frozenset(e): 0 for e in subgraph.edges()}  # the number of shortest paths starting from s that use the edge e
        vflow = {i: 1 for i in subgraph.nodes()} #the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        #BFS
        queue=[s]
        spnum[s]=1
        distance[s]=0
        while queue != []:
            c=queue.pop(0)
            tree.append(c)
            for i in subgraph[c]:
                if distance[i] == -1: #if vertex i has not been visited
                    queue.append(i)
                    distance[i]=distance[c]+1
                if distance[i] == distance[c]+1: #if we have just found another shortest path from s to i
                    spnum[i]+=spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while tree != []:
            c=tree.pop()
            for i in parents[c]:
                eflow[frozenset({c,i})]+=vflow[c] * (spnum[i]/spnum[c]) #the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i]+=eflow[frozenset({c,i})] #each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                edge_btw[frozenset({c,i})]+=eflow[frozenset({c,i})] #betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c]+=vflow[c] #betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex

    return edge_btw,node_btw

def top_betweenness(G,k,j):
    #PARALLELIZZAZIONE
    pq=PriorityQueue()
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        lista=parallel(delayed(betweenness_par)(G,X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
        #Aggregates the results

    for j in lista:
        for i in j[1].keys():
            pq.add(i,-j[1][i])#Fase di assemblaggio

    out=[]
    for i in range(k):
        out.append(pq.pop())
    return out


def rank(graph,d=0.85,n_iterations=50):

    V = graph.number_of_nodes()  #is the number of nodes of the graph
    ranks = dict()#dict of ranks
    for node in graph.nodes():
        ranks[node] = 1/V

    for _ in range(n_iterations):
        for el in graph.nodes():
            rank_sum = 0
            curr_rank = ranks[el]

            for n in graph.neighbors(el):
                if ranks[n] is not None:
                    outlinks = len(list(graph.neighbors(n)))
                    rank_sum += (1 / float(outlinks)) * ranks[n]

            ranks[el] = ((1 - float(d)) * (1/float(V))) + d*rank_sum

    return ranks


def top_rank(k,rank):
    pq = PriorityQueue()
    for u in rank.keys():
        pq.add(u, -rank[u])
    out=[]
    for i in range(k):
        out.append(pq.pop())
    return out
#HITS
def hits(graph,k):

    auth = dict()
    hubs= dict()
    for node in graph.nodes():
        auth[node] = 1
        hubs[node] = 1
    for i in range(k):  #We perform a sequence of k hub-authority updates
        for node in graph.nodes():
            auth[node] =sum(hubs[el] for el in graph[node])#First apply the Authority Update Rule to the current set of scores.
        for node in graph.nodes():
            hubs[node] =sum(auth[el] for el in graph[node])#Then apply the Hub Update Rule to the resulting set of scores.

    auth_n,hubs_n=normalize_naive(graph,auth,hubs)
    return auth_n,hubs_n

def normalize_naive(G,auth,hubs):
    auth_sum = sum(auth[node] for node in G.nodes())
    hub_sum = sum(hubs[node] for node in G.nodes())

    for node in G.nodes():
        auth[node] =auth[node]/auth_sum
        hubs[node] =hubs[node]/hub_sum
    return auth,hubs

def top_hits(G,k,num_node):
    pq = PriorityQueue()
    pq2=PriorityQueue()
    auth_n,hubs_n=hits(G,k)
    for u in G.nodes():
        pq.add(u, -auth_n[u])  # We use negative value because PriorityQueue returns first values whose priority value is lower
    for u in G.nodes():
        pq2.add(u, -hubs_n[u])  # We use negative value because PriorityQueue returns first values whose priority value is lower
    out=[]
    out2=[]
    for i in range(num_node):
        out.append(pq.pop())
        out2.append(pq2.pop())
    return out,out2

def manipulation(G,p,c,B,b):

    #--------------Elezione non manipolata-------------
    sum_votes_c1=0
    for el in b:
        lista_pref=preference_by_dist(p, el)
        if lista_pref[0]==c:
            sum_votes_c1+=1

    print("Group number 2; number of votes for candidate c before the manipulation: "+str(sum_votes_c1))

    #----------------Elezione manipolata---------------

    #Le centralità del final project mi ritornano una dizionario
    #diction=shapley_degree(G)
    #sort_orders = sorted(diction.items(), key=lambda x: x[1], reverse=True)
    #S=list()

    #Le centralità del mid-term ritornano una lista
    #S=top(G,degree,B) #Facciamo ritornare in una lista i primi B nodi secondo la centralità espressa come secondo parametro
    #S=top_parallel(G, B, 40)
    #S=top_betweenness(G, B, 50)
    #r = rank(G,0.85,50)
    #S = top_rank(B, r)
    S, h = top_hits(G, 30, B)


    #Dopo aver ordinato tutti i nodi dl grafo in ordine decrescente di shapley_degree
    #prendiamo gli elementi con maggiore shapley degree limitatamente al budget B
    count=0
    '''
    for el in sort_orders:
        if(count<=B):
           S.append(el[0])
           count+=1
        else:
            break
    '''
    #Dopo aver selezionato i seed andiamo ad abbinare a tali seed i valori di belief
    for u in range(len(S)):
        b[int(S[u])]=p[c]#vado a modificare il picco del nodo influencer in base al partito del candidato c


    #Aggiornamento degli attributi nel grafo dei votanti, considerando anche i seed.
    set_attributes(G,b,S)
    x_u,ts2 = FJ_dynamics(G,2500)
    sum_votes_c=0
    for el in x_u.keys():
        lista_pref=preference_by_dist(p, x_u[el])#passo il picco di ogni votante (x_u)
        #print("Preferenza votante con manipolazione " + str(el) +" lista: "+str(lista_pref))
        if lista_pref[0]==c:#se il voto spetta al candidato c
            sum_votes_c+=1

    print("Number of votes for candidate c after the manipulation: "+str(sum_votes_c)+" convergence after "+str(ts2)+" time step.")
    print("Difference of votes between two elections: ",str(sum_votes_c-sum_votes_c1))

def main():
    # print("Inserisci il numero dei candidati politici:")
    # m=int(input())
    m = 1000
    '''print("Inserisci il numero dei votanti:")
    n = int(input())'''
    #n = 22470
    # print("Inserisci budget:")
    # B=int(input())
    B = 200
    p = list()
    for i in range(m):
        rand = round(random.random(), 5)
        p.append(rand)
    # print("Inserisci il numero del candidato speciale:")
    # c = int(input())
    c = 2
    while (c < 0 or c > len(p) - 1):
        print("Inserisci il numero del candidato speciale:")
        c = int(input())
    b = list()

    # G=load_graph()
    n = 10000
    #G = randomG(n, 0.1)
    G=affiliationG(n, 4, 0.5, 3, 0.8, 2)
    for i in range(len(G.nodes())):
        rand = round(random.random(), 5)
        b.append(rand)
    print("Politici:", p)
    print("Votanti:", b)
    manipulation(G, p, c, B, b)
if __name__ == "__main__":
     main()