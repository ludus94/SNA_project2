"""
Generiamo le pi che vanno da 1 a m nel range [0,1] indicavano la
 corrente politica del candidato
Generiamo la rete G dei votanti che vanno da 1 a n

    random [0,1] per settare l' attributo bu di ogni votante
    creo una list  b  such that len(b) = len(G.nodes())

    for u in G.nodes()#e G is an undirected, unweighted graph

  function (G,m,[p1,...,pm],c,B,b) c è il candidato da far eleggere  c is in {0, …, len(p)-1}
#nd b is a Python list such that len(b) = len(G.nodes())
 if u is not in S:
         S[u]=1/2
        else:
          S[u]=1
        node.setAttribute(b[u],s[u])
    while(not(convergenza(x(t)))):
         if (time=0)
           ricavare il set S(influencers) tra i nodi di G, tali che il loro numero sia al massimo B
           settare il loro valore bu in maniera tale da massimizzare la differenza tra i voti di c prima dell'influenza e dopo .
            (un possibile modo è:
            for u in S:
               b[u]=p[c] (picco del candidato)
               )
        x(t)=Run FJ
    if convergenza(x(t)):
        for u in G.nodes():
            p[u]=x_t[u] ????????????????????????
            #calcolo della ranked list di preferenza per ogni elettore
            preference_by_dist(u)

--------
dobbimo ricavare:n the number of votes obtained by the candidate c
questa cosa la possiamo ricavare prendendo xu prima e dopo la convergenza di ogni votante e:
su ogni xu, che rappresenta un peak di preferenza politica, mi devo ricavare la ranked list di preferenza, dove, i p_u più vicini a xu sono all'inizio e più preferiti, e se ho 2 candidati equidistanti, do preferenza a quello sulla sinistra
"""