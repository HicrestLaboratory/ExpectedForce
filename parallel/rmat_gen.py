import snap

N = 16
EF = 4

Rnd = snap.TRnd()
nodes = pow(2, N)
edges = nodes * EF
Graph = snap.GenRMat(nodes, edges, .6, .1, .15, Rnd)
for EI in Graph.Edges():
        print( EI.GetSrcNId(), EI.GetDstNId())
