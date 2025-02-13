
from yoda.alignments import load_rfam
from yoda.graphs import ali2graph

a,l = load_rfam(full=True)
ali = a[1200]

graph = ali2graph.set_weight_label(ali)
graph = ali2graph.dillute(graph)
graph = ali2graph.set_weight(graph)
graph =   ali2graph.multiGraph(ali)
graph =  ali2graph.donest(graph)

def all():
    for ali in a:
        graph = ali2graph.set_weight_label(ali)
        graph = ali2graph.dillute(graph)
        graph = ali2graph.set_weight(graph)
        graph =   ali2graph.multiGraph(ali)
        graph =  ali2graph.donest(graph)


