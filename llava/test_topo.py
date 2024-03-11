
import networkx as nx


# 检查哈密顿路径是否可行
def is_hamiltonian_path(G, path):
    # 检查路径是否包含每个节点
    if set(path) != set(G.nodes):
        return False
    # 检查路径是否连续
    for i in range(len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return False
    return True
outputs = "0->2->4->1->5->3."
image_file = "/mnt/sdb1/NLGraph/NLGraph/hamilton/image/easy/standard/graph8_dot_ellipse_1.0_white_filled.png"
candidate_path = outputs.split(".")[0].split('->')
candidate_path = list(map(int, candidate_path))
graph_file = image_file.replace("image", "graph").split("_")[0] + '.txt'



# 创建无向图
G = nx.Graph()
with open(graph_file,"r") as f:
    n, m = [int(x) for x in next(f).split()]
    array = []
    for line in f:  # read rest of lines
        array.append([int(x) for x in line.split()])
    edges = array[:m]
    assert len(edges) == m
    G.add_nodes_from(range(n))
    for edge in edges:
        G.add_edge(edge[0], edge[1])
print(G, candidate_path)
if is_hamiltonian_path(G=G, path=candidate_path): print('true')
else: print('false')