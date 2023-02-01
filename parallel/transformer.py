# global variables to make work with variables in functions easier
map_size = 0
node_map = dict()
nodes = dict()

def get_node_index(node):
    global map_size
    if node not in node_map:
        val = map_size
        node_map[node] = val
        map_size += 1
        return val
    return node_map[node]
    
def add_pair(node1, node2):
    if node1 == node2:
        return

    if node1 not in nodes:
        nodes[node1] = set()

    if node2 not in nodes:
        nodes[node2] = set()

    nodes[node1].add(node2)
    nodes[node2].add(node1)


def transform(load, store, delimiter='\t'):
    global map_size, node_map, nodes
    map_size = 0
    node_map = dict()
    nodes = dict()
    with open(load, 'r') as file:
        for line in file.readlines():
            if len(line.strip()) == 0 or line.startswith('#'):
                #empty newline 
                continue
            
            target, source = line.strip().split(delimiter)
            target = get_node_index(int(target))
            source = get_node_index(int(source))
            add_pair(target, source)

    # saves the transformed graph - vertices starting from 0, no single-directional edges
    with open(store, 'w') as file:
        for key in sorted(nodes):
            neighbors = sorted(nodes[key])
            for neighbor in neighbors:
                file.write(f'{key}  {neighbor}\n')

    # prints graph details
    max_deg = 0
    most_clusters = 0
    total_clusters = 0
    total_edges = 0
    for key in nodes:
        size = len(nodes[key])
        total_edges += size
        cluster_size = size * (size - 1) / 2
        most_clusters = max(most_clusters, cluster_size)
        total_clusters += cluster_size
        max_deg = max(max_deg, size)

    # each cluster is counted 3 times, so we rescale the count.
    total_clusters = total_clusters // 3
    print(f'{store}: {len(nodes)} vertices, {total_edges} edges, max degree {max_deg}, total clusters {total_clusters}, max clusters in a vertex: {most_clusters}\n')


if __name__ == '__main__':
    import sys
    src = sys.argv[1]
    dst = sys.argv[2]

    transform(src, dst, ' ')

# example usage:         
#transform('Amazon0302.txt', 'ready/Amazon0302.txt', ' ')
#transform('roadNet-CA.txt', 'ready/road-net-ca.txt')
#transform('fb_social_circles.txt', 'ready/fb-social-circles.txt', ' ')
#transform('soc-pokec-relationships.txt', 'ready/pokec.txt')
#transform('web-Google.txt', 'ready/google.txt')
