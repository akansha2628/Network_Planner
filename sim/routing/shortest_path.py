import heapq
from sim.core import topology as Topology
from sim.core.constants import *

def dijkstra_shortest_path(topology, src, dest):
    """Return (distance, path) using Dijkstra on weighted adjacency matrix."""
    N = len(topology)
    dist = [float('inf')] * N
    prev = [-1] * N
    dist[src] = 0
    pq = [(0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == dest:
            break
        if d > dist[u]:
            continue
        for v in range(N):
            if topology[u][v] > 0:  # there is a link
                alt = d + topology[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, v))

    if dist[dest] == float('inf'):
        return float('inf'), []
    # reconstruct path
    path = []
    v = dest
    while v != -1:
        path.append(v)
        v = prev[v]
    path.reverse()
    return dist[dest], path


def yens_k_shortest_paths(topology, src, dest, K=3):
    """Return up to K shortest paths using Yen’s algorithm on modified topology."""
    dist, path = dijkstra_shortest_path(topology, src, dest)
    if not path:
        return []
    A = [path]  # shortest paths found
    B = []  # potential candidates

    for k in range(1, K):
        for i in range(len(A[-1]) - 1):
            spur_node = A[-1][i]
            root_path = A[-1][:i + 1]

            # Copy topology for modification
            topology_copy = [row[:] for row in topology]

            # Remove edges that would create loops
            for p in A:
                if len(p) > i and p[:i + 1] == root_path:
                    u = p[i]
                    v = p[i + 1]
                    topology_copy[u][v] = 0
                    topology_copy[v][u] = 0

            # Remove nodes in root path (except spur node)
            for node in root_path[:-1]:
                for v in range(len(topology_copy)):
                    topology_copy[node][v] = 0
                    topology_copy[v][node] = 0

            dist_spur, spur_path = dijkstra_shortest_path(topology_copy, spur_node, dest)
            if spur_path:
                total_path = root_path[:-1] + spur_path
                if total_path not in A:
                    total_dist = 0
                    for j in range(len(total_path) - 1):
                        total_dist += topology[total_path[j]][total_path[j + 1]]
                    heapq.heappush(B, (total_dist, total_path))

        if not B:
            break
        dist_k, path_k = heapq.heappop(B)
        A.append(path_k)
    return A

def build_k_shortest_paths(topology):
    """
    Precompute K-shortest paths for all src-dst pairs.
    Returns a 3D list: PATHS[src][dst] = list of K paths (each path = list of node IDs)
    """
    num_nodes = len(topology)
    PATHS = [[[] for _ in range(num_nodes)] for _ in range(num_nodes)]
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src != dst:
                PATHS[src][dst] = yens_k_shortest_paths(topology, src, dst, k_paths)
    # print("K-shortest paths precomputed for all source-destination pairs.")
    return PATHS