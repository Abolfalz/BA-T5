from scipy.optimize import curve_fit
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

m = 4
G = nx.complete_graph(m)
n = 10**4
degree_distributions = []
avg_cc = []

while G.number_of_nodes() < n:
    new_node = G.number_of_nodes()
    G.add_node(new_node)
    targets = list(range(new_node)) + list(dict(G.degree).keys())
    targets = set(targets) - set([new_node])
    links = list(range(new_node-m, new_node))
    G.add_edges_from([(new_node, t) for t in links])
    if G.number_of_nodes() == 10**2 or G.number_of_nodes() == 10**3 or G.number_of_nodes() == 10**4:
        avg_cc.append(nx.average_clustering(G))
        degree_distributions.append(nx.degree_histogram(G))

for i in range(len(degree_distributions)):
    plt.plot(degree_distributions[i], label=f'n={10**(i+2)}')
plt.legend()
plt.show()


# A ------------------------------------------------------------------------------------------------------------
steps = [10**2, 10**3, 10**4]
for step in steps:
    subgraph = G.subgraph(range(step))
    degrees = [d for n, d in subgraph.degree()]
    degree_counts = Counter(degrees)
    degree_values = sorted(set(degrees))
    degree_probs = [degree_counts[d] / len(subgraph) for d in degree_values]
    plt.bar(np.array(degree_values), np.array(
        degree_probs), label=f'step {step}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution at Intermediate Steps')
plt.legend()
plt.show()
# ------------------------------------------------------------------------------------------------------------

# B ------------------------------------------------------------------------------------------------------------
def power_law(x, a, gamma):
    return a * x ** (-gamma)

m = 4
G = nx.complete_graph(m)
n = 10**4
degree_distributions = []
while G.number_of_nodes() < n:
    new_node = G.number_of_nodes()
    G.add_node(new_node)
    targets = list(range(new_node)) + list(dict(G.degree).keys())
    targets = set(targets) - set([new_node])
    links = list(range(new_node-m, new_node))
    G.add_edges_from([(new_node, t) for t in links])
    if G.number_of_nodes() == 10**2 or G.number_of_nodes() == 10**3 or G.number_of_nodes() == 10**4:
        degree_counts = np.array(nx.degree_histogram(G))
        degree_values = np.arange(len(degree_counts))
        degree_probs = degree_counts / degree_counts.sum()
        popt, pcov = curve_fit(power_law, degree_values, degree_probs)
        gamma = popt[1]
        plt.plot(degree_values, degree_probs, 'o', markersize=3,
                 label=f'step {new_node}')
        plt.plot(degree_values, power_law(degree_values, *popt),
                 'r--', label=f'γ = {gamma:.2f}')
        degree_distributions.append(degree_probs)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution at Intermediate Steps')
plt.legend()
plt.show()

epsilon = 1e-15
for i in range(1, len(degree_distributions)):
    p1 = degree_distributions[i-1] + epsilon
    p2 = degree_distributions[i] + epsilon
    kl_divergence = np.sum(p1 * np.log(p1/p2))
    print(f'KL-divergence between step {i-1} and step {i}: {kl_divergence}')
# ------------------------------------------------------------------------------------------------------------

# C ------------------------------------------------------------------------------------------------------------
cumulative_distributions = []
for i in range(len(degree_distributions)):
    cumulative_distributions.append(np.cumsum(degree_distributions[i]))
plt.plot(cumulative_distributions[0], label='step 100')
plt.plot(cumulative_distributions[1], label='step 1000')
plt.plot(cumulative_distributions[2], label='step 10000')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('Cumulative Distribution')
plt.title('Cumulative Degree Distribution at Intermediate Steps')
plt.legend()
plt.show()
# ------------------------------------------------------------------------------------------------------------

# D ------------------------------------------------------------------------------------------------------------
N_values = [100, 1000, 10000]
plt.plot(N_values, avg_cc, 'o-')
plt.xlabel('N')
plt.ylabel('Average Clustering Coefficient')
plt.xscale('log')
plt.show()
# ------------------------------------------------------------------------------------------------------------

# E ------------------------------------------------------------------------------------------------------------
m = 4
G = nx.complete_graph(m)
degree_dynamics = [[0, m]]
n = 10**4

timesteps = [100, 1000, 5000]
for i in range(m, n):

    G.add_node(i)
    targets = list(range(i))
    if i - m > 0:
        targets += list(dict(G.degree(range(i-m, i))).keys())
    targets = set(targets) - set([i])

    new_edges = []
    for j in range(m):
        chosen = np.random.choice(list(targets))
        targets -= set([chosen])
        new_edges.append((i, chosen))
    G.add_edges_from(new_edges)

    if i in timesteps or i == n-1:
        degree_counts = dict(nx.degree(G))
        initial_degree = degree_counts[0]
        timestep_degrees = [degree_counts.get(
            timestep, 0) for timestep in timesteps]
        degree_dynamics.append([initial_degree] + timestep_degrees)

print("Degree dynamics of node 0 and nodes added at t=100, t=1000, and t=5000:")
print("t\tNode 0\tt=100\tt=1000\tt=5000")
for i, row in enumerate(degree_dynamics):
    print(i, end='\t')
    print('\t'.join([str(x) for x in row]))
# ------------------------------------------------------------------------------------------------------------
