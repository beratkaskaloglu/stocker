"""
core/entropy/graph.py
Yönlü bilgi akış grafiği (NTE tabanlı).
"""
from __future__ import annotations

import networkx as nx
import numpy as np


class InfoFlowGraph:
    """
    NTE matrisinden yönlü bilgi akış grafiği oluşturur.
    Her kenar, bir hissenin diğerini ne kadar etkilediğini gösterir.
    """

    def build(
        self,
        nte_matrix: np.ndarray,
        symbols: list[str],
        threshold: float = 0.01,
    ) -> nx.DiGraph:
        """
        NTE matrisinden directed graph oluşturur.

        Parameters
        ----------
        nte_matrix : np.ndarray – shape (n, n), NTE[i,j] = net info flow i→j
        symbols    : list[str]  – sembol isimleri
        threshold  : float      – |NTE| < threshold olan kenarlar atılır

        Returns
        -------
        nx.DiGraph – düğümler: semboller, kenarlar: bilgi akış yönü + ağırlık
        """
        n = len(symbols)
        G = nx.DiGraph()
        G.add_nodes_from(symbols)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                nte_val = nte_matrix[i, j]
                if abs(nte_val) < threshold:
                    continue

                if nte_val > 0:
                    # i → j: i, j'yi etkiliyor
                    G.add_edge(symbols[i], symbols[j], weight=nte_val)
                else:
                    # NTE < 0: j → i (reverse direction)
                    G.add_edge(symbols[j], symbols[i], weight=abs(nte_val))

        return G

    def get_features(self, graph: nx.DiGraph, symbol: str) -> dict:
        """
        Bir sembol için 5 adet graph centrality feature döndürür.

        Returns
        -------
        dict with keys:
            - in_degree_centrality   : ne kadar etkileniyor [0,1]
            - out_degree_centrality  : ne kadar etkiliyor [0,1]
            - pagerank               : genel etki skoru
            - clustering_coefficient : yakın grup bağları
            - betweenness_centrality : bilgi köprüsü skoru
        """
        if symbol not in graph:
            return {
                "in_degree_centrality": 0.0,
                "out_degree_centrality": 0.0,
                "pagerank": 0.0,
                "clustering_coefficient": 0.0,
                "betweenness_centrality": 0.0,
            }

        in_deg = nx.in_degree_centrality(graph)
        out_deg = nx.out_degree_centrality(graph)
        pr = nx.pagerank(graph, weight="weight")
        betw = nx.betweenness_centrality(graph, weight="weight")

        # Clustering coefficient: DiGraph için undirected'a çevirmek gerekir
        undirected = graph.to_undirected()
        clust = nx.clustering(undirected, weight="weight")

        return {
            "in_degree_centrality": in_deg.get(symbol, 0.0),
            "out_degree_centrality": out_deg.get(symbol, 0.0),
            "pagerank": pr.get(symbol, 0.0),
            "clustering_coefficient": clust.get(symbol, 0.0),
            "betweenness_centrality": betw.get(symbol, 0.0),
        }
