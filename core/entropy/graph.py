"""
core/entropy/graph.py
Yönlü bilgi akış grafiği (NTE tabanlı).
"""
from __future__ import annotations

import networkx as nx
import numpy as np


class InfoFlowGraph:
    """
    PSEUDO:
    1. build(nte_matrix: np.ndarray, symbols: list[str], threshold=0.01) → nx.DiGraph
       a. Her (i, j) çifti için NTE > threshold ise kenar ekle
       b. Kenar ağırlığı = NTE değeri
       c. NTE > 0: i → j (i, j'yi etkiliyor)
       d. NTE < 0: j → i (j, i'yi etkiliyor)
    2. get_features(graph, symbol) → dict
       a. in_degree_centrality   — ne kadar etkileniliyor
       b. out_degree_centrality  — ne kadar etkiliyor
       c. pagerank               — genel etki skoru
       d. clustering_coefficient — aynı gruba ait hisselerle bağ
       e. betweenness_centrality — bilgi köprüsü mü?
    3. save(graph, path) / load(path) → nx.DiGraph
       - networkx GraphML veya pickle formatı
    """

    def build(
        self,
        nte_matrix: np.ndarray,
        symbols: list[str],
        threshold: float = 0.01,
    ) -> nx.DiGraph:
        # TODO: implement
        raise NotImplementedError

    def get_features(self, graph: nx.DiGraph, symbol: str) -> dict:
        # TODO: implement — 5 adet graph feature döndür
        raise NotImplementedError
