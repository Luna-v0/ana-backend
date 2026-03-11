"""Testes do pipeline de retrieval híbrido do sistema ANA.

Cobre RRF, MMR, BM25 e integração do pipeline —
sem dependência de GPU, modelos ou Qdrant.
"""

import math
import pytest
import numpy as np

from ana.rag.retrieval import (
    IndiceBM25,
    reciprocal_rank_fusion,
    maximum_marginal_relevance,
)


class TestReciprocalRankFusion:
    """Testes da implementação de Reciprocal Rank Fusion."""

    def test_fusao_dois_rankings(self):
        """Verifica fusão de dois rankings distintos."""
        ranking_sem = ["A", "B", "C", "D"]
        ranking_bm25 = ["B", "A", "D", "C"]
        resultado = dict(reciprocal_rank_fusion([ranking_sem, ranking_bm25]))
        # A e B aparecem nas duas primeiras posições em rankings distintos
        assert resultado["A"] > resultado["C"]
        assert resultado["B"] > resultado["D"]

    def test_item_em_ambos_rankings_tem_score_maior(self):
        """Verifica que item no topo de ambos rankings tem score mais alto."""
        # Item "X" em posição 1 de ambos os rankings
        fusao = dict(reciprocal_rank_fusion([["X", "Y", "Z"], ["X", "Z", "Y"]]))
        assert fusao["X"] > fusao["Y"]
        assert fusao["X"] > fusao["Z"]

    def test_formula_matematica_correta(self):
        """Verifica a fórmula RRF: Σ 1/(k + rank) com k=60."""
        fusao = dict(reciprocal_rank_fusion([["A", "B"], ["B", "A"]], k=60))
        # A: pos1 no ranking 1 + pos2 no ranking 2
        esperado_a = 1/(60+1) + 1/(60+2)
        # B: pos2 no ranking 1 + pos1 no ranking 2
        esperado_b = 1/(60+2) + 1/(60+1)
        assert abs(fusao["A"] - esperado_a) < 1e-12
        assert abs(fusao["B"] - esperado_b) < 1e-12
        # A == B pois estão simétricamente posicionados
        assert abs(fusao["A"] - fusao["B"]) < 1e-12

    def test_k_personalizado(self):
        """Verifica que k personalizado altera os scores corretamente."""
        fusao_k60 = dict(reciprocal_rank_fusion([["A"]], k=60))
        fusao_k10 = dict(reciprocal_rank_fusion([["A"]], k=10))
        # k menor → score maior (menos penalização de ranking)
        assert fusao_k10["A"] > fusao_k60["A"]

    def test_ranking_unico(self):
        """Verifica fusão com apenas um ranking (sem fusão real)."""
        resultado = dict(reciprocal_rank_fusion([["A", "B", "C"]]))
        assert resultado["A"] > resultado["B"] > resultado["C"]

    def test_lista_vazia_retorna_lista_vazia(self):
        """Verifica que lista de rankings vazia retorna lista vazia."""
        assert reciprocal_rank_fusion([]) == []

    def test_ordenacao_decrescente(self):
        """Verifica que o resultado é sempre ordenado por score decrescente."""
        fusao = reciprocal_rank_fusion([["A", "B", "C"], ["A", "B", "C"]])
        scores = [score for _, score in fusao]
        assert scores == sorted(scores, reverse=True)

    def test_item_apenas_em_um_ranking(self):
        """Verifica que item em apenas um ranking tem score menor."""
        fusao = dict(reciprocal_rank_fusion([["A", "B"], ["A", "C"]]))
        # A aparece em ambos (pos 1), B e C apenas em um
        assert fusao["A"] > fusao["B"]
        assert fusao["A"] > fusao["C"]


class TestMaximumMarginalRelevance:
    """Testes da implementação de Maximum Marginal Relevance."""

    def _vetor(self, *componentes: float) -> list[float]:
        """Cria vetor normalizado para testes."""
        arr = np.array(componentes, dtype=float)
        norma = np.linalg.norm(arr)
        return (arr / norma).tolist() if norma > 0 else arr.tolist()

    def test_retorna_top_k_resultados(self):
        """Verifica que MMR retorna exatamente top_k resultados."""
        query = self._vetor(1.0, 0.0, 0.0)
        candidatos = [(str(i), self._vetor(1.0, float(i)*0.1, 0.0)) for i in range(10)]
        resultado = maximum_marginal_relevance(query, candidatos, top_k=5)
        assert len(resultado) == 5

    def test_todos_candidatos_se_menor_que_top_k(self):
        """Verifica que retorna todos se menos candidatos que top_k."""
        query = self._vetor(1.0, 0.0)
        candidatos = [("A", self._vetor(1.0, 0.0)), ("B", self._vetor(0.9, 0.0))]
        resultado = maximum_marginal_relevance(query, candidatos, top_k=10)
        assert len(resultado) == 2

    def test_lambda_1_priorizou_relevancia(self):
        """Com lambda=1.0, MMR degenera para ranking por relevância pura."""
        query = self._vetor(1.0, 0.0)
        # A é mais similar à query que B
        candidatos = [
            ("A", self._vetor(1.0, 0.0)),
            ("B", self._vetor(0.0, 1.0)),
        ]
        resultado = maximum_marginal_relevance(
            query, candidatos, lambda_mmr=1.0, top_k=2
        )
        assert resultado[0] == "A"

    def test_candidatos_vazios_retorna_lista_vazia(self):
        """Verifica retorno vazio para lista de candidatos vazia."""
        query = self._vetor(1.0, 0.0)
        resultado = maximum_marginal_relevance(query, [], top_k=5)
        assert resultado == []

    def test_todos_ids_unicos(self):
        """Verifica que MMR nunca seleciona o mesmo documento duas vezes."""
        query = self._vetor(1.0, 0.0, 0.0)
        candidatos = [(str(i), self._vetor(1.0, 0.0, 0.0)) for i in range(5)]
        resultado = maximum_marginal_relevance(query, candidatos, top_k=5)
        assert len(resultado) == len(set(resultado))


class TestIndiceBM25:
    """Testes do índice BM25 em memória."""

    @pytest.fixture
    def indice_populado(self) -> IndiceBM25:
        """Cria e popula um índice BM25 com corpus jurídico de teste."""
        corpus = [
            ("art1", "tratamento dados pessoais proteção privacidade"),
            ("art2", "consentimento titular dados pessoais fornecimento"),
            ("art7", "tratamento dados somente hipóteses consentimento titular"),
            ("art8", "consentimento escrito manifestação vontade titular"),
            ("art11", "dados sensíveis saúde biométricos genéticos consentimento específico"),
        ]
        indice = IndiceBM25()
        indice.construir(corpus)
        return indice

    def test_construcao_com_corpus(self, indice_populado: IndiceBM25):
        """Verifica que o índice é construído com o tamanho correto."""
        assert indice_populado.tamanho == 5

    def test_busca_retorna_resultados_relevantes(self, indice_populado: IndiceBM25):
        """Verifica que busca por 'consentimento titular' retorna artigos corretos."""
        resultados = indice_populado.buscar("consentimento titular", top_n=3)
        assert len(resultados) > 0
        ids = [r[0] for r in resultados]
        # Art. 2, 7 e 8 falam de consentimento
        assert any(i in ["art2", "art7", "art8"] for i in ids)

    def test_busca_dados_sensiveis(self, indice_populado: IndiceBM25):
        """Verifica que busca por 'dados sensíveis' retorna art11."""
        resultados = indice_populado.buscar("dados sensíveis biométricos")
        assert len(resultados) > 0
        assert resultados[0][0] == "art11"

    def test_score_positivo(self, indice_populado: IndiceBM25):
        """Verifica que resultados têm score maior que zero."""
        resultados = indice_populado.buscar("consentimento titular")
        for _, score in resultados:
            assert score > 0.0

    def test_busca_sem_resultados(self, indice_populado: IndiceBM25):
        """Verifica que busca por termo inexistente retorna lista vazia."""
        resultados = indice_populado.buscar("xyzzy qwerty foobar")
        assert resultados == []

    def test_ordenacao_decrescente_por_score(self, indice_populado: IndiceBM25):
        """Verifica que resultados são ordenados por score decrescente."""
        resultados = indice_populado.buscar("dados pessoais consentimento")
        if len(resultados) > 1:
            scores = [s for _, s in resultados]
            assert scores == sorted(scores, reverse=True)

    def test_indice_vazio_retorna_lista_vazia(self):
        """Verifica que busca em índice não construído retorna lista vazia."""
        indice = IndiceBM25()
        resultados = indice.buscar("qualquer coisa")
        assert resultados == []

    def test_construir_vazio_nao_lanca_erro(self):
        """Verifica que construir com lista vazia não lança exceção."""
        indice = IndiceBM25()
        indice.construir([])  # Não deve lançar exceção
        assert indice.tamanho == 0

    def test_top_n_limita_resultados(self, indice_populado: IndiceBM25):
        """Verifica que top_n limita o número de resultados retornados."""
        resultados = indice_populado.buscar("dados consentimento", top_n=2)
        assert len(resultados) <= 2
