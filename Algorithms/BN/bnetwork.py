#bnetwork.py
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import truncnorm

from Algorithms.BN.utils import funcoes, carregar_repositorio, gerar_cpt


# =============================================================================
# Classe BNetwork 
# =============================================================================
class BNetwork:
    def __init__(self):
        self.model = DiscreteBayesianNetwork()
        self.nodes = {}
        self.evidence_distributions = {}
        self.evidence_hard = {}
        self._cpd_set_count = {}  # <<< contador de sets de CPD por nó

    def createNode(self, node_id, name, outcomes):
        self.nodes[node_id] = {"name": name, "outcomes": outcomes}
        self.model.add_node(node_id)

    def addEdge(self, parent_id, child_id):
        self.model.add_edge(parent_id, child_id)

    def setNodeCPD(self, node_id, cpt_values):
        parent_ids = list(self.model.get_parents(node_id))
        parent_cards = [len(self.nodes[p]["outcomes"]) for p in parent_ids]
        var_card = len(self.nodes[node_id]["outcomes"])

        cpt_values = np.array(cpt_values)
        if cpt_values.shape != (var_card, np.prod(parent_cards)):
            raise ValueError(f"Formato errado para {node_id}. Esperado {(var_card, np.prod(parent_cards))}, mas recebeu {cpt_values.shape}")

        # Garantir que os state_names estão consistentes
        state_names = {node_id: self.nodes[node_id]["outcomes"]}
        for pid in parent_ids:
            state_names[pid] = self.nodes[pid]["outcomes"]
        
        cpd = TabularCPD(
            variable=node_id,
            variable_card=var_card,
            values=cpt_values.tolist(),
            evidence=parent_ids if parent_ids else None,
            evidence_card=parent_cards if parent_cards else None,
            state_names=state_names
        )
        self._cpd_set_count[node_id] = self._cpd_set_count.get(node_id, 0) + 1
        print(f"[CPD-SET] {node_id} count={self._cpd_set_count[node_id]}")

        self.model.add_cpds(cpd)


    def setEvidence(self, node_id, value):
        outcomes = self.nodes[node_id]["outcomes"]
        num_states = len(outcomes)
        
        if isinstance(value, str):
            if value not in outcomes:
                raise ValueError(f"Estado {value} inválido para nó {node_id}")
            idx = outcomes.index(value)
            dist = [0.0] * num_states
            dist[idx] = 1.0
            self.evidence_distributions[node_id] = dist
            self.evidence_hard[node_id] = value
            
        elif isinstance(value, (list, np.ndarray)):
            if len(value) != num_states:
                raise ValueError(f"Número de estados incompatível. Esperado {num_states}, recebido {len(value)}")
            if not np.isclose(sum(value), 1.0, atol=0.01):
                raise ValueError(f"Probabilidades somam {sum(value):.4f}, deveria ser 1")
            self.evidence_distributions[node_id] = list(value)
            if node_id in self.evidence_hard:
                del self.evidence_hard[node_id]
        else:
            raise TypeError("Tipo inválido para evidência. Use string ou lista")

    def calculateTPN(self, node_id):
        if not self.model.get_cpds():
            raise ValueError("Modelo não possui CPTs definidas. Use setNodeCPD()")
            
        infer = VariableElimination(self.model)
        
        # Preparar evidências virtuais no formato correto
        virtual_evidence = []
        for node, dist in self.evidence_distributions.items():
            # Criar TabularCPD para a evidência virtual
            cpd = TabularCPD(
                variable=node,
                variable_card=len(dist),
                values=[[p] for p in dist],
                state_names={node: self.nodes[node]["outcomes"]}
            )
            virtual_evidence.append(cpd)
        
        # Realizar inferência com as evidências virtuais
        result = infer.query(
            variables=[node_id],
            virtual_evidence=virtual_evidence,
            show_progress=False
        )
        
        return result.values.tolist()

    



