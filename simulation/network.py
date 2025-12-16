import random
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import math

class NetworkType(Enum):
    ERDOS_RENYI = "erdos_renyi"
    BARABASI_ALBERT = "barabasi_albert"
    WATTS_STROGATZ = "watts_strogatz"
    COMPLETE = "complete"
    CUSTOM = "custom"

@dataclass
class SocialNetwork:
    """
    Represents a social network structure with nodes and edges.
    Supports directed graphs for follower/following relationships.
    """
    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # (from, to) - directed
    node_attributes: Dict[str, Dict] = field(default_factory=dict)
    edge_weights: Dict[Tuple[str, str], float] = field(default_factory=dict)
    network_type: NetworkType = NetworkType.BARABASI_ALBERT
    
    def add_node(self, node_id: str, attributes: Dict = None):
        """Add a node to the network"""
        if node_id not in self.nodes:
            self.nodes.append(node_id)
            self.node_attributes[node_id] = attributes or {}
    
    def add_edge(self, from_node: str, to_node: str, weight: float = 1.0):
        """Add a directed edge (from_node follows/influences to_node)"""
        if (from_node, to_node) not in self.edges:
            self.edges.append((from_node, to_node))
            self.edge_weights[(from_node, to_node)] = weight
    
    def get_neighbors(self, node_id: str, direction: str = "outgoing") -> List[str]:
        """Get neighbors of a node"""
        if direction == "outgoing":
            return [to_node for from_node, to_node in self.edges if from_node == node_id]
        elif direction == "incoming":
            return [from_node for from_node, to_node in self.edges if to_node == node_id]
        else:  # both
            outgoing = [to_node for from_node, to_node in self.edges if from_node == node_id]
            incoming = [from_node for from_node, to_node in self.edges if to_node == node_id]
            return list(set(outgoing + incoming))
    
    def get_followers(self, node_id: str) -> List[str]:
        """Get followers of a node (incoming edges)"""
        return self.get_neighbors(node_id, "incoming")
    
    def get_following(self, node_id: str) -> List[str]:
        """Get nodes that this node follows (outgoing edges)"""
        return self.get_neighbors(node_id, "outgoing")
    
    def get_degree(self, node_id: str, direction: str = "both") -> int:
        """Get degree of a node"""
        return len(self.get_neighbors(node_id, direction))
    
    def get_edge_weight(self, from_node: str, to_node: str) -> float:
        """Get weight of an edge"""
        return self.edge_weights.get((from_node, to_node), 0.0)
    
    def compute_centrality(self) -> Dict[str, float]:
        """Compute betweenness centrality approximation"""
        centrality = {node: 0.0 for node in self.nodes}
        
        for node in self.nodes:
            # Simple approximation: degree centrality
            in_degree = len(self.get_followers(node))
            out_degree = len(self.get_following(node))
            centrality[node] = (in_degree + out_degree) / (2 * len(self.nodes)) if self.nodes else 0
        
        return centrality
    
    def get_communities(self, num_communities: int = 2) -> Dict[str, int]:
        """Simple community detection based on connectivity"""
        communities = {}
        unassigned = set(self.nodes)
        current_community = 0
        
        while unassigned and current_community < num_communities:
            # Start from a random unassigned node
            seed = random.choice(list(unassigned))
            community_nodes = {seed}
            frontier = [seed]
            
            # BFS to find connected component
            while frontier and len(community_nodes) < len(self.nodes) // num_communities + 1:
                current = frontier.pop(0)
                neighbors = self.get_neighbors(current, "both")
                for neighbor in neighbors:
                    if neighbor in unassigned and neighbor not in community_nodes:
                        community_nodes.add(neighbor)
                        frontier.append(neighbor)
            
            for node in community_nodes:
                communities[node] = current_community
                unassigned.discard(node)
            
            current_community += 1
        
        # Assign remaining nodes
        for node in unassigned:
            communities[node] = current_community - 1
        
        return communities
    
    def to_dict(self) -> Dict:
        """Convert network to dictionary for serialization"""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "node_attributes": self.node_attributes,
            "edge_weights": {f"{k[0]}-{k[1]}": v for k, v in self.edge_weights.items()},
            "network_type": self.network_type.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SocialNetwork":
        """Create network from dictionary"""
        network = cls()
        network.nodes = data.get("nodes", [])
        network.edges = [tuple(e) for e in data.get("edges", [])]
        network.node_attributes = data.get("node_attributes", {})
        network.edge_weights = {
            (k.split("-")[0], k.split("-")[1]): v 
            for k, v in data.get("edge_weights", {}).items()
        }
        network.network_type = NetworkType(data.get("network_type", "barabasi_albert"))
        return network
    
    def to_json(self, filepath: str):
        """Save network to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> "SocialNetwork":
        """Load network from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class NetworkGenerator:
    """
    Factory class for generating various types of social networks.
    """
    
    @staticmethod
    def generate_erdos_renyi(
        n_nodes: int,
        edge_probability: float = 0.1,
        node_ids: List[str] = None,
        seed: int = None
    ) -> SocialNetwork:
        """
        Generate Erdős-Rényi random graph.
        Each edge exists with probability p.
        """
        if seed is not None:
            random.seed(seed)
        
        network = SocialNetwork(network_type=NetworkType.ERDOS_RENYI)
        
        # Create nodes
        if node_ids:
            for node_id in node_ids[:n_nodes]:
                network.add_node(node_id)
        else:
            for i in range(n_nodes):
                network.add_node(f"node_{i}")
        
        # Create edges with probability p
        for i, node_i in enumerate(network.nodes):
            for j, node_j in enumerate(network.nodes):
                if i != j and random.random() < edge_probability:
                    network.add_edge(node_i, node_j)
        
        return network
    
    @staticmethod
    def generate_barabasi_albert(
        n_nodes: int,
        m_edges: int = 2,
        node_ids: List[str] = None,
        seed: int = None
    ) -> SocialNetwork:
        """
        Generate Barabási-Albert scale-free network.
        Preferential attachment model where new nodes connect to existing nodes
        with probability proportional to their degree.
        """
        if seed is not None:
            random.seed(seed)
        
        network = SocialNetwork(network_type=NetworkType.BARABASI_ALBERT)
        
        # Prepare node IDs
        if node_ids:
            nodes = node_ids[:n_nodes]
        else:
            nodes = [f"node_{i}" for i in range(n_nodes)]
        
        # Initialize with m_edges + 1 fully connected nodes
        initial_nodes = min(m_edges + 1, n_nodes)
        for i in range(initial_nodes):
            network.add_node(nodes[i])
        
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                network.add_edge(nodes[i], nodes[j])
                network.add_edge(nodes[j], nodes[i])
        
        # Add remaining nodes with preferential attachment
        for i in range(initial_nodes, n_nodes):
            new_node = nodes[i]
            network.add_node(new_node)
            
            # Calculate degree of existing nodes
            existing_nodes = network.nodes[:-1]
            degrees = [network.get_degree(n) + 1 for n in existing_nodes]  # +1 to avoid zero
            total_degree = sum(degrees)
            
            # Select m_edges nodes to connect to
            targets = set()
            while len(targets) < min(m_edges, len(existing_nodes)):
                r = random.random() * total_degree
                cumsum = 0
                for node, degree in zip(existing_nodes, degrees):
                    cumsum += degree
                    if r <= cumsum:
                        targets.add(node)
                        break
            
            # Add edges
            for target in targets:
                network.add_edge(new_node, target)
                network.add_edge(target, new_node)
        
        return network
    
    @staticmethod
    def generate_watts_strogatz(
        n_nodes: int,
        k_neighbors: int = 4,
        rewire_prob: float = 0.1,
        node_ids: List[str] = None,
        seed: int = None
    ) -> SocialNetwork:
        """
        Generate Watts-Strogatz small-world network.
        Starts with ring lattice and rewires edges with probability p.
        """
        if seed is not None:
            random.seed(seed)
        
        network = SocialNetwork(network_type=NetworkType.WATTS_STROGATZ)
        
        # Prepare node IDs
        if node_ids:
            nodes = node_ids[:n_nodes]
        else:
            nodes = [f"node_{i}" for i in range(n_nodes)]
        
        for node in nodes:
            network.add_node(node)
        
        # Create ring lattice
        for i in range(n_nodes):
            for j in range(1, k_neighbors // 2 + 1):
                neighbor = (i + j) % n_nodes
                network.add_edge(nodes[i], nodes[neighbor])
                network.add_edge(nodes[neighbor], nodes[i])
        
        # Rewire edges
        edges_to_remove = []
        edges_to_add = []
        
        for i in range(n_nodes):
            for j in range(1, k_neighbors // 2 + 1):
                if random.random() < rewire_prob:
                    neighbor = (i + j) % n_nodes
                    # Find new random target
                    possible_targets = [n for n in range(n_nodes) 
                                       if n != i and (nodes[i], nodes[n]) not in network.edges]
                    if possible_targets:
                        new_target = random.choice(possible_targets)
                        edges_to_remove.append((nodes[i], nodes[neighbor]))
                        edges_to_add.append((nodes[i], nodes[new_target]))
        
        # Apply changes
        for edge in edges_to_remove:
            if edge in network.edges:
                network.edges.remove(edge)
        
        for edge in edges_to_add:
            network.add_edge(edge[0], edge[1])
        
        return network
    
    @staticmethod
    def generate_community_network(
        n_nodes: int,
        n_communities: int = 3,
        intra_prob: float = 0.3,
        inter_prob: float = 0.05,
        node_ids: List[str] = None,
        seed: int = None
    ) -> SocialNetwork:
        """
        Generate network with community structure.
        Higher connection probability within communities.
        """
        if seed is not None:
            random.seed(seed)
        
        network = SocialNetwork(network_type=NetworkType.CUSTOM)
        
        # Prepare node IDs
        if node_ids:
            nodes = node_ids[:n_nodes]
        else:
            nodes = [f"node_{i}" for i in range(n_nodes)]
        
        # Assign nodes to communities
        community_size = n_nodes // n_communities
        communities = {}
        for i, node in enumerate(nodes):
            community = min(i // community_size, n_communities - 1)
            communities[node] = community
            network.add_node(node, {"community": community})
        
        # Create edges
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i >= j:
                    continue
                
                # Higher probability for same community
                if communities[node_i] == communities[node_j]:
                    prob = intra_prob
                else:
                    prob = inter_prob
                
                if random.random() < prob:
                    network.add_edge(node_i, node_j)
                    network.add_edge(node_j, node_i)
        
        return network
    
    @classmethod
    def generate(
        cls,
        network_type: NetworkType,
        n_nodes: int,
        node_ids: List[str] = None,
        seed: int = None,
        **kwargs
    ) -> SocialNetwork:
        """
        Factory method to generate network of specified type.
        """
        if network_type == NetworkType.ERDOS_RENYI:
            return cls.generate_erdos_renyi(
                n_nodes, 
                kwargs.get("edge_probability", 0.1),
                node_ids, 
                seed
            )
        elif network_type == NetworkType.BARABASI_ALBERT:
            return cls.generate_barabasi_albert(
                n_nodes,
                kwargs.get("m_edges", 2),
                node_ids,
                seed
            )
        elif network_type == NetworkType.WATTS_STROGATZ:
            return cls.generate_watts_strogatz(
                n_nodes,
                kwargs.get("k_neighbors", 4),
                kwargs.get("rewire_prob", 0.1),
                node_ids,
                seed
            )
        elif network_type == NetworkType.COMPLETE:
            network = SocialNetwork(network_type=NetworkType.COMPLETE)
            nodes = node_ids[:n_nodes] if node_ids else [f"node_{i}" for i in range(n_nodes)]
            for node in nodes:
                network.add_node(node)
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if i != j:
                        network.add_edge(node_i, node_j)
            return network
        else:
            return cls.generate_barabasi_albert(n_nodes, 2, node_ids, seed)


class NetworkLoader:
    """
    Utility class for loading networks from various file formats.
    """
    
    @staticmethod
    def load_from_edge_list(filepath: str, delimiter: str = ",") -> SocialNetwork:
        """Load network from edge list file (CSV/TSV)"""
        network = SocialNetwork(network_type=NetworkType.CUSTOM)
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(delimiter)
                if len(parts) >= 2:
                    from_node, to_node = parts[0].strip(), parts[1].strip()
                    network.add_node(from_node)
                    network.add_node(to_node)
                    
                    weight = float(parts[2]) if len(parts) > 2 else 1.0
                    network.add_edge(from_node, to_node, weight)
        
        return network
    
    @staticmethod
    def load_from_json(filepath: str) -> SocialNetwork:
        """Load network from JSON file"""
        return SocialNetwork.from_json(filepath)
    
    @staticmethod
    def load_from_directory(directory: str) -> Dict[str, SocialNetwork]:
        """Load all networks from a directory"""
        networks = {}
        
        if not os.path.exists(directory):
            return networks
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            if filename.endswith('.json'):
                networks[filename[:-5]] = NetworkLoader.load_from_json(filepath)
            elif filename.endswith('.csv'):
                networks[filename[:-4]] = NetworkLoader.load_from_edge_list(filepath, ',')
            elif filename.endswith('.tsv'):
                networks[filename[:-4]] = NetworkLoader.load_from_edge_list(filepath, '\t')
        
        return networks
