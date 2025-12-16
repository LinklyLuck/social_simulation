import random
import time
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent import Agent, AgentFactory, Message
from agents.llm_service import LLMService
from config import AgentStance
from simulation.network import SocialNetwork, NetworkGenerator, NetworkType

class SimulationStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class SimulationLog:
    """Log entry for simulation events"""
    round_num: int
    timestamp: str
    event_type: str
    agent_id: Optional[str]
    message_id: Optional[str]
    content: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class SimulationState:
    """Current state of the simulation"""
    round_num: int = 0
    status: SimulationStatus = SimulationStatus.IDLE
    total_messages: int = 0
    total_shares: int = 0
    coverage: float = 0.0
    max_cascade_depth: int = 0  # Maximum cascade depth reached
    stance_distribution: Dict[str, int] = field(default_factory=dict)
    infected_nodes: set = field(default_factory=set)

class SimulationManager:
    """
    Main orchestrator for social activity simulations.
    Coordinates agents, network topology, task execution, and evaluation.
    """
    
    def __init__(
        self,
        network: SocialNetwork = None,
        agents: List[Agent] = None,
        llm_service: LLMService = None,
        config: Dict = None
    ):
        self.network = network
        self.agents = agents or []
        self.agents_by_id: Dict[str, Agent] = {}
        self.llm_service = llm_service or LLMService()
        self.config = config or {}
        
        self.state = SimulationState()
        self.logs: List[SimulationLog] = []
        self.round_data: List[Dict] = []
        
        # Platform-specific settings
        self.platform_mode = "generic"
        self.enable_persona = True
        self.enable_topic_drift = False
        self.trust_weight = 0.5
        
        # Platform behavior configurations
        self.platform_configs = {
            "twitter": {
                "max_message_length": 280,
                "share_name": "retweet",
                "reply_prefix": "@",
                "hashtag_enabled": True,
                "thread_enabled": True,
                "share_probability_boost": 1.2,  # Twitter is more viral
                "cascade_depth_weight": 1.5,
            },
            "reddit": {
                "max_message_length": 10000,
                "share_name": "crosspost",
                "reply_prefix": "",
                "hashtag_enabled": False,
                "thread_enabled": True,
                "share_probability_boost": 0.8,  # Reddit is more discussion-based
                "upvote_threshold": 5,  # Affects visibility
            },
            "weibo": {
                "max_message_length": 2000,
                "share_name": "转发",
                "reply_prefix": "@",
                "hashtag_enabled": True,
                "thread_enabled": False,
                "share_probability_boost": 1.3,  # Weibo is highly viral
            },
            "generic": {
                "max_message_length": 1000,
                "share_name": "share",
                "reply_prefix": "@",
                "hashtag_enabled": False,
                "thread_enabled": False,
                "share_probability_boost": 1.0,
            }
        }
        
        # Callbacks for UI updates
        self.on_round_complete: Optional[Callable] = None
        self.on_message_sent: Optional[Callable] = None
        self.on_state_change: Optional[Callable] = None
        
        # Initialize if agents provided
        if self.agents:
            self._index_agents()
    
    def get_platform_config(self, key: str, default=None):
        """Get platform-specific configuration value"""
        config = self.platform_configs.get(self.platform_mode, self.platform_configs["generic"])
        return config.get(key, default)
    
    def _index_agents(self):
        """Build agent lookup dictionary"""
        self.agents_by_id = {agent.agent_id: agent for agent in self.agents}
    
    def _log(self, event_type: str, content: str, agent_id: str = None, 
             message_id: str = None, metadata: Dict = None):
        """Add log entry"""
        log = SimulationLog(
            round_num=self.state.round_num,
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            agent_id=agent_id,
            message_id=message_id,
            content=content,
            metadata=metadata or {}
        )
        self.logs.append(log)
    
    def setup(
        self,
        n_agents: int = 20,
        network_type: NetworkType = NetworkType.BARABASI_ALBERT,
        agent_distribution: Dict[str, float] = None,
        seed: int = None,
        platform_mode: str = "generic",
        enable_persona: bool = True,
        enable_topic_drift: bool = False,
        trust_weight: float = 0.5,
        **network_kwargs
    ):
        """
        Initialize simulation with specified parameters.
        Creates network and populates with agents.
        
        Args:
            platform_mode: "twitter", "reddit", "weibo", or "generic"
            enable_persona: Whether to use agent personas for LLM generation
            enable_topic_drift: Whether topics can evolve during simulation
            trust_weight: Weight of trust in propagation decisions (0-1)
        """
        if seed is not None:
            random.seed(seed)
        
        # Store platform configuration
        self.platform_mode = platform_mode.lower() if platform_mode else "generic"
        self.enable_persona = enable_persona
        self.enable_topic_drift = enable_topic_drift
        self.trust_weight = trust_weight
        
        self._log("setup", f"Initializing simulation with {n_agents} agents, platform={self.platform_mode}")
        
        # Create agents
        self.agents = AgentFactory.create_population(
            n_agents=n_agents,
            distribution=agent_distribution,
            llm_service=self.llm_service
        )
        self._index_agents()
        
        # Generate network
        node_ids = [agent.agent_id for agent in self.agents]
        self.network = NetworkGenerator.generate(
            network_type=network_type,
            n_nodes=n_agents,
            node_ids=node_ids,
            seed=seed,
            **network_kwargs
        )
        
        # Set up agent social connections based on network
        for agent in self.agents:
            agent.followers = self.network.get_followers(agent.agent_id)
            agent.following = self.network.get_following(agent.agent_id)
            
            # Initialize trust network (higher trust for direct connections)
            for neighbor in agent.following:
                agent.trust_network[neighbor] = random.uniform(0.5, 0.9)
        
        self._log("setup", f"Network created: {len(self.network.edges)} edges")
        self.state.status = SimulationStatus.IDLE
        
        return self
    
    def inject_message(
        self,
        content: str,
        source_agent_ids: List[str] = None,
        is_rumor: bool = None,
        topic: str = ""
    ) -> Message:
        """
        Inject an initial message into the simulation.
        Can specify source agents or randomly select.
        """
        if not source_agent_ids:
            # Random selection weighted by influence
            weights = [agent.influence_score for agent in self.agents]
            source_agent = random.choices(self.agents, weights=weights, k=1)[0]
            source_agent_ids = [source_agent.agent_id]
        
        # Create message
        primary_source = source_agent_ids[0]
        message = Message.create(
            content=content,
            author_id=primary_source,
            timestamp=self.state.round_num,
            is_rumor=is_rumor,
            topic=topic
        )
        
        # Add to source agent's posted messages
        source_agent = self.agents_by_id.get(primary_source)
        if source_agent:
            source_agent.posted_messages.append(message)
            source_agent.seen_messages.add(message.id)
        
        # Distribute to followers
        for source_id in source_agent_ids:
            followers = self.network.get_followers(source_id)
            for follower_id in followers:
                follower = self.agents_by_id.get(follower_id)
                if follower:
                    follower.inbox.append(message)
        
        self._log(
            "inject",
            f"Message injected: '{content[:50]}...'",
            agent_id=primary_source,
            message_id=message.id,
            metadata={"is_rumor": is_rumor, "topic": topic}
        )
        
        self.state.total_messages += 1
        self.state.infected_nodes.add(primary_source)
        
        return message
    
    def run_round(self, infection_probability: float = 0.3) -> Dict[str, Any]:
        """
        Execute one round of the simulation.
        Each agent processes their inbox and potentially propagates messages.
        """
        self.state.round_num += 1
        self.state.status = SimulationStatus.RUNNING
        round_start = time.time()
        
        round_stats = {
            "round": self.state.round_num,
            "messages_sent": 0,
            "shares": 0,
            "stance_changes": 0,
            "new_infected": 0,
            "agents_processed": 0
        }
        
        self._log("round_start", f"Round {self.state.round_num} started")
        
        # Collect all messages to be propagated (to avoid modifying during iteration)
        propagation_queue: List[tuple] = []  # (sender_id, message, recipients)
        
        # Process each agent
        for agent in self.agents:
            if not agent.inbox:
                continue
            
            round_stats["agents_processed"] += 1
            
            for message in agent.inbox:
                if message.id in agent.seen_messages:
                    continue
                
                agent.seen_messages.add(message.id)
                
                # Generate reply
                reply_content = agent.generate_reply(message)
                round_stats["messages_sent"] += 1
                
                self._log(
                    "reply",
                    f"{agent.name}: {reply_content[:100]}",
                    agent_id=agent.agent_id,
                    message_id=message.id
                )
                
                # Decide to share
                if agent.decide_to_share(message):
                    # Get followers and apply infection probability
                    followers = self.network.get_followers(agent.agent_id)
                    recipients = []
                    
                    # Apply platform-specific share probability boost
                    share_boost = self.get_platform_config("share_probability_boost", 1.0)
                    
                    for follower_id in followers:
                        # P0-A FIX: Use follower's trust in agent (sender), not agent's trust in follower
                        follower_agent = self.agents_by_id.get(follower_id)
                        if follower_agent:
                            trust = follower_agent.trust_network.get(agent.agent_id, 0.5)
                        else:
                            trust = 0.5
                        
                        # Apply trust_weight to balance trust influence
                        weighted_trust = self.trust_weight * trust + (1 - self.trust_weight) * 0.5
                        prob = infection_probability * weighted_trust * share_boost
                        
                        if random.random() < min(prob, 1.0):  # Cap at 1.0
                            recipients.append(follower_id)
                    
                    if recipients:
                        # Create repost with platform-specific formatting
                        share_name = self.get_platform_config("share_name", "share")
                        reply_prefix = self.get_platform_config("reply_prefix", "@")
                        repost = message.create_repost(agent.agent_id, self.state.round_num, reply_content[:50])
                        propagation_queue.append((agent.agent_id, repost, recipients))
                        round_stats["shares"] += 1
                        
                        # Track cascade depth
                        if repost.depth > self.state.max_cascade_depth:
                            self.state.max_cascade_depth = repost.depth
                        
                        self._log(
                            share_name,
                            f"{agent.name} {share_name}d message to {len(recipients)} followers",
                            agent_id=agent.agent_id,
                            message_id=message.id,
                            metadata={"recipients": len(recipients), "platform": self.platform_mode, "depth": repost.depth}
                        )
            
            # Check stance change
            if agent.decide_stance_change(agent.inbox, self.state.round_num):
                round_stats["stance_changes"] += 1
                self._log(
                    "stance_change",
                    f"{agent.name} changed stance to {agent.stance.name}",
                    agent_id=agent.agent_id
                )
            
            # Clear processed inbox
            agent.inbox.clear()
        
        # Execute propagation
        for sender_id, message, recipients in propagation_queue:
            for recipient_id in recipients:
                recipient = self.agents_by_id.get(recipient_id)
                if recipient:
                    recipient.inbox.append(message)
                    if recipient_id not in self.state.infected_nodes:
                        self.state.infected_nodes.add(recipient_id)
                        round_stats["new_infected"] += 1
        
        # Update state
        self.state.total_messages += round_stats["messages_sent"]
        self.state.total_shares += round_stats["shares"]
        self.state.coverage = len(self.state.infected_nodes) / len(self.agents)
        
        # Update stance distribution
        self.state.stance_distribution = {}
        for agent in self.agents:
            stance_name = agent.stance.name
            self.state.stance_distribution[stance_name] = \
                self.state.stance_distribution.get(stance_name, 0) + 1
        
        round_stats["duration"] = time.time() - round_start
        round_stats["coverage"] = self.state.coverage
        round_stats["stance_distribution"] = self.state.stance_distribution.copy()
        
        self.round_data.append(round_stats)
        self._log("round_end", f"Round {self.state.round_num} completed", 
                 metadata=round_stats)
        
        # Trigger callback if set
        if self.on_round_complete:
            self.on_round_complete(round_stats)
        
        return round_stats
    
    def run(
        self,
        n_rounds: int = 10,
        infection_probability: float = 0.3,
        stop_on_full_coverage: bool = False
    ) -> List[Dict]:
        """
        Run full simulation for specified number of rounds.
        """
        self._log("run_start", f"Starting simulation for {n_rounds} rounds")
        
        for round_num in range(n_rounds):
            if self.state.status == SimulationStatus.PAUSED:
                break
            
            round_stats = self.run_round(infection_probability)
            
            # Check stopping condition
            if stop_on_full_coverage and self.state.coverage >= 0.99:
                self._log("early_stop", "Full coverage reached, stopping early")
                break
        
        self.state.status = SimulationStatus.COMPLETED
        self._log("run_complete", f"Simulation completed after {self.state.round_num} rounds")
        
        return self.round_data
    
    def pause(self):
        """Pause the simulation"""
        self.state.status = SimulationStatus.PAUSED
        self._log("pause", "Simulation paused")
    
    def resume(self):
        """Resume paused simulation"""
        if self.state.status == SimulationStatus.PAUSED:
            self.state.status = SimulationStatus.RUNNING
            self._log("resume", "Simulation resumed")
    
    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive simulation results"""
        return {
            "config": self.config,
            "final_state": {
                "rounds_completed": self.state.round_num,
                "total_messages": self.state.total_messages,
                "total_shares": self.state.total_shares,
                "coverage": self.state.coverage,
                "stance_distribution": self.state.stance_distribution
            },
            "round_data": self.round_data,
            "agents": [agent.to_dict() for agent in self.agents],
            "network": {
                "nodes": len(self.network.nodes),
                "edges": len(self.network.edges),
                "type": self.network.network_type.value
            },
            "logs": [
                {
                    "round": log.round_num,
                    "type": log.event_type,
                    "content": log.content
                }
                for log in self.logs[-100:]  # Last 100 logs
            ]
        }
    
    def get_agent_states(self) -> List[Dict]:
        """Get current state of all agents"""
        return [
            {
                "id": agent.agent_id,
                "name": agent.name,
                "stance": agent.stance.name,
                "stance_value": agent.stance.value,
                "influence": agent.influence_score,
                "followers": len(agent.followers),
                "messages_posted": len(agent.posted_messages),
                "infected": agent.agent_id in self.state.infected_nodes
            }
            for agent in self.agents
        ]
    
    def export_logs(self, filepath: str):
        """Export logs to JSON file"""
        logs_data = [
            {
                "round": log.round_num,
                "timestamp": log.timestamp,
                "event_type": log.event_type,
                "agent_id": log.agent_id,
                "message_id": log.message_id,
                "content": log.content,
                "metadata": log.metadata
            }
            for log in self.logs
        ]
        
        with open(filepath, 'w') as f:
            json.dump(logs_data, f, indent=2)
    
    def reset(self):
        """Reset simulation to initial state"""
        self.state = SimulationState()
        self.logs = []
        self.round_data = []
        
        for agent in self.agents:
            agent.inbox.clear()
            agent.seen_messages.clear()
            agent.stance_history.clear()
            agent.shared_messages.clear()  # Important: prevent share suppression in rerun
        
        self._log("reset", "Simulation reset")
