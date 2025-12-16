import random
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent import Agent, Message, AgentFactory
from agents.llm_service import LLMService
from config import AgentStance
from simulation.network import SocialNetwork, NetworkGenerator, NetworkType
from simulation.manager import SimulationManager

class TaskType(Enum):
    INFO_DIFFUSION = "Information Diffusion"
    RUMOR_DETECTION = "Rumor Detection"
    STANCE_EVOLUTION = "Stance Evolution"
    MULTI_ROLE_COLLAB = "Multi-Role Collaboration"
    GROUP_POLARIZATION = "Group Polarization"
    OPINION_DYNAMICS = "Opinion Dynamics"

@dataclass
class TaskResult:
    """Results from task execution"""
    task_type: TaskType
    success: bool
    metrics: Dict[str, Any]
    round_data: List[Dict]
    summary: str

class BaseTask(ABC):
    """Abstract base class for simulation tasks"""
    
    def __init__(self, manager: SimulationManager = None):
        self.manager = manager
        self.results: Optional[TaskResult] = None
    
    @abstractmethod
    def setup(self, **kwargs) -> "BaseTask":
        """Setup task-specific configuration"""
        pass
    
    @abstractmethod
    def run(self, n_rounds: int = 10, infection_probability: float = 0.3) -> TaskResult:
        """Execute the task"""
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate task-specific metrics"""
        pass


class InfoDiffusionTask(BaseTask):
    """
    Information Diffusion Task
    Simulates how information spreads through a social network.
    Measures coverage, depth, and spread velocity.
    """
    
    task_type = TaskType.INFO_DIFFUSION
    
    def __init__(self, manager: SimulationManager = None):
        super().__init__(manager)
        self.initial_message: Optional[Message] = None
        self.intervention_round: Optional[int] = None
        self.intervention_type: str = "none"
    
    def setup(
        self,
        message_content: str = "Breaking news: Major announcement coming soon!",
        source_count: int = 1,
        enable_intervention: bool = False,
        intervention_round: int = 5,
        intervention_type: str = "counter_message",
        **kwargs
    ) -> "InfoDiffusionTask":
        """
        Setup information diffusion scenario.
        
        Args:
            message_content: Initial message to spread
            source_count: Number of initial source nodes
            enable_intervention: Whether to introduce intervention
            intervention_round: Round to introduce intervention
            intervention_type: Type of intervention (counter_message, block, boost)
        """
        if not self.manager:
            raise ValueError("SimulationManager required")
        
        # Select source agents (high influence)
        agents_by_influence = sorted(
            self.manager.agents, 
            key=lambda a: a.influence_score, 
            reverse=True
        )
        source_agents = [a.agent_id for a in agents_by_influence[:source_count]]
        
        # Inject initial message
        self.initial_message = self.manager.inject_message(
            content=message_content,
            source_agent_ids=source_agents,
            topic="diffusion_test"
        )
        
        if enable_intervention:
            self.intervention_round = intervention_round
            self.intervention_type = intervention_type
        
        return self
    
    def _apply_intervention(self):
        """Apply intervention strategy"""
        if self.intervention_type == "counter_message":
            # Inject counter-message from fact-checkers
            fact_checkers = [
                a for a in self.manager.agents 
                if "fact" in a.persona.lower() or a.stance == AgentStance.OPPOSE
            ]
            if fact_checkers:
                counter_msg = f"FACT CHECK: The previous message may be misleading. Please verify before sharing."
                self.manager.inject_message(
                    content=counter_msg,
                    source_agent_ids=[fact_checkers[0].agent_id],
                    topic="fact_check"
                )
        elif self.intervention_type == "reduce_spread":
            # Reduce infection probability for remaining rounds
            pass  # Handled in run()
    
    def run(self, n_rounds: int = 10, infection_probability: float = 0.3) -> TaskResult:
        """Run diffusion simulation"""
        coverage_over_time = []
        current_infection_prob = infection_probability
        
        for round_num in range(n_rounds):
            # Check for intervention
            if self.intervention_round and round_num == self.intervention_round:
                self._apply_intervention()
                if self.intervention_type == "reduce_spread":
                    current_infection_prob *= 0.5
            
            round_stats = self.manager.run_round(current_infection_prob)
            coverage_over_time.append(self.manager.state.coverage)
            
            # Early stop if full coverage
            if self.manager.state.coverage >= 0.99:
                break
        
        metrics = self.evaluate()
        metrics["coverage_over_time"] = coverage_over_time
        
        self.results = TaskResult(
            task_type=self.task_type,
            success=True,
            metrics=metrics,
            round_data=self.manager.round_data,
            summary=f"Diffusion completed. Final coverage: {metrics['final_coverage']:.2%}"
        )
        
        return self.results
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate diffusion metrics"""
        infected = self.manager.state.infected_nodes
        total = len(self.manager.agents)
        
        # Calculate total shares
        total_shares = sum(r.get("shares", 0) for r in self.manager.round_data)
        
        # Get actual cascade depth from simulation state
        max_depth = self.manager.state.max_cascade_depth
        
        # Calculate velocity (rounds to reach 50% coverage)
        rounds_to_half = None
        for i, r in enumerate(self.manager.round_data):
            if r.get("coverage", 0) >= 0.5:
                rounds_to_half = i + 1
                break
        
        # Calculate spread velocity (coverage per round)
        avg_velocity = (len(infected) / total) / max(self.manager.state.round_num, 1) if total > 0 else 0
        
        return {
            "final_coverage": len(infected) / total if total > 0 else 0,
            "infected_count": len(infected),
            "total_agents": total,
            "total_shares": total_shares,
            "max_cascade_depth": max_depth,
            "spread_velocity": avg_velocity,
            "rounds_completed": self.manager.state.round_num,
            "rounds_to_50_percent": rounds_to_half,
            "messages_generated": self.manager.state.total_messages
        }


class RumorDetectionTask(BaseTask):
    """
    Rumor Detection Task
    Tests the network's ability to identify and correct misinformation.
    Uses a belief tracking system separate from stance.
    """
    
    task_type = TaskType.RUMOR_DETECTION
    
    def __init__(self, manager: SimulationManager = None):
        super().__init__(manager)
        self.rumor_message: Optional[Message] = None
        self.truth_message: Optional[Message] = None
        # Track agent beliefs: {agent_id: {"exposed_to_rumor": bool, "exposed_to_truth": bool, "belief": float}}
        # belief: -1 (rejected rumor), 0 (uncertain), 1 (believed rumor)
        self.agent_beliefs: Dict[str, Dict] = {}
        self.belief_history: List[Dict] = []
    
    def setup(
        self,
        rumor_content: str = "ALERT: Unverified reports suggest major company bankruptcy imminent!",
        truth_content: str = "FACT CHECK: The bankruptcy rumors are FALSE. Company finances are stable.",
        fact_checker_ratio: float = 0.1,
        rumor_delay_rounds: int = 3,
        **kwargs
    ) -> "RumorDetectionTask":
        """
        Setup rumor detection scenario.
        
        Args:
            rumor_content: The rumor to spread
            truth_content: The counter-message (truth)
            fact_checker_ratio: Proportion of agents acting as fact-checkers
            rumor_delay_rounds: Rounds before truth is released
        """
        if not self.manager:
            raise ValueError("SimulationManager required")
        
        # Initialize belief state for all agents
        for agent in self.manager.agents:
            self.agent_beliefs[agent.agent_id] = {
                "exposed_to_rumor": False,
                "exposed_to_truth": False,
                "belief": 0.0,  # -1 to 1 scale
                "is_fact_checker": False
            }
        
        # Ensure some fact-checker agents exist
        fact_checker_count = int(len(self.manager.agents) * fact_checker_ratio)
        fact_checker_ids = []
        
        for i, agent in enumerate(self.manager.agents):
            if i < fact_checker_count:
                agent.persona = "You are a dedicated fact-checker who verifies information and combats misinformation."
                agent.susceptibility = 0.1
                self.agent_beliefs[agent.agent_id]["is_fact_checker"] = True
                self.agent_beliefs[agent.agent_id]["belief"] = -0.5  # Skeptical by default
                fact_checker_ids.append(agent.agent_id)
        
        # Inject rumor
        self.rumor_message = self.manager.inject_message(
            content=rumor_content,
            is_rumor=True,
            topic="rumor_test"
        )
        
        # Mark initial spreaders as exposed
        for agent in self.manager.agents:
            if agent.agent_id in self.manager.state.infected_nodes:
                self.agent_beliefs[agent.agent_id]["exposed_to_rumor"] = True
        
        # Store truth for later injection
        self._truth_content = truth_content
        self._fact_checker_ids = fact_checker_ids
        self._truth_delay = rumor_delay_rounds
        
        return self
    
    def _update_beliefs(self):
        """Update agent beliefs based on message exposure"""
        for agent in self.manager.agents:
            belief_data = self.agent_beliefs[agent.agent_id]
            
            # Check recent inbox for rumor/truth exposure
            for msg in agent.inbox:
                if msg.is_rumor:
                    belief_data["exposed_to_rumor"] = True
                elif msg.is_rumor is False:  # Explicit truth (fact-check)
                    belief_data["exposed_to_truth"] = True
            
            # Also check seen messages
            if agent.agent_id in self.manager.state.infected_nodes:
                belief_data["exposed_to_rumor"] = True
            
            # Update belief based on exposure and agent characteristics
            if belief_data["exposed_to_rumor"] and not belief_data["exposed_to_truth"]:
                # Only exposed to rumor - belief increases based on susceptibility
                if not belief_data["is_fact_checker"]:
                    belief_change = 0.3 * agent.susceptibility
                    belief_data["belief"] = min(1.0, belief_data["belief"] + belief_change)
            
            elif belief_data["exposed_to_truth"]:
                # Exposed to truth - belief decreases (debunking effect)
                debunk_strength = 0.4 if belief_data["is_fact_checker"] else 0.2
                belief_data["belief"] = max(-1.0, belief_data["belief"] - debunk_strength)
    
    def run(self, n_rounds: int = 10, infection_probability: float = 0.3) -> TaskResult:
        """Run rumor detection simulation"""
        belief_over_time = []
        
        for round_num in range(n_rounds):
            # Inject truth after delay
            if round_num == self._truth_delay and self._fact_checker_ids:
                self.truth_message = self.manager.inject_message(
                    content=self._truth_content,
                    source_agent_ids=self._fact_checker_ids[:1],
                    is_rumor=False,
                    topic="fact_check"
                )
            
            round_stats = self.manager.run_round(infection_probability)
            
            # Update beliefs based on message exposure
            self._update_beliefs()
            
            # Track belief distribution
            believers = sum(1 for b in self.agent_beliefs.values() if b["belief"] > 0.3)
            skeptics = sum(1 for b in self.agent_beliefs.values() if b["belief"] < -0.3)
            uncertain = len(self.agent_beliefs) - believers - skeptics
            
            snapshot = {
                "round": round_num,
                "believers": believers,
                "skeptics": skeptics,
                "uncertain": uncertain,
                "avg_belief": sum(b["belief"] for b in self.agent_beliefs.values()) / len(self.agent_beliefs)
            }
            self.belief_history.append(snapshot)
            belief_over_time.append(believers / len(self.manager.agents))
        
        metrics = self.evaluate()
        metrics["belief_over_time"] = belief_over_time
        metrics["belief_history"] = self.belief_history
        
        self.results = TaskResult(
            task_type=self.task_type,
            success=True,
            metrics=metrics,
            round_data=self.manager.round_data,
            summary=f"Rumor detection completed. Detection accuracy: {metrics['detection_accuracy']:.2%}"
        )
        
        return self.results
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate rumor detection metrics"""
        total = len(self.manager.agents)
        
        # Count based on belief system
        believers = [b for b in self.agent_beliefs.values() if b["belief"] > 0.3]
        skeptics = [b for b in self.agent_beliefs.values() if b["belief"] < -0.3]
        exposed_to_rumor = [b for b in self.agent_beliefs.values() if b["exposed_to_rumor"]]
        exposed_to_truth = [b for b in self.agent_beliefs.values() if b["exposed_to_truth"]]
        
        # Detection accuracy: proportion who rejected rumor among those exposed
        exposed_count = len(exposed_to_rumor)
        if exposed_count > 0:
            detection_accuracy = len([b for b in exposed_to_rumor if b["belief"] < 0]) / exposed_count
        else:
            detection_accuracy = 0.0
        
        # False belief rate: proportion who believe rumor among all agents
        false_belief_rate = len(believers) / total if total > 0 else 0
        
        # Correction effectiveness: among those who saw truth, how many rejected rumor
        if exposed_to_truth:
            correction_rate = len([b for b in exposed_to_truth if b["belief"] < 0]) / len(exposed_to_truth)
        else:
            correction_rate = 0.0
        
        return {
            "detection_accuracy": detection_accuracy,
            "false_belief_rate": false_belief_rate,
            "correction_rate": correction_rate,
            "total_believers": len(believers),
            "total_skeptics": len(skeptics),
            "rumor_exposure_rate": len(exposed_to_rumor) / total if total > 0 else 0,
            "truth_exposure_rate": len(exposed_to_truth) / total if total > 0 else 0,
            "rumor_spread": len(self.manager.state.infected_nodes) / total,
            "rounds_completed": self.manager.state.round_num
        }


class StanceEvolutionTask(BaseTask):
    """
    Stance Evolution Task
    Studies how opinions change through social interaction.
    Measures polarization, consensus formation, and stance shifts.
    """
    
    task_type = TaskType.STANCE_EVOLUTION
    
    def __init__(self, manager: SimulationManager = None):
        super().__init__(manager)
        self.topic: str = ""
        self.initial_distribution: Dict[str, int] = {}
    
    def setup(
        self,
        topic: str = "AI regulation should be stricter",
        initial_support_ratio: float = 0.4,
        initial_oppose_ratio: float = 0.4,
        **kwargs
    ) -> "StanceEvolutionTask":
        """
        Setup stance evolution scenario.
        
        Args:
            topic: The controversial topic for debate
            initial_support_ratio: Initial proportion supporting
            initial_oppose_ratio: Initial proportion opposing
        """
        if not self.manager:
            raise ValueError("SimulationManager required")
        
        self.topic = topic
        
        # Assign initial stances
        n_agents = len(self.manager.agents)
        n_support = int(n_agents * initial_support_ratio)
        n_oppose = int(n_agents * initial_oppose_ratio)
        
        random.shuffle(self.manager.agents)
        
        for i, agent in enumerate(self.manager.agents):
            if i < n_support:
                agent.stance = random.choice([AgentStance.SUPPORT, AgentStance.STRONG_SUPPORT])
            elif i < n_support + n_oppose:
                agent.stance = random.choice([AgentStance.OPPOSE, AgentStance.STRONG_OPPOSE])
            else:
                agent.stance = AgentStance.NEUTRAL
        
        # Record initial distribution
        self.initial_distribution = self._get_stance_distribution()
        
        # Inject topic message from various perspectives
        supporters = [a for a in self.manager.agents if a.stance.value > 0]
        opposers = [a for a in self.manager.agents if a.stance.value < 0]
        
        if supporters:
            self.manager.inject_message(
                content=f"I strongly believe: {topic}. Here's why we need this change!",
                source_agent_ids=[supporters[0].agent_id],
                topic=topic
            )
        
        if opposers:
            self.manager.inject_message(
                content=f"I disagree with '{topic}'. The risks outweigh the benefits.",
                source_agent_ids=[opposers[0].agent_id],
                topic=topic
            )
        
        return self
    
    def _get_stance_distribution(self) -> Dict[str, int]:
        """Get current stance distribution"""
        distribution = {}
        for agent in self.manager.agents:
            stance_name = agent.stance.name
            distribution[stance_name] = distribution.get(stance_name, 0) + 1
        return distribution
    
    def _calculate_polarization_index(self) -> float:
        """
        Calculate polarization index.
        Higher value = more polarized (extreme positions).
        """
        values = [a.stance.value for a in self.manager.agents]
        if not values:
            return 0.0
        
        # Standard deviation of stance values
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        
        # Normalize to 0-1 range (max variance would be 4 for values -2 to 2)
        return min(variance / 4, 1.0)
    
    def run(self, n_rounds: int = 10, infection_probability: float = 0.4) -> TaskResult:
        """Run stance evolution simulation"""
        stance_history = []
        polarization_history = []
        
        for round_num in range(n_rounds):
            round_stats = self.manager.run_round(infection_probability)
            
            # Track stance evolution
            stance_history.append(self._get_stance_distribution())
            polarization_history.append(self._calculate_polarization_index())
            
            # Generate discussion messages
            if round_num % 2 == 0:
                # Random agents share their views
                for _ in range(min(3, len(self.manager.agents))):
                    agent = random.choice(self.manager.agents)
                    msg = agent.post_new_message(self.topic, self.manager.state.round_num)
                    if msg:
                        # Distribute to followers
                        for follower_id in agent.followers[:5]:
                            follower = self.manager.agents_by_id.get(follower_id)
                            if follower:
                                follower.inbox.append(msg)
        
        metrics = self.evaluate()
        metrics["stance_history"] = stance_history
        metrics["polarization_history"] = polarization_history
        
        self.results = TaskResult(
            task_type=self.task_type,
            success=True,
            metrics=metrics,
            round_data=self.manager.round_data,
            summary=f"Stance evolution completed. Polarization change: {metrics['polarization_change']:+.2f}"
        )
        
        return self.results
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate stance evolution metrics"""
        final_distribution = self._get_stance_distribution()
        final_polarization = self._calculate_polarization_index()
        
        # Calculate initial polarization from initial distribution
        initial_values = []
        for stance, count in self.initial_distribution.items():
            stance_enum = AgentStance[stance]
            initial_values.extend([stance_enum.value] * count)
        
        if initial_values:
            initial_mean = sum(initial_values) / len(initial_values)
            initial_variance = sum((v - initial_mean) ** 2 for v in initial_values) / len(initial_values)
            initial_polarization = min(initial_variance / 4, 1.0)
        else:
            initial_polarization = 0.0
        
        # Count stance changes
        total_changes = sum(len(a.stance_history) for a in self.manager.agents)
        
        return {
            "initial_distribution": self.initial_distribution,
            "final_distribution": final_distribution,
            "initial_polarization": initial_polarization,
            "final_polarization": final_polarization,
            "polarization_change": final_polarization - initial_polarization,
            "total_stance_changes": total_changes,
            "agents_changed": len([a for a in self.manager.agents if a.stance_history]),
            "rounds_completed": self.manager.state.round_num
        }


class MultiRoleCollabTask(BaseTask):
    """
    Multi-Role Collaboration Task
    Tests collaborative problem-solving across agents.
    Each agent has partial information; they must combine to reach conclusion.
    """
    
    task_type = TaskType.MULTI_ROLE_COLLAB
    
    def __init__(self, manager: SimulationManager = None):
        super().__init__(manager)
        self.puzzle_clues: Dict[str, str] = {}
        self.correct_answer: str = ""
        self.clue_distribution: Dict[str, List[str]] = {}
    
    def setup(
        self,
        puzzle_type: str = "mystery",
        clues: List[str] = None,
        answer: str = None,
        **kwargs
    ) -> "MultiRoleCollabTask":
        """
        Setup collaborative puzzle scenario.
        
        Args:
            puzzle_type: Type of collaborative puzzle
            clues: List of clues to distribute
            answer: The correct answer agents should reach
        """
        if not self.manager:
            raise ValueError("SimulationManager required")
        
        # Default mystery puzzle
        if clues is None:
            clues = [
                "Clue 1: The event happened on a Tuesday night.",
                "Clue 2: The suspect was seen near the old library.",
                "Clue 3: A blue car was spotted leaving the scene.",
                "Clue 4: The witness heard a loud noise at 9 PM.",
                "Clue 5: Security cameras show only one person entered.",
            ]
        
        if answer is None:
            answer = "The librarian, driving a blue car, caused the incident at 9 PM on Tuesday."
        
        self.correct_answer = answer
        
        # Distribute clues among agents
        n_agents = len(self.manager.agents)
        clues_per_agent = max(1, len(clues) // n_agents)
        
        shuffled_clues = clues.copy()
        random.shuffle(shuffled_clues)
        
        for i, agent in enumerate(self.manager.agents):
            start_idx = (i * clues_per_agent) % len(clues)
            agent_clues = shuffled_clues[start_idx:start_idx + clues_per_agent]
            
            if not agent_clues:
                agent_clues = [random.choice(clues)]
            
            self.clue_distribution[agent.agent_id] = agent_clues
            
            # Update agent's belief state with their clues
            agent.belief_state["clues"] = agent_clues
            agent.persona = f"You have the following information: {', '.join(agent_clues)}. Share and combine information with others to solve the puzzle."
            agent.stance = AgentStance.NEUTRAL
            agent.susceptibility = 0.7  # More open to information
        
        # Inject initial collaboration message
        first_agent = self.manager.agents[0]
        self.manager.inject_message(
            content=f"Let's work together to solve this mystery. I have some clues to share: {self.clue_distribution[first_agent.agent_id][0]}",
            source_agent_ids=[first_agent.agent_id],
            topic="collaboration"
        )
        
        return self
    
    def run(self, n_rounds: int = 8, infection_probability: float = 0.5) -> TaskResult:
        """Run collaboration simulation"""
        info_aggregation = []
        
        for round_num in range(n_rounds):
            round_stats = self.manager.run_round(infection_probability)
            
            # Track information sharing
            unique_clues_seen = set()
            for agent in self.manager.agents:
                for clue in agent.belief_state.get("clues", []):
                    unique_clues_seen.add(clue)
            
            info_aggregation.append(len(unique_clues_seen))
            
            # Encourage information sharing
            if round_num % 2 == 0:
                for agent in random.sample(self.manager.agents, min(3, len(self.manager.agents))):
                    clues = agent.belief_state.get("clues", [])
                    if clues:
                        msg = agent.post_new_message(
                            f"Here's what I know: {random.choice(clues)}",
                            self.manager.state.round_num
                        )
                        if msg:
                            for follower_id in agent.followers:
                                follower = self.manager.agents_by_id.get(follower_id)
                                if follower:
                                    follower.inbox.append(msg)
        
        metrics = self.evaluate()
        metrics["info_aggregation"] = info_aggregation
        
        self.results = TaskResult(
            task_type=self.task_type,
            success=True,
            metrics=metrics,
            round_data=self.manager.round_data,
            summary=f"Collaboration completed. Information coverage: {metrics['info_coverage']:.2%}"
        )
        
        return self.results
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate collaboration metrics"""
        # Count unique clues that reached all agents
        all_clues = set()
        for clues in self.clue_distribution.values():
            all_clues.update(clues)
        
        # Simplified metric: information coverage based on message sharing
        total_shares = self.manager.state.total_shares
        max_possible = len(all_clues) * len(self.manager.agents)
        
        return {
            "total_clues": len(all_clues),
            "agents_involved": len(self.manager.agents),
            "total_shares": total_shares,
            "info_coverage": min(total_shares / max_possible, 1.0) if max_possible > 0 else 0,
            "rounds_completed": self.manager.state.round_num,
            "messages_exchanged": self.manager.state.total_messages
        }


class GroupPolarizationTask(BaseTask):
    """
    Group Polarization Task
    Studies echo chamber formation and opinion extremification.
    """
    
    task_type = TaskType.GROUP_POLARIZATION
    
    def __init__(self, manager: SimulationManager = None):
        super().__init__(manager)
        self.communities: Dict[str, int] = {}
    
    def setup(
        self,
        n_communities: int = 2,
        intra_connection_prob: float = 0.4,
        inter_connection_prob: float = 0.05,
        topic: str = "Climate change requires immediate action",
        **kwargs
    ) -> "GroupPolarizationTask":
        """Setup polarization scenario with community structure"""
        if not self.manager:
            raise ValueError("SimulationManager required")
        
        # Get or create community structure
        self.communities = self.manager.network.get_communities(n_communities)
        
        # Assign stances based on community
        for agent in self.manager.agents:
            community = self.communities.get(agent.agent_id, 0)
            
            if community == 0:
                agent.stance = random.choice([AgentStance.SUPPORT, AgentStance.STRONG_SUPPORT])
                agent.persona = f"You belong to a community that supports: {topic}"
            else:
                agent.stance = random.choice([AgentStance.OPPOSE, AgentStance.STRONG_OPPOSE])
                agent.persona = f"You belong to a community that opposes: {topic}"
            
            agent.susceptibility = 0.3  # Less susceptible to change
        
        # Inject initial messages from each community
        for community_id in range(n_communities):
            community_agents = [a for a in self.manager.agents 
                             if self.communities.get(a.agent_id) == community_id]
            if community_agents:
                stance = "support" if community_id == 0 else "oppose"
                self.manager.inject_message(
                    content=f"Our community must {stance} this: {topic}",
                    source_agent_ids=[community_agents[0].agent_id],
                    topic=topic
                )
        
        return self
    
    def run(self, n_rounds: int = 10, infection_probability: float = 0.3) -> TaskResult:
        """Run polarization simulation"""
        polarization_history = []
        community_stance_history = []
        
        for round_num in range(n_rounds):
            round_stats = self.manager.run_round(infection_probability)
            
            # Track polarization
            values = [a.stance.value for a in self.manager.agents]
            mean = sum(values) / len(values) if values else 0
            variance = sum((v - mean) ** 2 for v in values) / len(values) if values else 0
            polarization_history.append(variance / 4)  # Normalized
            
            # Track community stances
            community_means = {}
            for community_id in set(self.communities.values()):
                community_agents = [a for a in self.manager.agents 
                                  if self.communities.get(a.agent_id) == community_id]
                if community_agents:
                    community_means[community_id] = sum(a.stance.value for a in community_agents) / len(community_agents)
            community_stance_history.append(community_means)
        
        metrics = self.evaluate()
        metrics["polarization_history"] = polarization_history
        metrics["community_stance_history"] = community_stance_history
        
        self.results = TaskResult(
            task_type=self.task_type,
            success=True,
            metrics=metrics,
            round_data=self.manager.round_data,
            summary=f"Polarization study completed. Final polarization index: {metrics['final_polarization']:.3f}"
        )
        
        return self.results
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate polarization metrics"""
        values = [a.stance.value for a in self.manager.agents]
        mean = sum(values) / len(values) if values else 0
        variance = sum((v - mean) ** 2 for v in values) / len(values) if values else 0
        
        # Inter-community distance
        community_means = {}
        for community_id in set(self.communities.values()):
            community_agents = [a for a in self.manager.agents 
                              if self.communities.get(a.agent_id) == community_id]
            if community_agents:
                community_means[community_id] = sum(a.stance.value for a in community_agents) / len(community_agents)
        
        inter_distance = 0
        means_list = list(community_means.values())
        if len(means_list) >= 2:
            inter_distance = abs(means_list[0] - means_list[1])
        
        return {
            "final_polarization": variance / 4,
            "inter_community_distance": inter_distance,
            "community_means": community_means,
            "extreme_agents": len([a for a in self.manager.agents if abs(a.stance.value) == 2]),
            "moderate_agents": len([a for a in self.manager.agents if a.stance.value == 0]),
            "rounds_completed": self.manager.state.round_num
        }


# Task Factory
class TaskFactory:
    """Factory for creating task instances"""
    
    TASK_CLASSES = {
        TaskType.INFO_DIFFUSION: InfoDiffusionTask,
        TaskType.RUMOR_DETECTION: RumorDetectionTask,
        TaskType.STANCE_EVOLUTION: StanceEvolutionTask,
        TaskType.MULTI_ROLE_COLLAB: MultiRoleCollabTask,
        TaskType.GROUP_POLARIZATION: GroupPolarizationTask,
    }
    
    @classmethod
    def create(cls, task_type: TaskType, manager: SimulationManager) -> BaseTask:
        """Create task instance of specified type"""
        task_class = cls.TASK_CLASSES.get(task_type)
        if not task_class:
            raise ValueError(f"Unknown task type: {task_type}")
        return task_class(manager)
    
    @classmethod
    def available_tasks(cls) -> List[TaskType]:
        """Get list of available task types"""
        return list(cls.TASK_CLASSES.keys())
