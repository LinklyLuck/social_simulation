import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum, IntEnum

class SimulationMode(Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    WEIBO = "weibo"
    GENERIC = "generic"

class TaskType(Enum):
    INFO_DIFFUSION = "Information Diffusion"
    RUMOR_DETECTION = "Rumor Detection"
    STANCE_EVOLUTION = "Stance Evolution"
    MULTI_ROLE_COLLAB = "Multi-Role Collaboration"
    GROUP_POLARIZATION = "Group Polarization"
    OPINION_DYNAMICS = "Opinion Dynamics"

class AgentStance(IntEnum):
    """Agent stance with numerical values for polarization metrics"""
    STRONG_SUPPORT = 2
    SUPPORT = 1
    NEUTRAL = 0
    OPPOSE = -1
    STRONG_OPPOSE = -2
    
    @classmethod
    def from_string(cls, s: str) -> 'AgentStance':
        """Convert string to AgentStance"""
        mapping = {
            'strong_support': cls.STRONG_SUPPORT,
            'support': cls.SUPPORT,
            'neutral': cls.NEUTRAL,
            'oppose': cls.OPPOSE,
            'strong_oppose': cls.STRONG_OPPOSE,
        }
        return mapping.get(s.lower(), cls.NEUTRAL)

class NetworkType(Enum):
    ERDOS_RENYI = "Erdős-Rényi Random"
    BARABASI_ALBERT = "Barabási-Albert Scale-Free"
    WATTS_STROGATZ = "Watts-Strogatz Small-World"
    CUSTOM = "Custom Dataset"

@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    agent_id: str
    persona: str
    stance: AgentStance = AgentStance.NEUTRAL
    trust_level: float = 0.5
    influence_score: float = 0.5
    susceptibility: float = 0.5
    
@dataclass
class SimulationConfig:
    """Main simulation configuration"""
    mode: SimulationMode = SimulationMode.TWITTER
    task_type: TaskType = TaskType.INFO_DIFFUSION
    network_type: NetworkType = NetworkType.BARABASI_ALBERT
    num_agents: int = 20
    num_rounds: int = 10
    infection_probability: float = 0.3
    trust_weight: float = 0.5
    random_seed: int = 42
    enable_topic_drift: bool = False
    enable_persona: bool = True
    
    # LLM settings
    llm_model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 150

@dataclass 
class EvaluationConfig:
    """Configuration for evaluation metrics"""
    track_coverage: bool = True
    track_depth: bool = True
    track_stance_changes: bool = True
    track_sentiment: bool = True
    compute_centrality: bool = True

# Persona templates for different agent types
PERSONA_TEMPLATES = {
    "skeptic": "You are a skeptical user who questions information and asks for sources. You rarely share unverified content.",
    "enthusiast": "You are an enthusiastic user who loves sharing interesting content. You engage actively in discussions.",
    "neutral": "You are a neutral observer who considers multiple perspectives before forming opinions.",
    "influencer": "You are a popular user with many followers. Your opinions carry weight in discussions.",
    "fact_checker": "You are dedicated to verifying information. You actively debunk misinformation.",
    "echo_chamber": "You strongly support your initial stance and tend to reinforce views within your group.",
    "bridge": "You connect different communities and help spread information across groups.",
    "lurker": "You mostly observe without engaging much. You rarely share or comment.",
}

# Sample controversial topics for stance evolution
SAMPLE_TOPICS = [
    "AI regulation should be stricter",
    "Social media platforms should verify all users",
    "Cryptocurrency will replace traditional currency",
    "Remote work is better than office work",
    "Electric vehicles should be mandatory by 2030",
]

# Sample rumor scenarios
SAMPLE_RUMORS = [
    {
        "content": "Breaking: A major tech company is secretly collecting user data through smart devices!",
        "is_rumor": True,
        "source": "Anonymous whistleblower"
    },
    {
        "content": "New study confirms that regular exercise improves mental health outcomes.",
        "is_rumor": False,
        "source": "Medical Research Journal"
    },
    {
        "content": "Government plans to implement digital currency and eliminate cash by next year!",
        "is_rumor": True,
        "source": "Viral social media post"
    },
]

# Default data paths
DATALAKE_PATH = os.path.join(os.path.dirname(__file__), "datalake")
NETWORKS_PATH = os.path.join(DATALAKE_PATH, "networks")
CONTENT_PATH = os.path.join(DATALAKE_PATH, "content")
