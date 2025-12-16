import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import uuid
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AgentStance
from .llm_service import LLMService, LLMResponse

@dataclass
class Message:
    """Represents a message/post in the simulation"""
    id: str
    content: str
    author_id: str
    origin_id: str  # Original message ID if repost
    timestamp: int  # Simulation round
    parent_id: Optional[str] = None  # ID of message being replied/reposted (for cascade tree)
    depth: int = 0  # Cascade depth (0 = original, 1 = first repost, etc.)
    is_rumor: Optional[bool] = None
    topic: str = ""
    engagement: int = 0
    
    @classmethod
    def create(cls, content: str, author_id: str, timestamp: int, 
               is_rumor: Optional[bool] = None, topic: str = "") -> "Message":
        msg_id = str(uuid.uuid4())[:8]
        return cls(
            id=msg_id,
            content=content,
            author_id=author_id,
            origin_id=msg_id,
            parent_id=None,
            depth=0,
            timestamp=timestamp,
            is_rumor=is_rumor,
            topic=topic
        )
    
    def create_repost(self, new_author_id: str, timestamp: int, comment: str = "") -> "Message":
        """Create a repost/share of this message"""
        new_content = f"RT @{self.author_id}: {self.content}"
        if comment:
            new_content = f"{comment} // {new_content}"
        
        return Message(
            id=str(uuid.uuid4())[:8],
            content=new_content,
            author_id=new_author_id,
            origin_id=self.origin_id,
            parent_id=self.id,  # Track parent for cascade tree
            depth=self.depth + 1,  # Increment depth
            timestamp=timestamp,
            is_rumor=self.is_rumor,
            topic=self.topic
        )

@dataclass
class Agent:
    """
    Social media agent with persona, stance, and behavior.
    Driven by LLM for natural language generation.
    """
    agent_id: str
    name: str
    persona: str
    stance: AgentStance = AgentStance.NEUTRAL
    trust_level: float = 0.5
    influence_score: float = 0.5
    susceptibility: float = 0.5
    
    # Social network
    followers: List[str] = field(default_factory=list)
    following: List[str] = field(default_factory=list)
    trust_network: Dict[str, float] = field(default_factory=dict)
    
    # Message handling
    inbox: List[Message] = field(default_factory=list)
    posted_messages: List[Message] = field(default_factory=list)
    seen_messages: set = field(default_factory=set)
    shared_messages: set = field(default_factory=set)  # Track origin_ids that have been shared
    
    # State tracking
    stance_history: List[tuple] = field(default_factory=list)
    belief_state: Dict[str, Any] = field(default_factory=dict)
    
    # LLM service reference (set during initialization)
    _llm_service: Optional[LLMService] = field(default=None, repr=False)
    
    def set_llm_service(self, service: LLMService):
        """Set the LLM service for this agent"""
        self._llm_service = service
    
    def get_system_prompt(self) -> str:
        """Generate system prompt based on persona and stance"""
        stance_desc = {
            AgentStance.STRONG_SUPPORT: "You strongly support this viewpoint and actively advocate for it.",
            AgentStance.SUPPORT: "You lean towards supporting this viewpoint.",
            AgentStance.NEUTRAL: "You are neutral and consider multiple perspectives.",
            AgentStance.OPPOSE: "You lean towards opposing this viewpoint.",
            AgentStance.STRONG_OPPOSE: "You strongly oppose this viewpoint and actively argue against it.",
        }
        
        return f"""You are {self.name}, a social media user with the following characteristics:
{self.persona}

Stance on current topic: {stance_desc.get(self.stance, 'Neutral')}
Trust level: {'High' if self.trust_level > 0.7 else 'Medium' if self.trust_level > 0.3 else 'Low'}
Influence: {'Influential' if self.influence_score > 0.7 else 'Average' if self.influence_score > 0.3 else 'Low profile'}

Respond naturally as this character would. Keep responses concise (1-3 sentences) and authentic to social media style."""
    
    def generate_reply(self, message: Message, context: str = "") -> str:
        """Generate a reply to a message using LLM"""
        if not self._llm_service:
            return f"[{self.name}]: Interesting point."
        
        prompt = f"""You received this message from another user:
"{message.content}"

{context}

How do you respond? Keep it natural and brief (1-2 sentences, social media style)."""
        
        response = self._llm_service.generate_response(
            prompt=prompt,
            system_prompt=self.get_system_prompt()
        )
        return response.content
    
    def decide_to_share(self, message: Message) -> bool:
        """Decide whether to share/repost a message"""
        # Check if already shared this original message (prevent duplicate shares)
        if message.origin_id in self.shared_messages:
            return False
        
        sender_trust = self.trust_network.get(message.author_id, 0.5)
        
        # Base share probability - higher to make propagation more visible
        # Factors: trust in sender (0-1), susceptibility (0-1), base probability
        base_prob = 0.6  # Increased from 0.5 for better propagation
        
        # Stance alignment check
        if message.is_rumor is not None:
            if self.stance.value > 0 and message.is_rumor:
                # Supportive agents more likely to share rumors
                share_prob = 0.8 * sender_trust * self.susceptibility
            elif self.stance.value < 0 and message.is_rumor:
                # Opposing agents less likely to share rumors
                share_prob = 0.3 * sender_trust * self.susceptibility
            else:
                share_prob = base_prob * sender_trust * self.susceptibility
        else:
            # Regular messages: base probability weighted by trust and susceptibility
            share_prob = base_prob * sender_trust * (0.5 + 0.5 * self.susceptibility)
        
        should_share = random.random() < share_prob
        if should_share:
            self.shared_messages.add(message.origin_id)
        return should_share
    
    def decide_stance_change(self, messages: List[Message], round_num: int) -> bool:
        """
        Decide if agent should change stance based on received messages.
        Uses LLM for structured analysis when available, falls back to enhanced heuristics.
        """
        if not messages:
            return False
        
        # Try LLM-based stance analysis if available (non-mock)
        if self._llm_service and len(messages) > 0:
            # Check if using real LLM (not mock)
            client_type = getattr(self._llm_service, '_preferred_client', 'mock')
            if client_type != 'mock':
                try:
                    result = self._llm_stance_analysis(messages, round_num)
                    if result is not None:  # Successfully parsed
                        return result
                except Exception:
                    pass  # Fall back to heuristic method
        
        # Always use heuristic method for mock LLM or when LLM fails
        return self._heuristic_stance_analysis(messages, round_num)
    
    def _llm_stance_analysis(self, messages: List[Message], round_num: int) -> bool:
        """Use LLM to analyze messages and determine stance change"""
        # Combine recent messages for analysis
        message_summary = "\n".join([
            f"- {msg.content[:100]}" for msg in messages[:5]  # Limit to 5 messages
        ])
        
        prompt = f"""Analyze these social media messages and determine their overall persuasive direction.

Messages:
{message_summary}

Your current stance value: {self.stance.value} ({self.stance.name})
Your susceptibility to influence: {self.susceptibility:.2f}

Respond with ONLY a JSON object in this exact format:
{{"stance_direction": "support" or "oppose" or "neutral", "strength": 0.0 to 1.0, "should_change": true or false}}

Where:
- stance_direction: The dominant persuasive direction of the messages
- strength: How persuasive the messages are (0 = not at all, 1 = very)
- should_change: Whether you would change your stance based on these messages
"""
        
        response = self._llm_service.generate_response(
            prompt=prompt,
            system_prompt=self.get_system_prompt()
        )
        
        # Parse LLM response
        import json
        # Extract JSON from response
        content = response.content.strip()
        # Handle potential markdown code blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        # Try to parse JSON - raise if fails to trigger fallback
        result = json.loads(content)
        
        if result.get("should_change", False):
            old_stance = self.stance
            direction = result.get("stance_direction", "neutral")
            strength = result.get("strength", 0.5)
            
            # Apply change based on direction and strength
            if direction == "support" and self.stance.value < 2:
                change = 1 if strength > 0.6 else 0
                if change > 0:
                    new_value = min(2, self.stance.value + change)
                    self.stance = AgentStance(new_value)
            elif direction == "oppose" and self.stance.value > -2:
                change = 1 if strength > 0.6 else 0
                if change > 0:
                    new_value = max(-2, self.stance.value - change)
                    self.stance = AgentStance(new_value)
            
            if self.stance != old_stance:
                self.stance_history.append((round_num, old_stance, self.stance))
                return True
        
        return False
    
    def _heuristic_stance_analysis(self, messages: List[Message], round_num: int) -> bool:
        """Enhanced heuristic method for stance change detection"""
        # Expanded keyword sets for better detection
        support_keywords = [
            'agree', 'support', 'true', 'correct', 'yes', 'absolutely',
            'right', 'definitely', 'exactly', 'good point', 'well said',
            'i think so', 'makes sense', 'valid', 'important', 'necessary'
        ]
        oppose_keywords = [
            'disagree', 'oppose', 'false', 'wrong', 'no', 'incorrect',
            'misleading', 'fake', 'doubt', 'skeptical', 'nonsense',
            'bad idea', 'problematic', 'dangerous', 'flawed', 'debunk'
        ]
        
        support_signals = 0
        oppose_signals = 0
        
        for msg in messages:
            sender_trust = self.trust_network.get(msg.author_id, 0.5)
            # Stronger stances harder to change
            weight = sender_trust * (1 - abs(self.stance.value) / 3)
            
            content_lower = msg.content.lower()
            
            # Count keyword matches (weighted by position - earlier = more important)
            for i, keyword in enumerate(support_keywords):
                if keyword in content_lower:
                    support_signals += weight * (1 - i * 0.03)  # Slight decay for later keywords
                    
            for i, keyword in enumerate(oppose_keywords):
                if keyword in content_lower:
                    oppose_signals += weight * (1 - i * 0.03)
            
            # Consider message sentiment patterns
            if msg.content.startswith("RT") or "retweet" in content_lower:
                # Reposts indicate agreement
                support_signals += weight * 0.3
            
            if "?" in msg.content and any(w in content_lower for w in ['really', 'sure', 'think']):
                # Skeptical questions
                oppose_signals += weight * 0.2
        
        # Calculate net influence
        net_influence = support_signals - oppose_signals
        
        # Threshold based on susceptibility
        threshold = 0.5 * (1 - self.susceptibility)
        
        if abs(net_influence) > threshold:
            old_stance = self.stance
            if net_influence > 0 and self.stance.value < 2:
                new_value = min(2, self.stance.value + 1)
                self.stance = AgentStance(new_value)
            elif net_influence < 0 and self.stance.value > -2:
                new_value = max(-2, self.stance.value - 1)
                self.stance = AgentStance(new_value)
            
            if self.stance != old_stance:
                self.stance_history.append((round_num, old_stance, self.stance))
                return True
        
        return False
    
    def post_new_message(self, topic: str, round_num: int) -> Optional[Message]:
        """Generate and post a new message about a topic"""
        if not self._llm_service:
            content = f"[{self.name}] My thoughts on {topic}"
        else:
            prompt = f"Write a short social media post (1-2 sentences) about: {topic}"
            response = self._llm_service.generate_response(
                prompt=prompt,
                system_prompt=self.get_system_prompt()
            )
            content = response.content
        
        message = Message.create(
            content=content,
            author_id=self.agent_id,
            timestamp=round_num,
            topic=topic
        )
        self.posted_messages.append(message)
        return message
    
    def process_inbox(self, round_num: int) -> List[Dict[str, Any]]:
        """Process all messages in inbox and return actions taken"""
        actions = []
        
        for message in self.inbox:
            if message.id in self.seen_messages:
                continue
            
            self.seen_messages.add(message.id)
            
            # Generate reply
            reply = self.generate_reply(message)
            actions.append({
                "type": "reply",
                "agent_id": self.agent_id,
                "message_id": message.id,
                "content": reply,
                "round": round_num
            })
            
            # Decide to share
            if self.decide_to_share(message):
                actions.append({
                    "type": "share",
                    "agent_id": self.agent_id,
                    "message_id": message.id,
                    "round": round_num
                })
        
        # Check stance change
        if self.decide_stance_change(self.inbox, round_num):
            actions.append({
                "type": "stance_change",
                "agent_id": self.agent_id,
                "new_stance": self.stance.name,
                "round": round_num
            })
        
        # Clear inbox after processing
        self.inbox.clear()
        
        return actions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "persona": self.persona,
            "stance": self.stance.name,
            "trust_level": self.trust_level,
            "influence_score": self.influence_score,
            "susceptibility": self.susceptibility,
            "followers_count": len(self.followers),
            "following_count": len(self.following),
            "messages_posted": len(self.posted_messages),
            "stance_history": [(r, o.name, n.name) for r, o, n in self.stance_history]
        }


class AgentFactory:
    """Factory class for creating agents with various configurations"""
    
    PERSONA_TEMPLATES = {
        "skeptic": {
            "persona": "You are a skeptical user who questions information and asks for sources. You rarely share unverified content.",
            "stance": AgentStance.OPPOSE,
            "susceptibility": 0.2,
            "trust_level": 0.3
        },
        "enthusiast": {
            "persona": "You are an enthusiastic user who loves sharing interesting content. You engage actively in discussions.",
            "stance": AgentStance.SUPPORT,
            "susceptibility": 0.7,
            "trust_level": 0.7
        },
        "neutral": {
            "persona": "You are a neutral observer who considers multiple perspectives before forming opinions.",
            "stance": AgentStance.NEUTRAL,
            "susceptibility": 0.5,
            "trust_level": 0.5
        },
        "influencer": {
            "persona": "You are a popular user with many followers. Your opinions carry weight in discussions.",
            "stance": AgentStance.SUPPORT,
            "susceptibility": 0.3,
            "trust_level": 0.8,
            "influence_score": 0.9
        },
        "fact_checker": {
            "persona": "You are dedicated to verifying information. You actively debunk misinformation and share verified facts.",
            "stance": AgentStance.OPPOSE,
            "susceptibility": 0.1,
            "trust_level": 0.9
        },
        "echo_chamber": {
            "persona": "You strongly support your initial stance and tend to reinforce views within your group.",
            "stance": AgentStance.STRONG_SUPPORT,
            "susceptibility": 0.1,
            "trust_level": 0.4
        },
        "bridge": {
            "persona": "You connect different communities and help spread information across groups. You're open to different viewpoints.",
            "stance": AgentStance.NEUTRAL,
            "susceptibility": 0.6,
            "trust_level": 0.6
        },
        "lurker": {
            "persona": "You mostly observe without engaging much. You rarely share or comment but read a lot.",
            "stance": AgentStance.NEUTRAL,
            "susceptibility": 0.8,
            "trust_level": 0.5,
            "influence_score": 0.1
        },
    }
    
    NAMES = [
        "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery",
        "Sage", "River", "Phoenix", "Dakota", "Skyler", "Charlie", "Drew", "Blake",
        "Jamie", "Reese", "Cameron", "Hayden", "Emery", "Finley", "Parker", "Rowan"
    ]
    
    @classmethod
    def create_agent(
        cls,
        agent_type: str = "neutral",
        llm_service: Optional[LLMService] = None,
        custom_name: Optional[str] = None,
        custom_stance: Optional[AgentStance] = None
    ) -> Agent:
        """Create an agent of specified type"""
        template = cls.PERSONA_TEMPLATES.get(agent_type, cls.PERSONA_TEMPLATES["neutral"])
        
        agent_id = str(uuid.uuid4())[:8]
        name = custom_name or random.choice(cls.NAMES) + str(random.randint(100, 999))
        
        agent = Agent(
            agent_id=agent_id,
            name=name,
            persona=template["persona"],
            stance=custom_stance if custom_stance else template["stance"],
            trust_level=template.get("trust_level", 0.5),
            influence_score=template.get("influence_score", 0.5),
            susceptibility=template.get("susceptibility", 0.5)
        )
        
        if llm_service:
            agent.set_llm_service(llm_service)
        
        return agent
    
    @classmethod
    def create_population(
        cls,
        n_agents: int,
        distribution: Dict[str, float] = None,
        llm_service: Optional[LLMService] = None
    ) -> List[Agent]:
        """Create a population of agents with specified distribution"""
        if distribution is None:
            distribution = {
                "neutral": 0.4,
                "enthusiast": 0.2,
                "skeptic": 0.15,
                "fact_checker": 0.05,
                "lurker": 0.1,
                "influencer": 0.05,
                "bridge": 0.05
            }
        
        agents = []
        used_names = set()
        
        for agent_type, proportion in distribution.items():
            count = int(n_agents * proportion)
            for _ in range(count):
                # Get unique name
                name = random.choice(cls.NAMES) + str(random.randint(100, 999))
                while name in used_names:
                    name = random.choice(cls.NAMES) + str(random.randint(100, 999))
                used_names.add(name)
                
                agent = cls.create_agent(
                    agent_type=agent_type,
                    llm_service=llm_service,
                    custom_name=name
                )
                agents.append(agent)
        
        # Fill remaining with neutral agents
        while len(agents) < n_agents:
            name = random.choice(cls.NAMES) + str(random.randint(100, 999))
            while name in used_names:
                name = random.choice(cls.NAMES) + str(random.randint(100, 999))
            used_names.add(name)
            
            agent = cls.create_agent("neutral", llm_service, name)
            agents.append(agent)
        
        return agents[:n_agents]
