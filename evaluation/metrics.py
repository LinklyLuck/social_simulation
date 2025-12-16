from typing import List, Dict, Any, Optional
import math

class MetricsCalculator:
    """
    Calculates various metrics for simulation evaluation.
    """
    
    @staticmethod
    def coverage_metrics(infected_count: int, total_agents: int) -> Dict[str, float]:
        """Calculate coverage-related metrics"""
        coverage = infected_count / total_agents if total_agents > 0 else 0
        return {
            "coverage": coverage,
            "coverage_percent": coverage * 100,
            "uninfected_count": total_agents - infected_count,
            "uninfected_ratio": (total_agents - infected_count) / total_agents if total_agents > 0 else 0
        }
    
    @staticmethod
    def spread_velocity(coverage_timeline: List[float], target_coverage: float = 0.5) -> Dict[str, Any]:
        """Calculate how fast information spreads"""
        rounds_to_target = None
        for i, coverage in enumerate(coverage_timeline):
            if coverage >= target_coverage:
                rounds_to_target = i + 1
                break
        
        # Calculate average velocity
        if len(coverage_timeline) > 1:
            avg_velocity = (coverage_timeline[-1] - coverage_timeline[0]) / len(coverage_timeline)
        else:
            avg_velocity = 0
        
        return {
            "rounds_to_50_percent": rounds_to_target,
            "average_velocity": avg_velocity,
            "max_single_round_spread": max(
                coverage_timeline[i] - coverage_timeline[i-1] 
                for i in range(1, len(coverage_timeline))
            ) if len(coverage_timeline) > 1 else 0
        }
    
    @staticmethod
    def polarization_index(stance_values: List[int]) -> float:
        """
        Calculate polarization index based on stance distribution.
        Returns value between 0 (consensus) and 1 (maximum polarization).
        """
        if not stance_values:
            return 0.0
        
        mean = sum(stance_values) / len(stance_values)
        variance = sum((v - mean) ** 2 for v in stance_values) / len(stance_values)
        
        # Normalize to 0-1 (max variance for values -2 to 2 is 4)
        return min(variance / 4, 1.0)
    
    @staticmethod
    def stance_change_metrics(agents_data: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics related to stance changes"""
        total_changes = 0
        agents_changed = 0
        
        for agent in agents_data:
            history = agent.get('stance_history', [])
            if history:
                total_changes += len(history)
                agents_changed += 1
        
        return {
            "total_stance_changes": total_changes,
            "agents_who_changed": agents_changed,
            "change_ratio": agents_changed / len(agents_data) if agents_data else 0,
            "avg_changes_per_agent": total_changes / len(agents_data) if agents_data else 0
        }
    
    @staticmethod
    def rumor_detection_metrics(
        agents_data: List[Dict],
        ground_truth: bool = True  # True if message was indeed a rumor
    ) -> Dict[str, float]:
        """Calculate rumor detection accuracy metrics"""
        # Agents with negative stance are considered to have detected the rumor
        detected_rumor = len([a for a in agents_data if a.get('stance_value', 0) < 0])
        believed_rumor = len([a for a in agents_data if a.get('stance_value', 0) > 0])
        neutral = len([a for a in agents_data if a.get('stance_value', 0) == 0])
        total = len(agents_data)
        
        if ground_truth:  # Message was a rumor
            true_positives = detected_rumor
            false_negatives = believed_rumor + neutral
        else:  # Message was true
            true_positives = believed_rumor
            false_negatives = detected_rumor + neutral
        
        precision = true_positives / (true_positives + believed_rumor) if (true_positives + believed_rumor) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "detection_accuracy": detected_rumor / total if total > 0 else 0,
            "false_belief_rate": believed_rumor / total if total > 0 else 0,
            "neutral_rate": neutral / total if total > 0 else 0,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    @staticmethod
    def network_influence_metrics(agents_data: List[Dict]) -> Dict[str, Any]:
        """Calculate influence distribution metrics"""
        influences = [a.get('influence', 0) for a in agents_data]
        followers = [a.get('followers', 0) for a in agents_data]
        
        if not influences:
            return {}
        
        # Gini coefficient for influence inequality
        sorted_influences = sorted(influences)
        n = len(sorted_influences)
        cumsum = sum((i + 1) * x for i, x in enumerate(sorted_influences))
        gini = (2 * cumsum) / (n * sum(sorted_influences)) - (n + 1) / n if sum(sorted_influences) > 0 else 0
        
        return {
            "avg_influence": sum(influences) / len(influences),
            "max_influence": max(influences),
            "min_influence": min(influences),
            "influence_gini": gini,
            "avg_followers": sum(followers) / len(followers) if followers else 0,
            "max_followers": max(followers) if followers else 0
        }
    
    @staticmethod
    def collaboration_metrics(
        info_coverage: float,
        rounds_used: int,
        messages_exchanged: int
    ) -> Dict[str, float]:
        """Calculate collaboration efficiency metrics"""
        efficiency = info_coverage / rounds_used if rounds_used > 0 else 0
        msg_efficiency = info_coverage / messages_exchanged if messages_exchanged > 0 else 0
        
        return {
            "info_coverage": info_coverage,
            "rounds_efficiency": efficiency,
            "message_efficiency": msg_efficiency,
            "total_messages": messages_exchanged
        }
    
    @staticmethod
    def community_metrics(
        community_stances: Dict[int, List[int]]
    ) -> Dict[str, Any]:
        """Calculate metrics for community-based analysis"""
        community_means = {}
        community_variances = {}
        
        for comm_id, stances in community_stances.items():
            if stances:
                mean = sum(stances) / len(stances)
                variance = sum((s - mean) ** 2 for s in stances) / len(stances)
                community_means[comm_id] = mean
                community_variances[comm_id] = variance
        
        # Inter-community distance
        means_list = list(community_means.values())
        if len(means_list) >= 2:
            inter_distance = max(means_list) - min(means_list)
        else:
            inter_distance = 0
        
        # Average intra-community variance
        avg_intra_variance = sum(community_variances.values()) / len(community_variances) if community_variances else 0
        
        return {
            "community_means": community_means,
            "community_variances": community_variances,
            "inter_community_distance": inter_distance,
            "avg_intra_variance": avg_intra_variance,
            "echo_chamber_index": inter_distance / (1 + avg_intra_variance) if (1 + avg_intra_variance) > 0 else 0
        }


class ReportGenerator:
    """
    Generates comprehensive reports from simulation results.
    """
    
    @staticmethod
    def generate_summary(results: Dict[str, Any]) -> str:
        """Generate text summary of simulation results"""
        summary_parts = []
        
        final_state = results.get('final_state', {})
        
        summary_parts.append(f"# Simulation Summary\n")
        summary_parts.append(f"## Overview")
        summary_parts.append(f"- Rounds completed: {final_state.get('rounds_completed', 'N/A')}")
        summary_parts.append(f"- Total messages: {final_state.get('total_messages', 'N/A')}")
        summary_parts.append(f"- Total shares: {final_state.get('total_shares', 'N/A')}")
        summary_parts.append(f"- Final coverage: {final_state.get('coverage', 0):.2%}")
        
        stance_dist = final_state.get('stance_distribution', {})
        if stance_dist:
            summary_parts.append(f"\n## Stance Distribution")
            for stance, count in stance_dist.items():
                summary_parts.append(f"- {stance}: {count}")
        
        network_info = results.get('network', {})
        if network_info:
            summary_parts.append(f"\n## Network Information")
            summary_parts.append(f"- Nodes: {network_info.get('nodes', 'N/A')}")
            summary_parts.append(f"- Edges: {network_info.get('edges', 'N/A')}")
            summary_parts.append(f"- Type: {network_info.get('type', 'N/A')}")
        
        return "\n".join(summary_parts)
    
    @staticmethod
    def generate_metrics_table(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert metrics dict to table format"""
        table_data = []
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                table_data.append({"Metric": key, "Value": formatted_value})
        
        return table_data
