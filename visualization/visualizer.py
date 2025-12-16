import json
from typing import List, Dict, Any, Optional, Tuple
import random
import math

# Check available plotting libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class NetworkVisualizer:
    """
    Visualizes social network graphs with various layouts and styles.
    """
    
    @staticmethod
    def create_network_figure(
        nodes: List[str],
        edges: List[Tuple[str, str]],
        node_colors: Dict[str, str] = None,
        node_sizes: Dict[str, float] = None,
        node_labels: Dict[str, str] = None,
        title: str = "Social Network",
        layout: str = "spring",
        width: int = 800,
        height: int = 600
    ) -> Optional[Any]:
        """
        Create interactive network visualization using Plotly.
        
        Args:
            nodes: List of node IDs
            edges: List of (source, target) tuples
            node_colors: Dict mapping node_id to color
            node_sizes: Dict mapping node_id to size
            node_labels: Dict mapping node_id to label text
            title: Chart title
            layout: Layout algorithm (spring, circular, random)
            width: Figure width
            height: Figure height
        """
        if not PLOTLY_AVAILABLE or not NETWORKX_AVAILABLE:
            return None
        
        # Create NetworkX graph
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Calculate layout
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42, k=2/math.sqrt(len(nodes)) if nodes else 1)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai" and len(nodes) < 100:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.random_layout(G, seed=42)
        
        # Extract edge coordinates
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            opacity=0.6
        )
        
        # Extract node coordinates and attributes
        node_x = [pos[node][0] for node in nodes]
        node_y = [pos[node][1] for node in nodes]
        
        colors = [node_colors.get(node, '#1f77b4') if node_colors else '#1f77b4' for node in nodes]
        sizes = [node_sizes.get(node, 20) if node_sizes else 20 for node in nodes]
        labels = [node_labels.get(node, node) if node_labels else node for node in nodes]
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=labels,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                showscale=True,
                colorscale='RdYlGn',
                reversescale=True,
                color=colors,
                size=sizes,
                colorbar=dict(
                    thickness=15,
                    title='Node Status',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=width,
                height=height,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
        )
        
        return fig
    
    @staticmethod
    def create_propagation_animation(
        nodes: List[str],
        edges: List[Tuple[str, str]],
        infection_timeline: List[Dict[str, bool]],
        title: str = "Information Propagation"
    ) -> Optional[Any]:
        """
        Create animated visualization of information spread over time.
        
        Args:
            nodes: List of node IDs
            edges: List of (source, target) tuples
            infection_timeline: List of dicts mapping node_id to infected status per round
        """
        if not PLOTLY_AVAILABLE or not NETWORKX_AVAILABLE:
            return None
        
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        pos = nx.spring_layout(G, seed=42)
        
        frames = []
        
        for round_idx, infected_state in enumerate(infection_timeline):
            colors = ['#ff4444' if infected_state.get(node, False) else '#44aa44' for node in nodes]
            
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=[pos[node][0] for node in nodes],
                        y=[pos[node][1] for node in nodes],
                        mode='markers',
                        marker=dict(color=colors, size=15)
                    )
                ],
                name=str(round_idx)
            )
            frames.append(frame)
        
        # Initial frame
        initial_colors = ['#ff4444' if infection_timeline[0].get(node, False) else '#44aa44' for node in nodes]
        
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=[pos[node][0] for node in nodes],
                    y=[pos[node][1] for node in nodes],
                    mode='markers+text',
                    text=nodes,
                    textposition="top center",
                    marker=dict(color=initial_colors, size=15)
                )
            ],
            layout=go.Layout(
                title=title,
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(label="Play",
                                 method="animate",
                                 args=[None, {"frame": {"duration": 500}}])
                        ]
                    )
                ]
            ),
            frames=frames
        )
        
        return fig


class ChartVisualizer:
    """
    Creates various charts for simulation metrics.
    """
    
    @staticmethod
    def create_coverage_chart(
        coverage_data: List[float],
        title: str = "Information Coverage Over Time"
    ) -> Optional[Any]:
        """Create line chart showing coverage over rounds"""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(coverage_data) + 1)),
            y=coverage_data,
            mode='lines+markers',
            name='Coverage',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Round",
            yaxis_title="Coverage (%)",
            yaxis=dict(tickformat='.0%', range=[0, 1]),
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_stance_distribution_chart(
        stance_data: Dict[str, int],
        title: str = "Stance Distribution"
    ) -> Optional[Any]:
        """Create bar chart showing stance distribution"""
        if not PLOTLY_AVAILABLE:
            return None
        
        # Define colors for different stances
        stance_colors = {
            'STRONG_SUPPORT': '#27ae60',
            'SUPPORT': '#2ecc71',
            'NEUTRAL': '#95a5a6',
            'OPPOSE': '#e74c3c',
            'STRONG_OPPOSE': '#c0392b'
        }
        
        stances = list(stance_data.keys())
        values = list(stance_data.values())
        colors = [stance_colors.get(s, '#3498db') for s in stances]
        
        fig = go.Figure(data=[
            go.Bar(
                x=stances,
                y=values,
                marker_color=colors,
                text=values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Stance",
            yaxis_title="Number of Agents",
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_stance_evolution_chart(
        stance_history: List[Dict[str, int]],
        title: str = "Stance Evolution Over Time"
    ) -> Optional[Any]:
        """Create stacked area chart showing stance evolution"""
        if not PLOTLY_AVAILABLE or not stance_history:
            return None
        
        # Get all stance types
        all_stances = set()
        for sh in stance_history:
            all_stances.update(sh.keys())
        
        stance_colors = {
            'STRONG_SUPPORT': '#27ae60',
            'SUPPORT': '#2ecc71',
            'NEUTRAL': '#95a5a6',
            'OPPOSE': '#e74c3c',
            'STRONG_OPPOSE': '#c0392b'
        }
        
        fig = go.Figure()
        
        for stance in sorted(all_stances):
            values = [sh.get(stance, 0) for sh in stance_history]
            fig.add_trace(go.Scatter(
                x=list(range(1, len(stance_history) + 1)),
                y=values,
                mode='lines',
                name=stance,
                stackgroup='one',
                line=dict(color=stance_colors.get(stance, '#3498db'))
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Round",
            yaxis_title="Number of Agents",
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_polarization_chart(
        polarization_data: List[float],
        title: str = "Polarization Index Over Time"
    ) -> Optional[Any]:
        """Create line chart showing polarization index"""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(polarization_data) + 1)),
            y=polarization_data,
            mode='lines+markers',
            name='Polarization',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Round",
            yaxis_title="Polarization Index",
            yaxis=dict(range=[0, 1]),
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_metrics_dashboard(
        metrics: Dict[str, Any],
        title: str = "Simulation Metrics Dashboard"
    ) -> Optional[Any]:
        """Create multi-panel dashboard with key metrics"""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Coverage Over Time',
                'Stance Distribution',
                'Messages & Shares',
                'Agent Activity'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "pie"}]
            ]
        )
        
        # Coverage over time
        if 'coverage_over_time' in metrics:
            fig.add_trace(
                go.Scatter(
                    y=metrics['coverage_over_time'],
                    mode='lines+markers',
                    name='Coverage'
                ),
                row=1, col=1
            )
        
        # Stance distribution
        if 'final_distribution' in metrics:
            dist = metrics['final_distribution']
            fig.add_trace(
                go.Bar(
                    x=list(dist.keys()),
                    y=list(dist.values()),
                    name='Stances'
                ),
                row=1, col=2
            )
        
        # Messages & Shares
        fig.add_trace(
            go.Bar(
                x=['Messages', 'Shares'],
                y=[
                    metrics.get('messages_generated', 0),
                    metrics.get('total_shares', 0)
                ],
                name='Activity'
            ),
            row=2, col=1
        )
        
        # Agent participation pie
        infected = metrics.get('infected_count', 0)
        total = metrics.get('total_agents', 1)
        fig.add_trace(
            go.Pie(
                labels=['Infected', 'Not Infected'],
                values=[infected, total - infected],
                name='Participation'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=700,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_agent_influence_chart(
        agents_data: List[Dict],
        title: str = "Agent Influence Distribution"
    ) -> Optional[Any]:
        """Create scatter plot of agent influence vs followers"""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        # Color by stance
        stance_colors = {
            'STRONG_SUPPORT': '#27ae60',
            'SUPPORT': '#2ecc71',
            'NEUTRAL': '#95a5a6',
            'OPPOSE': '#e74c3c',
            'STRONG_OPPOSE': '#c0392b'
        }
        
        colors = [stance_colors.get(a['stance'], '#3498db') for a in agents_data]
        
        fig.add_trace(go.Scatter(
            x=[a.get('followers', 0) for a in agents_data],
            y=[a.get('influence', 0) for a in agents_data],
            mode='markers',
            marker=dict(
                size=[max(8, a.get('messages_posted', 0) * 3) for a in agents_data],
                color=colors,
                opacity=0.7
            ),
            text=[a.get('name', '') for a in agents_data],
            hoverinfo='text+x+y'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Followers",
            yaxis_title="Influence Score",
            template='plotly_white',
            height=400
        )
        
        return fig


class ConversationVisualizer:
    """
    Creates conversation thread visualizations.
    """
    
    @staticmethod
    def format_twitter_style(
        messages: List[Dict],
        show_timestamps: bool = True
    ) -> str:
        """Format messages in Twitter-style feed"""
        html_parts = []
        
        for msg in messages:
            timestamp = f"<span style='color: #666; font-size: 0.8em;'>Round {msg.get('round', '?')}</span>" if show_timestamps else ""
            
            html_parts.append(f"""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; background: white;'>
                <div style='font-weight: bold; color: #1da1f2;'>@{msg.get('author', 'unknown')}</div>
                <div style='margin: 10px 0;'>{msg.get('content', '')}</div>
                {timestamp}
            </div>
            """)
        
        return "".join(html_parts)
    
    @staticmethod
    def format_reddit_style(
        messages: List[Dict],
        indent_level: int = 0
    ) -> str:
        """Format messages in Reddit-style comment thread"""
        html_parts = []
        
        for msg in messages:
            indent = indent_level * 20
            
            html_parts.append(f"""
            <div style='margin-left: {indent}px; border-left: 2px solid #ccc; padding-left: 15px; margin: 10px 0;'>
                <div style='color: #666; font-size: 0.9em;'>
                    <span style='color: #ff4500; font-weight: bold;'>{msg.get('author', 'unknown')}</span>
                    • Round {msg.get('round', '?')}
                </div>
                <div style='margin: 5px 0;'>{msg.get('content', '')}</div>
            </div>
            """)
        
        return "".join(html_parts)


def create_pyvis_network(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    node_colors: Dict[str, str] = None,
    node_titles: Dict[str, str] = None,
    output_file: str = "network.html"
) -> str:
    """
    Create interactive network visualization using pyvis.
    Returns path to generated HTML file.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return ""
    
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut()
    
    # Add nodes
    for node in nodes:
        color = node_colors.get(node, "#97c2fc") if node_colors else "#97c2fc"
        title = node_titles.get(node, node) if node_titles else node
        net.add_node(node, label=node, color=color, title=title)
    
    # Add edges
    for source, target in edges:
        net.add_edge(source, target)
    
    net.save_graph(output_file)
    return output_file
