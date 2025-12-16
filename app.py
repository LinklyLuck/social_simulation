import streamlit as st
import random
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import simulation components
from agents.agent import Agent, AgentFactory, Message
from agents.llm_service import LLMService
from config import AgentStance
from simulation.manager import SimulationManager
from simulation.network import NetworkGenerator, NetworkType, SocialNetwork
from tasks.tasks import (
    TaskType, TaskFactory, InfoDiffusionTask, RumorDetectionTask,
    StanceEvolutionTask, MultiRoleCollabTask, GroupPolarizationTask
)
from visualization.visualizer import NetworkVisualizer, ChartVisualizer, ConversationVisualizer
from evaluation.metrics import MetricsCalculator, ReportGenerator
from data.loader import DataLoader, SyntheticDataGenerator, create_sample_data

# Page configuration
st.set_page_config(
    page_title="Social Simulation Platform",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #333;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
    .status-running {
        color: #27ae60;
        font-weight: bold;
    }
    .status-completed {
        color: #3498db;
        font-weight: bold;
    }
    .agent-card {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    .log-entry {
        font-family: monospace;
        font-size: 0.85rem;
        padding: 5px;
        border-left: 3px solid #3498db;
        margin: 5px 0;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'simulation_manager' not in st.session_state:
        st.session_state.simulation_manager = None
    if 'llm_service' not in st.session_state:
        st.session_state.llm_service = LLMService(preferred_client="mock")
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'current_task' not in st.session_state:
        st.session_state.current_task = None
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False

init_session_state()

# Sidebar Configuration
def render_sidebar():
    """Render sidebar with configuration options"""
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Platform Mode Selection
    st.sidebar.markdown("### 📱 Platform Mode")
    platform_mode = st.sidebar.selectbox(
        "Simulation Style",
        ["Twitter", "Reddit", "Weibo", "Generic"],
        help="Choose the social platform style for simulation"
    )
    
    # Task Selection
    st.sidebar.markdown("### 🎯 Task Type")
    task_options = {
        "Information Diffusion": TaskType.INFO_DIFFUSION,
        "Rumor Detection": TaskType.RUMOR_DETECTION,
        "Stance Evolution": TaskType.STANCE_EVOLUTION,
        "Multi-Role Collaboration": TaskType.MULTI_ROLE_COLLAB,
        "Group Polarization": TaskType.GROUP_POLARIZATION,
    }
    task_name = st.sidebar.selectbox(
        "Select Task",
        list(task_options.keys()),
        help="Choose the type of social simulation task"
    )
    task_type = task_options[task_name]
    
    # Network Configuration
    st.sidebar.markdown("### 🕸️ Network Settings")
    
    # Data Source Selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Generated Network", "Kaggle Dataset", "Upload File"],
        help="Choose where to get network/content data"
    )
    
    # Initialize all variables with defaults
    kaggle_dataset = None
    kaggle_username = None
    kaggle_key = None
    sample_size = 5000
    max_edges = 50000
    uploaded_file = None
    
    if data_source == "Kaggle Dataset":
        with st.sidebar.expander("🔑 Kaggle API Configuration", expanded=True):
            kaggle_username = st.text_input("Kaggle Username", key="kaggle_user")
            kaggle_key = st.text_input("Kaggle API Key", type="password", key="kaggle_key")
            
            if kaggle_username and kaggle_key:
                st.success("✅ Credentials provided")
                
                # Popular datasets for social network analysis (verified on Kaggle)
                st.markdown("**Popular Datasets:**")
                popular_datasets = {
                    "Twitter Sentiment (1.6M tweets)": "kazanova/sentiment140",
                    "Fake News Dataset": "clmentbisaillon/fake-and-real-news-dataset",
                    "Twitter User Gender": "crowdflower/twitter-user-gender-classification",
                    "Facebook Social Network": "ashwinpathak/facebook-social-network",
                    "Social Network Edges": "mathurinache/twitter-edge-nodes",
                    "Custom...": "custom"
                }
                dataset_choice = st.selectbox("Select Dataset", list(popular_datasets.keys()))
                
                if dataset_choice == "Custom...":
                    kaggle_dataset = st.text_input(
                        "Dataset Reference",
                        placeholder="owner/dataset-name",
                        help="Enter Kaggle dataset reference (e.g., 'kazanova/sentiment140')"
                    )
                else:
                    kaggle_dataset = popular_datasets[dataset_choice]
                
                sample_size = st.number_input("Sample Size (0=all)", 0, 100000, 5000)
                max_edges = st.number_input("Max Edges (prevents large dataset issues)", 1000, 100000, 50000)
            else:
                st.info("Enter Kaggle credentials to load datasets")
    
    elif data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Network/Content CSV",
            type=['csv', 'json'],
            help="Upload CSV with 'source,target' columns for network, or 'text,user' for content"
        )
    
    # Network type (for generated networks)
    if data_source == "Generated Network":
        network_options = {
            "Barabási-Albert (Scale-Free)": NetworkType.BARABASI_ALBERT,
            "Watts-Strogatz (Small-World)": NetworkType.WATTS_STROGATZ,
            "Erdős-Rényi (Random)": NetworkType.ERDOS_RENYI,
        }
        network_name = st.sidebar.selectbox(
            "Network Type",
            list(network_options.keys())
        )
        network_type = network_options[network_name]
    else:
        network_type = NetworkType.CUSTOM
    
    num_agents = st.sidebar.slider("Number of Agents", 5, 100, 20, 5)
    
    # Simulation Parameters
    st.sidebar.markdown("### 🔄 Simulation Parameters")
    num_rounds = st.sidebar.slider("Number of Rounds", 1, 30, 10)
    infection_prob = st.sidebar.slider("Infection Probability", 0.1, 1.0, 0.3, 0.1)
    random_seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)
    
    # Advanced Options
    with st.sidebar.expander("🔧 Advanced Options"):
        enable_persona = st.checkbox("Enable Agent Personas", True)
        enable_topic_drift = st.checkbox("Enable Topic Drift", False)
        trust_weight = st.slider("Trust Weight", 0.0, 1.0, 0.5, 0.1)
    
    # LLM Configuration
    st.sidebar.markdown("### 🤖 LLM Settings")
    llm_client = st.sidebar.selectbox(
        "LLM Client",
        ["Mock (No API)", "OpenAI", "DeepSeek", "Anthropic"],
        help="Select LLM backend (Mock for testing without API)"
    )
    
    # API Key input for real LLM clients
    api_key = None
    api_base_url = None
    llm_model = None
    
    if llm_client != "Mock (No API)":
        with st.sidebar.expander("🔑 API Configuration", expanded=True):
            api_key = st.text_input(
                f"{llm_client} API Key",
                type="password",
                help=f"Enter your {llm_client} API key"
            )
            
            if llm_client == "DeepSeek":
                api_base_url = st.text_input(
                    "API Base URL",
                    value="https://api.deepseek.com/v1",
                    help="DeepSeek API endpoint"
                )
                llm_model = st.selectbox(
                    "Model",
                    ["deepseek-chat", "deepseek-coder"],
                    help="Select DeepSeek model"
                )
            elif llm_client == "OpenAI":
                # Support OpenAI-compatible APIs
                use_custom_endpoint = st.checkbox("Use Custom Endpoint", False)
                if use_custom_endpoint:
                    api_base_url = st.text_input(
                        "API Base URL",
                        placeholder="https://api.openai.com/v1",
                        help="Custom OpenAI-compatible API endpoint"
                    )
                llm_model = st.selectbox(
                    "Model",
                    ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                    help="Select OpenAI model"
                )
            elif llm_client == "Anthropic":
                llm_model = st.selectbox(
                    "Model",
                    ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
                    help="Select Anthropic model"
                )
            
            if api_key:
                st.success("✅ API Key provided")
            else:
                st.warning("⚠️ Enter API Key to use this LLM")
    
    return {
        "platform_mode": platform_mode.lower(),
        "task_type": task_type,
        "network_type": network_type,
        "num_agents": num_agents,
        "num_rounds": num_rounds,
        "infection_prob": infection_prob,
        "random_seed": random_seed,
        "enable_persona": enable_persona,
        "enable_topic_drift": enable_topic_drift,
        "trust_weight": trust_weight,
        "llm_client": llm_client.lower().split()[0],
        "api_key": api_key,
        "api_base_url": api_base_url,
        "llm_model": llm_model,
        # Data source options
        "data_source": data_source,
        "kaggle_dataset": kaggle_dataset if data_source == "Kaggle Dataset" else None,
        "kaggle_username": kaggle_username if data_source == "Kaggle Dataset" else None,
        "kaggle_key": kaggle_key if data_source == "Kaggle Dataset" else None,
        "kaggle_sample_size": sample_size if data_source == "Kaggle Dataset" else None,
        "kaggle_max_edges": max_edges if data_source == "Kaggle Dataset" else None,
        "uploaded_file": uploaded_file if data_source == "Upload File" else None,
    }

def render_task_config(task_type: TaskType):
    """Render task-specific configuration options"""
    st.markdown("### 📋 Task Configuration")
    
    config = {}
    
    if task_type == TaskType.INFO_DIFFUSION:
        col1, col2 = st.columns(2)
        with col1:
            config["message_content"] = st.text_area(
                "Initial Message",
                "Breaking news: Major announcement expected today!",
                height=100
            )
            config["source_count"] = st.number_input("Number of Sources", 1, 5, 1)
        with col2:
            config["enable_intervention"] = st.checkbox("Enable Intervention", False)
            if config["enable_intervention"]:
                config["intervention_round"] = st.number_input("Intervention Round", 1, 20, 5)
                config["intervention_type"] = st.selectbox(
                    "Intervention Type",
                    ["counter_message", "reduce_spread"]
                )
    
    elif task_type == TaskType.RUMOR_DETECTION:
        col1, col2 = st.columns(2)
        with col1:
            config["rumor_content"] = st.text_area(
                "Rumor Message",
                "ALERT: Unverified reports suggest company bankruptcy is imminent!",
                height=100
            )
        with col2:
            config["truth_content"] = st.text_area(
                "Truth/Counter Message",
                "FACT CHECK: The bankruptcy rumors are FALSE. Official statements confirm company stability. ✅",
                height=100
            )
        config["fact_checker_ratio"] = st.slider("Fact-Checker Ratio", 0.0, 0.3, 0.1, 0.05)
        config["rumor_delay_rounds"] = st.number_input("Rounds Before Counter-Message", 1, 10, 3)
    
    elif task_type == TaskType.STANCE_EVOLUTION:
        config["topic"] = st.text_input(
            "Debate Topic",
            "AI regulation should be significantly stricter"
        )
        col1, col2 = st.columns(2)
        with col1:
            config["initial_support_ratio"] = st.slider("Initial Support Ratio", 0.0, 1.0, 0.4, 0.1)
        with col2:
            config["initial_oppose_ratio"] = st.slider("Initial Oppose Ratio", 0.0, 1.0, 0.4, 0.1)
    
    elif task_type == TaskType.MULTI_ROLE_COLLAB:
        st.info("💡 Each agent will receive partial information. They must collaborate to solve the puzzle.")
        puzzle = SyntheticDataGenerator.generate_collaboration_puzzle()
        config["puzzle_type"] = "mystery"
        config["clues"] = puzzle["clues"]
        config["answer"] = puzzle["answer"]
        
        with st.expander("View Puzzle Details"):
            st.write(f"**Puzzle:** {puzzle['title']}")
            st.write("**Clues:**")
            for i, clue in enumerate(puzzle['clues'], 1):
                st.write(f"  {i}. {clue}")
            st.write(f"**Answer:** {puzzle['answer']}")
    
    elif task_type == TaskType.GROUP_POLARIZATION:
        config["n_communities"] = st.number_input("Number of Communities", 2, 5, 2)
        config["topic"] = st.text_input(
            "Polarizing Topic",
            "Climate change requires immediate drastic action"
        )
        col1, col2 = st.columns(2)
        with col1:
            config["intra_connection_prob"] = st.slider("Intra-Community Connection", 0.1, 0.8, 0.4, 0.1)
        with col2:
            config["inter_connection_prob"] = st.slider("Inter-Community Connection", 0.0, 0.3, 0.05, 0.01)
    
    return config

def run_simulation(config: dict, task_config: dict):
    """Run the simulation with given configuration"""
    with st.spinner("🔄 Initializing simulation..."):
        # Update LLM service with API key and configuration
        st.session_state.llm_service = LLMService(
            preferred_client=config["llm_client"],
            api_key=config.get("api_key"),
            api_base_url=config.get("api_base_url"),
            model=config.get("llm_model")
        )
        
        # Show LLM status
        available = st.session_state.llm_service.available_clients()
        if config["llm_client"] != "mock" and config["llm_client"] in available:
            st.success(f"✅ Using {config['llm_client'].upper()} LLM")
        elif config["llm_client"] != "mock":
            st.warning(f"⚠️ {config['llm_client'].upper()} not available, using Mock LLM")
        
        # Create simulation manager
        manager = SimulationManager(llm_service=st.session_state.llm_service)
        
        # Check data source
        data_source = config.get("data_source", "Generated Network")
        
        if data_source == "Kaggle Dataset" and config.get("kaggle_dataset"):
            # Load from Kaggle
            try:
                from data.loader import KaggleLoader
                
                loader = KaggleLoader()
                
                # Configure credentials
                if not loader.configure_api(
                    username=config["kaggle_username"],
                    key=config["kaggle_key"]
                ):
                    st.error("Failed to configure Kaggle API credentials")
                    raise Exception("Kaggle auth failed")
                
                st.info(f"📥 Downloading dataset: {config['kaggle_dataset']}...")
                
                # Download dataset
                data_path = loader.download_dataset(config["kaggle_dataset"])
                if not data_path:
                    raise Exception("Download failed")
                
                # Convert dataset_id to folder name (kaggle uses owner-dataset format)
                dataset_folder = config["kaggle_dataset"].replace("/", "-")
                
                # Import to simulation format
                dataset_info = loader.import_to_simulation(
                    dataset_name=dataset_folder,
                    max_nodes=config["num_agents"],
                    max_edges=config.get("kaggle_max_edges", 50000),
                    max_content=1000
                )
                
                if dataset_info and dataset_info.get("nodes"):
                    # Network dataset - create agents from nodes
                    from agents.agent import AgentFactory
                    from simulation.network import SocialNetwork
                    
                    nodes = dataset_info["nodes"][:config["num_agents"]]
                    edges = dataset_info.get("edges", [])
                    
                    manager.agents = AgentFactory.create_population(
                        n_agents=len(nodes),
                        llm_service=manager.llm_service
                    )
                    
                    # Build network
                    manager.network = SocialNetwork()
                    node_map = {old: agent.agent_id for old, agent in zip(nodes, manager.agents)}
                    
                    for agent in manager.agents:
                        manager.network.add_node(agent.agent_id)
                    
                    for src, dst in edges:
                        if src in node_map and dst in node_map:
                            manager.network.add_edge(node_map[src], node_map[dst])
                    
                    manager._index_agents()
                    manager.platform_mode = config.get("platform_mode", "generic")
                    manager.trust_weight = config.get("trust_weight", 0.5)
                    
                    # Setup trust networks
                    import random
                    for agent in manager.agents:
                        following = manager.network.get_following(agent.agent_id)
                        for fid in following:
                            agent.trust_network[fid] = 0.5 + 0.3 * random.random()
                    
                    st.success(f"✅ Loaded {len(manager.agents)} agents, {len(edges)} edges from Kaggle")
                else:
                    # Content dataset or failed - fallback to generated network
                    st.warning("Dataset doesn't contain network structure, using generated network")
                    manager.setup(
                        n_agents=config["num_agents"],
                        network_type=config["network_type"],
                        seed=config["random_seed"],
                        platform_mode=config.get("platform_mode", "generic"),
                        trust_weight=config.get("trust_weight", 0.5)
                    )
                    
                    # Inject content from dataset as seed messages
                    if dataset_info and dataset_info.get("content"):
                        import random
                        sample_content = random.sample(
                            dataset_info["content"], 
                            min(3, len(dataset_info["content"]))
                        )
                        for item in sample_content:
                            text = item.get("text", item.get("content", ""))[:500]
                            if text and manager.agents:
                                src = random.choice(manager.agents)
                                manager.inject_message(text, [src.agent_id])
                
            except ImportError:
                st.error("Kaggle package not installed. Run: pip install kaggle")
                return None
            except Exception as e:
                st.error(f"Failed to load Kaggle dataset: {e}")
                # Fallback to generated network
                manager.setup(
                    n_agents=config["num_agents"],
                    network_type=config["network_type"],
                    seed=config["random_seed"],
                    platform_mode=config.get("platform_mode", "generic"),
                    trust_weight=config.get("trust_weight", 0.5)
                )
        
        elif data_source == "Upload File" and config.get("uploaded_file"):
            # Load from uploaded file
            try:
                import tempfile
                import pandas as pd
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    tmp.write(config["uploaded_file"].getvalue())
                    tmp_path = tmp.name
                
                # Try to parse the file
                df = pd.read_csv(tmp_path)
                cols_lower = [c.lower() for c in df.columns]
                
                # Check if network or content
                if 'source' in cols_lower and 'target' in cols_lower:
                    # Network format
                    from agents.agent import AgentFactory
                    from simulation.network import SocialNetwork
                    
                    src_col = df.columns[cols_lower.index('source')]
                    tgt_col = df.columns[cols_lower.index('target')]
                    
                    edges = list(zip(df[src_col].astype(str), df[tgt_col].astype(str)))
                    nodes = list(set(df[src_col].astype(str)) | set(df[tgt_col].astype(str)))
                    nodes = nodes[:config["num_agents"]]
                    
                    manager.agents = AgentFactory.create_population(
                        n_agents=len(nodes),
                        llm_service=manager.llm_service
                    )
                    
                    manager.network = SocialNetwork()
                    node_map = {old: agent.agent_id for old, agent in zip(nodes, manager.agents)}
                    
                    for agent in manager.agents:
                        manager.network.add_node(agent.agent_id)
                    
                    for src, dst in edges:
                        if src in node_map and dst in node_map:
                            manager.network.add_edge(node_map[src], node_map[dst])
                    
                    manager._index_agents()
                    manager.platform_mode = config.get("platform_mode", "generic")
                    manager.trust_weight = config.get("trust_weight", 0.5)
                    
                    import random
                    for agent in manager.agents:
                        following = manager.network.get_following(agent.agent_id)
                        for fid in following:
                            agent.trust_network[fid] = 0.5 + 0.3 * random.random()
                    
                    st.success(f"✅ Loaded {len(manager.agents)} agents from uploaded network")
                else:
                    # Content format - use generated network but inject content
                    manager.setup(
                        n_agents=config["num_agents"],
                        network_type=config["network_type"],
                        seed=config["random_seed"],
                        platform_mode=config.get("platform_mode", "generic"),
                        trust_weight=config.get("trust_weight", 0.5)
                    )
                    
                    # Find text column
                    text_col = None
                    for c in df.columns:
                        if any(x in c.lower() for x in ['text', 'content', 'body', 'message']):
                            text_col = c
                            break
                    
                    if text_col:
                        import random
                        sample_texts = df[text_col].dropna().sample(min(3, len(df))).tolist()
                        for text in sample_texts:
                            if text and manager.agents:
                                src = random.choice(manager.agents)
                                manager.inject_message(str(text)[:500], [src.agent_id])
                        st.success(f"✅ Created network and injected {len(sample_texts)} seed messages")
                    else:
                        st.success(f"✅ Created network with {len(manager.agents)} agents")
                
                # Cleanup temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Failed to load uploaded file: {e}")
                # Fallback
                manager.setup(
                    n_agents=config["num_agents"],
                    network_type=config["network_type"],
                    seed=config["random_seed"],
                    platform_mode=config.get("platform_mode", "generic"),
                    trust_weight=config.get("trust_weight", 0.5)
                )
        
        else:
            # Generated network (default)
            manager.setup(
                n_agents=config["num_agents"],
                network_type=config["network_type"],
                seed=config["random_seed"],
                platform_mode=config.get("platform_mode", "generic"),
                enable_persona=config.get("enable_persona", True),
                enable_topic_drift=config.get("enable_topic_drift", False),
                trust_weight=config.get("trust_weight", 0.5)
            )
        
        st.session_state.simulation_manager = manager
        
        # Create and setup task
        task = TaskFactory.create(config["task_type"], manager)
        task.setup(**task_config)
        st.session_state.current_task = task
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run simulation via task.run() - this ensures task-specific logic is executed
    st.session_state.is_running = True
    status_text.text("Running simulation...")
    
    # Use task.run() to execute the simulation with task-specific logic
    task_result = task.run(
        n_rounds=config["num_rounds"],
        infection_probability=config["infection_prob"]
    )
    
    # Update progress
    progress_bar.progress(1.0)
    
    # Get results
    eval_result = task.evaluate()
    st.session_state.simulation_results = {
        "task_result": eval_result,
        "manager_results": manager.get_results(),
        "round_data": manager.round_data.copy()
    }
    st.session_state.is_running = False
    
    status_text.text("✅ Simulation completed!")
    return st.session_state.simulation_results

def render_results():
    """Render simulation results and visualizations"""
    if not st.session_state.simulation_results:
        st.info("Run a simulation to see results here.")
        return
    
    results = st.session_state.simulation_results
    manager = st.session_state.simulation_manager
    
    st.markdown("## 📊 Results")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    final_state = results["manager_results"]["final_state"]
    
    with col1:
        st.metric("Rounds", final_state["rounds_completed"])
    with col2:
        st.metric("Coverage", f"{final_state['coverage']:.1%}")
    with col3:
        st.metric("Messages", final_state["total_messages"])
    with col4:
        st.metric("Shares", final_state["total_shares"])
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "🕸️ Network", "👥 Agents", "📝 Logs"])
    
    with tab1:
        render_charts(results)
    
    with tab2:
        render_network(manager)
    
    with tab3:
        render_agents(manager)
    
    with tab4:
        render_logs(manager)

def render_charts(results):
    """Render result charts"""
    col1, col2 = st.columns(2)
    
    round_data = results.get("round_data", [])
    
    with col1:
        # Coverage over time
        if round_data:
            coverage_data = [r.get("coverage", 0) for r in round_data]
            fig = ChartVisualizer.create_coverage_chart(coverage_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(coverage_data)
    
    with col2:
        # Stance distribution
        final_state = results["manager_results"]["final_state"]
        stance_dist = final_state.get("stance_distribution", {})
        if stance_dist:
            fig = ChartVisualizer.create_stance_distribution_chart(stance_dist)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(stance_dist)
    
    # Additional metrics based on task type
    task_result = results.get("task_result", {})
    
    if "polarization_history" in task_result:
        st.markdown("### Polarization Trend")
        fig = ChartVisualizer.create_polarization_chart(task_result["polarization_history"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    if "stance_history" in task_result:
        st.markdown("### Stance Evolution")
        fig = ChartVisualizer.create_stance_evolution_chart(task_result["stance_history"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)

def render_network(manager: SimulationManager):
    """Render network visualization"""
    if not manager or not manager.network:
        st.warning("No network data available.")
        return
    
    network = manager.network
    
    # Prepare node data
    node_colors = {}
    node_sizes = {}
    node_labels = {}
    
    for agent in manager.agents:
        infected = agent.agent_id in manager.state.infected_nodes
        node_colors[agent.agent_id] = '#e74c3c' if infected else '#2ecc71'
        node_sizes[agent.agent_id] = 15 + agent.influence_score * 20
        node_labels[agent.agent_id] = agent.name
    
    # Try NetworkX-based visualization first
    fig = NetworkVisualizer.create_network_figure(
        nodes=network.nodes,
        edges=network.edges,
        node_colors=node_colors,
        node_sizes=node_sizes,
        node_labels=node_labels,
        title="Social Network (Red = Infected, Green = Not Infected)"
    )
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback: Pure Plotly visualization without networkx
        try:
            import plotly.graph_objects as go
            import math
            
            # Simple circular layout
            nodes = list(network.nodes)
            n = len(nodes)
            
            # Calculate positions in a circle
            pos = {}
            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / n
                pos[node] = (math.cos(angle), math.sin(angle))
            
            # Create edge traces
            edge_x = []
            edge_y = []
            for src, tgt in network.edges:
                if src in pos and tgt in pos:
                    x0, y0 = pos[src]
                    x1, y1 = pos[tgt]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create node traces
            node_x = [pos[n][0] for n in nodes]
            node_y = [pos[n][1] for n in nodes]
            colors = [node_colors.get(n, '#2ecc71') for n in nodes]
            sizes = [node_sizes.get(n, 15) for n in nodes]
            labels = [node_labels.get(n, n) for n in nodes]
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=labels,
                textposition="top center",
                textfont=dict(size=8),
                marker=dict(
                    color=colors,
                    size=sizes,
                    line=dict(width=1, color='white')
                )
            )
            
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Social Network (Red = Infected, Green = Not Infected)',
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=500
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.info("Network visualization requires plotly library. Install with: pip install plotly")
        except Exception as e:
            st.warning(f"Network visualization error: {e}")
        
        # Always show network stats
        st.write(f"**Nodes:** {len(network.nodes)}")
        st.write(f"**Edges:** {len(network.edges)}")
        st.write(f"**Network Type:** {network.network_type.value}")

def render_agents(manager: SimulationManager):
    """Render agent information"""
    if not manager or not manager.agents:
        st.warning("No agent data available.")
        return
    
    agents_data = manager.get_agent_states()
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        filter_infected = st.selectbox("Filter by Status", ["All", "Infected", "Not Infected"])
    with col2:
        filter_stance = st.selectbox("Filter by Stance", ["All"] + [s.name for s in AgentStance])
    
    # Apply filters
    filtered_agents = agents_data
    if filter_infected == "Infected":
        filtered_agents = [a for a in filtered_agents if a["infected"]]
    elif filter_infected == "Not Infected":
        filtered_agents = [a for a in filtered_agents if not a["infected"]]
    
    if filter_stance != "All":
        filtered_agents = [a for a in filtered_agents if a["stance"] == filter_stance]
    
    st.write(f"Showing {len(filtered_agents)} of {len(agents_data)} agents")
    
    # Display agents in grid
    cols = st.columns(3)
    for idx, agent in enumerate(filtered_agents):
        with cols[idx % 3]:
            status_color = "🔴" if agent["infected"] else "🟢"
            stance_emoji = {"SUPPORT": "👍", "OPPOSE": "👎", "NEUTRAL": "😐", 
                          "STRONG_SUPPORT": "💪👍", "STRONG_OPPOSE": "💪👎"}.get(agent["stance"], "❓")
            
            st.markdown(f"""
            <div class="agent-card">
                <strong>{agent['name']}</strong> {status_color}<br>
                Stance: {stance_emoji} {agent['stance']}<br>
                Followers: {agent['followers']}<br>
                Influence: {agent['influence']:.2f}
            </div>
            """, unsafe_allow_html=True)

def render_logs(manager: SimulationManager):
    """Render simulation logs"""
    if not manager or not manager.logs:
        st.info("No logs available.")
        return
    
    # Filter options
    log_types = list(set(log.event_type for log in manager.logs))
    selected_types = st.multiselect("Filter by Event Type", log_types, default=log_types)
    
    max_logs = st.slider("Max Logs to Display", 10, 200, 50)
    
    # Filter and display logs
    filtered_logs = [log for log in manager.logs if log.event_type in selected_types]
    
    for log in filtered_logs[-max_logs:]:
        type_emoji = {
            "setup": "⚙️",
            "inject": "💉",
            "reply": "💬",
            "share": "🔄",
            "stance_change": "🔀",
            "round_start": "▶️",
            "round_end": "⏹️",
        }.get(log.event_type, "📝")
        
        st.markdown(f"""
        <div class="log-entry">
            {type_emoji} <strong>Round {log.round_num}</strong> [{log.event_type}]: {log.content[:150]}
        </div>
        """, unsafe_allow_html=True)

def render_data_management():
    """Render data management section with Kaggle API support"""
    st.markdown("## 📁 Data Management")
    
    loader = DataLoader()
    create_sample_data()  # Ensure sample data exists
    
    # Create tabs for different data sources
    data_tab1, data_tab2, data_tab3 = st.tabs(["📂 Local Data", "🏆 Kaggle API", "🔧 Generate Data"])
    
    with data_tab1:
        st.markdown("### Available Local Datasets")
        datasets = loader.list_available_datasets()
        if datasets:
            for ds in datasets:
                st.write(f"📄 {ds}")
        else:
            st.info("No datasets found. Upload data or generate synthetic data.")
        
        # File upload
        st.markdown("### Upload Dataset")
        uploaded_file = st.file_uploader(
            "Upload network or content file",
            type=['csv', 'json', 'tsv'],
            help="Upload a CSV/TSV edge list or JSON dataset"
        )
        
        if uploaded_file:
            file_type = st.radio("File type:", ["Network (edges)", "Content (messages)"])
            if st.button("Import File"):
                try:
                    if file_type == "Network (edges)":
                        save_path = os.path.join(loader.networks_path, uploaded_file.name)
                    else:
                        save_path = os.path.join(loader.content_path, uploaded_file.name)
                    
                    with open(save_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Saved to {save_path}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with data_tab2:
        render_kaggle_section()
    
    with data_tab3:
        st.markdown("### Generate Synthetic Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔗 Generate Sample Network"):
                nodes = [f"user_{i}" for i in range(20)]
                network = NetworkGenerator.generate_barabasi_albert(20, 2, nodes, seed=42)
                loader.save_network(network.nodes, network.edges, "generated_network.json")
                st.success("Generated network saved!")
                st.rerun()
        
        with col2:
            if st.button("💬 Generate Sample Messages"):
                messages = SyntheticDataGenerator.generate_messages(20)
                loader.save_content(messages, "generated_messages.json")
                st.success("Generated messages saved!")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### Custom Generation")
        
        gen_col1, gen_col2 = st.columns(2)
        with gen_col1:
            n_nodes = st.number_input("Number of nodes", 10, 1000, 50)
            network_type = st.selectbox("Network type", ["barabasi_albert", "erdos_renyi", "watts_strogatz"])
        
        with gen_col2:
            n_messages = st.number_input("Number of messages", 10, 500, 50)
            include_rumors = st.checkbox("Include rumor scenarios", True)
        
        if st.button("🎲 Generate Custom Dataset"):
            with st.spinner("Generating..."):
                # Generate network
                nodes = [f"user_{i}" for i in range(n_nodes)]
                if network_type == "barabasi_albert":
                    network = NetworkGenerator.generate_barabasi_albert(n_nodes, 3, nodes, seed=42)
                elif network_type == "erdos_renyi":
                    network = NetworkGenerator.generate_erdos_renyi(n_nodes, 0.1, nodes, seed=42)
                else:
                    network = NetworkGenerator.generate_watts_strogatz(n_nodes, 4, 0.3, nodes, seed=42)
                
                loader.save_network(network.nodes, network.edges, f"custom_{network_type}_{n_nodes}.json")
                
                # Generate messages
                messages = SyntheticDataGenerator.generate_messages(n_messages)
                if include_rumors:
                    rumors = SyntheticDataGenerator.generate_rumor_scenarios(5)
                    for rumor in rumors:
                        messages.append({
                            "content": rumor["rumor"],
                            "is_rumor": True,
                            "truth": rumor["truth"]
                        })
                
                loader.save_content(messages, f"custom_messages_{n_messages}.json")
                
                st.success(f"Generated {n_nodes} nodes, {len(network.edges)} edges, {len(messages)} messages!")
                st.rerun()


def render_kaggle_section():
    """Render Kaggle API configuration and dataset download section"""
    from data.loader import KaggleLoader
    
    kaggle_loader = KaggleLoader()
    
    st.markdown("### 🏆 Kaggle Dataset Integration")
    
    # Check API status
    api_status = kaggle_loader.get_api_status()
    
    if api_status["authenticated"]:
        st.success("✅ Kaggle API connected and authenticated")
    else:
        st.warning("⚠️ Kaggle API not configured")
        
        with st.expander("🔧 Configure Kaggle API"):
            st.markdown("""
            To use Kaggle datasets, you need to:
            1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
            2. Go to Account Settings → API → Create New Token
            3. Enter your credentials below
            """)
            
            username = st.text_input("Kaggle Username")
            api_key = st.text_input("API Key", type="password")
            
            if st.button("Save Credentials"):
                if username and api_key:
                    success = kaggle_loader.configure_api(username, api_key)
                    if success:
                        st.success("✅ Credentials saved and verified!")
                        st.rerun()
                    else:
                        st.error("Failed to authenticate. Check your credentials.")
                else:
                    st.error("Please enter both username and API key")
        
        if api_status["error"]:
            st.info(f"Note: {api_status['error']}")
        
        return
    
    # Search datasets
    st.markdown("### 🔍 Search Datasets")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Search Kaggle", placeholder="e.g., twitter sentiment, social network")
    with col2:
        search_clicked = st.button("🔍 Search")
    
    if search_clicked and search_query:
        with st.spinner("Searching Kaggle..."):
            results = kaggle_loader.search_datasets(search_query, max_results=10)
            
            if results:
                st.session_state.kaggle_search_results = results
            else:
                st.info("No datasets found. Try different keywords.")
    
    # Display search results
    if hasattr(st.session_state, 'kaggle_search_results') and st.session_state.kaggle_search_results:
        st.markdown("### Search Results")
        
        for i, ds in enumerate(st.session_state.kaggle_search_results):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{ds['title']}**")
                    st.caption(f"ID: `{ds['id']}` | Downloads: {ds.get('downloads', 'N/A')}")
                with col2:
                    size_mb = ds.get('size', 0) / (1024 * 1024)
                    st.write(f"{size_mb:.1f} MB")
                with col3:
                    if st.button("⬇️ Download", key=f"dl_{i}"):
                        with st.spinner(f"Downloading {ds['title']}..."):
                            path = kaggle_loader.download_dataset(ds['id'])
                            if path:
                                st.success(f"Downloaded to {path}")
                                st.rerun()
                            else:
                                st.error("Download failed")
    
    # Recommended datasets
    with st.expander("📋 Recommended Datasets for Social Simulation"):
        for name, info in KaggleLoader.RECOMMENDED_DATASETS.items():
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**{name}**")
            with col2:
                st.caption(info["description"])
            with col3:
                if st.button("⬇️", key=f"rec_{name}"):
                    with st.spinner("Downloading..."):
                        path = kaggle_loader.download_dataset(info["id"])
                        if path:
                            st.success("Downloaded!")
                            st.rerun()
    
    # Show downloaded datasets
    st.markdown("### 📥 Downloaded Datasets")
    downloaded = kaggle_loader.list_downloaded_datasets()
    
    if downloaded:
        for ds in downloaded:
            with st.expander(f"📂 {ds['name']} ({ds['file_count']} files)"):
                st.write(f"Path: `{ds['path']}`")
                st.write(f"Total size: {ds['total_size'] / 1024:.1f} KB")
                st.write("Files:", ds['files'][:10])
                
                # Import options
                st.markdown("---")
                st.markdown("**Import to Simulation:**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_nodes = st.number_input("Max nodes", 100, 10000, 500, key=f"mn_{ds['name']}")
                with col2:
                    max_edges = st.number_input("Max edges", 1000, 100000, 50000, key=f"me_{ds['name']}")
                with col3:
                    max_content = st.number_input("Max content", 100, 10000, 500, key=f"mc_{ds['name']}")
                
                if st.button("🚀 Import Dataset", key=f"import_{ds['name']}"):
                    with st.spinner("Importing..."):
                        result = kaggle_loader.import_to_simulation(
                            ds['name'],
                            max_nodes=max_nodes,
                            max_edges=max_edges,
                            max_content=max_content
                        )
                        
                        if "error" not in result:
                            # Save to datalake
                            data_loader = DataLoader()
                            
                            if result["nodes"]:
                                data_loader.save_network(
                                    result["nodes"],
                                    result["edges"],
                                    f"kaggle_{ds['name']}_network.json"
                                )
                                st.success(f"Imported {len(result['nodes'])} nodes, {len(result['edges'])} edges")
                            
                            if result["content"]:
                                data_loader.save_content(
                                    result["content"],
                                    f"kaggle_{ds['name']}_content.json"
                                )
                                st.success(f"Imported {len(result['content'])} content items")
                            
                            st.rerun()
                        else:
                            st.error(f"Import failed: {result.get('error')}")
    else:
        st.info("No Kaggle datasets downloaded yet. Search and download above.")

def main():
    """Main application entry point"""
    # Header
    st.markdown('<h1 class="main-header">🌐 Social Simulation Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">LLM-Powered Multi-Agent Social Activity Simulation</p>', unsafe_allow_html=True)
    
    # Get configuration from sidebar
    config = render_sidebar()
    
    # Main content area
    main_tab1, main_tab2, main_tab3 = st.tabs(["🎮 Simulation", "📊 Results", "📁 Data"])
    
    with main_tab1:
        st.markdown("## 🎮 Run Simulation")
        
        # Task-specific configuration
        task_config = render_task_config(config["task_type"])
        
        # Control buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            run_button = st.button("▶️ Run Simulation", type="primary", use_container_width=True)
        
        with col2:
            reset_button = st.button("🔄 Reset", use_container_width=True)
        
        if run_button:
            results = run_simulation(config, task_config)
            st.success("Simulation completed! View results in the 'Results' tab.")
        
        if reset_button:
            st.session_state.simulation_manager = None
            st.session_state.simulation_results = None
            st.session_state.current_task = None
            st.rerun()
        
        # Quick preview if simulation exists
        if st.session_state.simulation_results:
            st.markdown("---")
            st.markdown("### Quick Preview")
            final_state = st.session_state.simulation_results["manager_results"]["final_state"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Coverage", f"{final_state['coverage']:.1%}")
            with col2:
                st.metric("Messages", final_state["total_messages"])
            with col3:
                st.metric("Rounds", final_state["rounds_completed"])
    
    with main_tab2:
        render_results()
    
    with main_tab3:
        render_data_management()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Social Simulation Platform | Built with Streamlit & Python | 
        <a href='#'>Documentation</a> | <a href='#'>GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
