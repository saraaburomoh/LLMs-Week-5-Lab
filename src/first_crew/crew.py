import os
import json
import yaml
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from crewai_tools import JSONSearchTool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from langchain_huggingface import HuggingFaceEmbeddings
from crewai import LLM

# 1. Bypass the underlying issue in CrewAI-Tools that forcefully checks for an OpenAI Key
os.environ["OPENAI_API_KEY"] = "NA"

# 2. Configure the global Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5'
)

local_llm = LLM(
    model="ollama_chat/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.0
)

# ── Global Knowledge: Yelp field definitions ──────────────────────────────────
yelp_doc_path = "docs/Yelp Data Translation.md"
with open(yelp_doc_path, "r", encoding="utf-8") as f:
    yelp_md = f.read()

yelp_knowledge = StringKnowledgeSource(
    content=yelp_md,
    embedder=embedding_model
)

# 3. RAG Config
rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    }
}

# 4. Initialize RAG Tools
user_rag_tool = JSONSearchTool(
    json_path='data/user_subset.json',
    config=rag_config,
    collection_name='v3_hf_user_data'
)
user_rag_tool.name = "search_user_profile_data"
user_rag_tool.description = "Useful to retrieve a specific user's giving habits, average stars, and review counts."

item_rag_tool = JSONSearchTool(
    json_path='data/item_subset.json',
    config=rag_config,
    collection_name='v3_hf_item_data'
)
item_rag_tool.name = "search_restaurant_feature_data"
item_rag_tool.description = "Useful to retrieve a specific restaurant's location, categories, attributes, and overall stars."

review_rag_tool = JSONSearchTool(
    json_path='data/review_subset.json',
    config=rag_config,
    collection_name='v3_hf_review_data'
)
review_rag_tool.name = "search_historical_reviews_data"
review_rag_tool.description = "Useful to retrieve the actual text content of past reviews for users or restaurants."

# ── Helper Tools — EXACT MATCH (fixes wrong user returned by RAG) ─────────────
@tool
def search_user_data(user_id: str = None, **kwargs) -> str:
    """Search for a specific user profile by user_id."""
    uid = user_id or kwargs.get('search_query')
    if not uid:
        return "Error: No user_id provided."
    try:
        with open('data/user_subset.json', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get('user_id') == uid:
                        item.pop('friends', None)  
                        return json.dumps(item)
        return f"Error: User {uid} not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def search_review_data(query_id: str = None, **kwargs) -> str:
    """Search for historical reviews by user_id or item_id."""
    qid = query_id or kwargs.get('search_query')
    if not qid:
        return "Error: No query_id provided."
    try:
        results = []
        with open('data/review_subset.json', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get('user_id') == qid or item.get('item_id') == qid:
                        results.append(item)
        if results:
            return json.dumps(results[:3]) 
        return f"Error: No reviews found for {qid}."
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def search_item_data(item_id: str = None, **kwargs) -> str:
    """Search for restaurant/business features by item_id."""
    iid = item_id or kwargs.get('search_query')
    if not iid:
        return "Error: No item_id provided."
    try:
        with open('data/item_subset.json', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get('item_id') == iid:
                        return json.dumps(item)
        return f"Error: Item {iid} not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"

# ── Load YAML configs ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(__file__)

with open(os.path.join(SCRIPT_DIR, "config", "agents.yaml"), "r") as f:
    agents_cfg = yaml.safe_load(f)

with open(os.path.join(SCRIPT_DIR, "config", "tasks.yaml"), "r") as f:
    tasks_cfg = yaml.safe_load(f)

# ── Agents ────────────────────────────────────────────────────────────────────
user_profiler = Agent(
    config=agents_cfg["user_profiler"],
    tools=[search_user_data, search_review_data],
    llm=local_llm,
    allow_delegation=False,
    verbose=True
)

item_analyst = Agent(
    config=agents_cfg["item_analyst"],
    tools=[search_item_data],
    llm=local_llm,
    allow_delegation=False,
    verbose=True
)

prediction_modeler = Agent(
    config=agents_cfg["prediction_modeler"],
    llm=local_llm,
    allow_delegation=False,
    verbose=True
)

# ── Tasks ─────────────────────────────────────────────────────────────────────
analyze_user_task = Task(
    config=tasks_cfg["analyze_user_task"],
    agent=user_profiler
)

analyze_item_task = Task(
    config=tasks_cfg["analyze_item_task"],
    agent=item_analyst
)

predict_review_task = Task(
    config=tasks_cfg["predict_review_task"],
    agent=prediction_modeler,
    context=[analyze_user_task, analyze_item_task],
    output_file="report.json"
)

# ── Crew ──────────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[user_profiler, item_analyst, prediction_modeler],
    tasks=[analyze_user_task, analyze_item_task, predict_review_task],
    process=Process.sequential,
    knowledge_sources=[yelp_knowledge],
    embedder={
        "provider": "sentence-transformer",
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    },
    verbose=True
)