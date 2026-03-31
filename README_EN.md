# LangChain RAG Agent

A production-ready RAG (Retrieval-Augmented Generation) agent system built with LangGraph and LangChain, featuring document retrieval, relevance scoring, query rewriting, and intelligent response generation.

## 🌟 Features

- **Intelligent Routing**: Dynamic routing based on tool types (retrieval vs non-retrieval)
- **Document Grading**: Relevance scoring for retrieved documents
- **Query Rewriting**: Automatic query improvement for better retrieval
- **Parallel Tool Execution**: Efficient parallel processing of tool calls
- **Memory Management**: Persistent memory with PostgreSQL or in-memory storage
- **LangChain v1 Support**: Built on LangChain v1 with content_blocks and Middleware support

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL (optional, for persistent storage)
- API keys for LLM providers (OpenAI, Azure, etc.)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/rag-agent.git
   cd rag-agent
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Activate it
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the project root:
   ```env
   # LLM Configuration
   DASHSCOPE_API_KEY=your_api_key_here
   LLM_TYPE=qwen  # or openai, oneapi, ollama
   
   # Database (optional)
   DB_URI=postgresql://user:password@localhost:5432/rag_agent
   ```

5. **Initialize Vector Database**
   ```bash
   python vectorSave.py
   ```

### Usage

#### Command Line Interface

```bash
# Run the original version
python ragAgent.py

# Run the v1 version (with LangChain v1 features)
python ragAgent_v1.py
```

#### API Service

```bash
# Run the original API service
python main.py

# Run the v1 API service
python main_v1.py

# The API will be available at http://localhost:8012
```

#### API Endpoints

- **Health Check**: `GET /health`
- **Chat Completions**: `POST /v1/chat/completions`

Example request:
```bash
curl -X POST http://localhost:8012/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "查询我的健康档案"}
    ],
    "stream": false,
    "userId": "user123"
  }'
```

## 📁 Project Structure

```
rag-agent/
├── ragAgent.py              # Original RAG agent (LangGraph)
├── ragAgent_v1.py           # LangChain v1 version ⭐
├── main.py                  # Original FastAPI service
├── main_v1.py              # v1 FastAPI service ⭐
├── vectorSave.py           # Vector database initialization
├── webUI.py                # Web interface
├── utils/
│   ├── config.py           # Configuration management
│   ├── llms.py             # LLM initialization
│   └── tools_config.py     # Tool definitions
├── prompts/
│   ├── prompt_template_agent.txt
│   ├── prompt_template_generate.txt
│   ├── prompt_template_grade.txt
│   └── prompt_template_rewrite.txt
├── docs/                   # Documentation ⭐
│   ├── MIGRATION_GUIDE.md  # Migration guide
│   └── COMPARISON.md       # Version comparison
└── tests/
    └── test_migration_v1.py # Test suite ⭐
```

⭐ = New files for LangChain v1 migration

## 🔄 Version Comparison

### Original Version (ragAgent.py)
- Standard LangGraph implementation
- Direct message handling
- Basic error handling

### v1 Version (ragAgent_v1.py)
- **Content Blocks**: Unified message format across providers
- **Middleware Support**: PII protection, conversation summarization
- **Enhanced Logging**: Detailed debugging information
- **Better Extensibility**: Modular middleware system

See [COMPARISON.md](docs/COMPARISON.md) for detailed comparison.

## 🏗️ Architecture

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    START([开始]) --> AGENT[Agent 节点]
    AGENT -->|tools_condition| TOOLS{调用工具?}
    TOOLS -->|是| CALL_TOOLS[并行执行工具]
    TOOLS -->|否| END([结束])
    CALL_TOOLS --> ROUTE_TOOLS{工具类型?}
    ROUTE_TOOLS -->|检索工具| GRADE[文档评分]
    ROUTE_TOOLS -->|其他工具| GENERATE[生成响应]
    GRADE --> ROUTE_GRADE{相关性?}
    ROUTE_GRADE -->|yes| GENERATE
    ROUTE_GRADE -->|no| REWRITE[重写查询]
    REWRITE --> AGENT
    GENERATE --> END
```

## 🛠️ Tools

- **retrieve**: 健康档案查询工具
- **multiply**: 计算工具（演示用）

## 🔐 Security

The v1 version includes built-in security middleware:

- **PIIMiddleware**: Automatically detects and masks sensitive information
- **SummarizationMiddleware**: Auto-summarizes long conversations to reduce context length

## 📚 Documentation

- [README_V1.md](README_V1.md) - Complete migration report
- [MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) - Migration guide
- [COMPARISON.md](docs/COMPARISON.md) - Detailed version comparison
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

## 🧪 Testing

```bash
# Run test suite
python test_migration_v1.py

# Run verification script
python verify_migration.py
```

## 📦 Dependencies

Key dependencies:
- `langchain` >= 1.0.0
- `langgraph` >= 0.2.0
- `langchain-core` >= 0.3.0
- `fastapi` >= 0.100.0
- `uvicorn` >= 0.23.0
- `langchain-openai` / `langchain-anthropic` (provider-specific)
- `psycopg2-binary` (PostgreSQL support)
- `psycopg-pool` (Connection pooling)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Framework for building LLM applications
- [LangGraph](https://github.com/langchain-ai/langgraph) - State-based agent framework
- All contributors and users of this project

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

⭐ Star this repo if you find it helpful!
