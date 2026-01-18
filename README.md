# YAML-Driven Multi-Agent Orchestration Engine

This project demonstrates a declarative approach to building and executing multi-agent AI workflows using YAML configuration files.

Instead of writing orchestration logic in code, developers define agents and their collaboration patterns entirely in YAML. The execution engine parses the configuration, instantiates agents, orchestrates their execution, and automatically passes shared context between them.

**Configuration defines collaboration. Execution is automatic.**

---

## Problem Motivation

Building multi-agent systems today requires significant orchestration code to manage:
- Agent lifecycles
- Execution order
- Parallelism
- Context passing
- Tool coordination

This approach is:
- Code-heavy
- Difficult to prototype quickly
- Hard to reason about at a high level

This project explores how multi-agent workflows can be defined declaratively, similar to infrastructure-as-code.

---

## Core Features

- Declarative agent definitions using YAML
- Sequential agent execution
- Parallel agent execution with aggregation
- Hierarchical agents (subagents)
- Automatic shared context passing
- Declarative tool usage per agent (MCP-style support)
- Persistent shared memory across workflow runs
- Deterministic execution flow
- Console-based execution and output

---

## Project Structure
```pgsql
yaml-multi-agent-orchestrator/
│
├── main.py
├── requirements.txt
│
├── engine/
│ ├── agent.py        # Agent logic + tool execution 
│ ├── parser.py       # YAML parsing and validation
│ ├── workflow.py     # Sequential and Parrallel orchestration 
| ├── memory.py       # Persistent memory store
│ └── __init__.py
│
├── configs/
│ ├── sequential.yaml
│ ├── parallel.yaml
│ └── subagents.yaml
│
└── outputs/
  └── sample_run.txt
```

---

## YAML Configuration Overview

### Sequential Workflow Example

```yaml
agents:
  - id: researcher
    role: Research Assistant
    goal: Find key insights about electric vehicles
    tools:
      - python 

  - id: writer
    role: Content Writer
    goal: Write a concise summary using the research

workflow:
  type: sequential
  steps:
    - agent: researcher
    - agent: writer
```

### Parallel Workflow Example

```yaml

agents:
  - id: backend
    role: Backend Engineer
    goal: Propose an API design

  - id: frontend
    role: Frontend Engineer
    goal: Propose a UI layout

  - id: reviewer
    role: Tech Lead
    goal: Review and consolidate proposals

workflow:
  type: parallel
  branches:
    - backend
    - frontend
  then:
    agent: reviewer
```

### Subagents (Hierarchical Agents)

Agents can declaratively define **subagents** that execute within the scope of a parent agent.

Subagents are:
- Defined inside an agent’s YAML configuration
- Instantiated during agent construction
- Executed as part of the parent agent’s logic
- Isolated from the global workflow graph

This enables **hierarchical orchestration** without modifying the top-level workflow.

**Example**

```yaml
agents:
  - id: reviewer
    role: Tech Lead
    goal: Review and consolidate proposals
    subagents:
      - id: backend_reviewer
        role: Backend Reviewer
        goal: Review backend proposal
      - id: frontend_reviewer
        role: Frontend Reviewer
        goal: Review frontend proposal

workflow:
  type: sequential
  steps:
    - agent: reviewer
```
---

## How It Works

1. YAML configuration is parsed and validated
2. Agents are instantiated from declarative definitions
3. Persistent shared memory is loaded at workflow start
4. Persistent memory is stored locally on the disk
5. If no memory file exists, it is automatically created on first run
6. Workflow engine executes agents sequentially or in parallel
7. Agents optionally invoke tools declared in configuration 
8. Agent outputs are stored in shared context
9. Downstream agents automatically receive relevant context
10. Updated memory state is persisted after execution 
11. Final results are printed to the console


---

## Running the Project

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Sequential Workflow

```bash
python3 main.py configs/sequential.yaml
```

### Run Parallel Workflow

```bash
python3 main.py configs/parallel.yaml
```

### Optional: Enable OpenAI-powered agents

By default, the engine runs in mock mode and does not require any API keys.

To enable real LLM-backed agent execution, set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Sample output is available in `outputs/sample_run.txt`.

---

## Design Principles

- Declarative over imperative
- Hierarchical agent composition
- Configuration-driven collaboration
- Explicit orchestration boundaries 
- Deterministic execution 
- Simplicity over production complexity

---

## Non-Goals

This project intentionally does not include:

- Fully autonomous or self-modifying agent graphs
- Advanced scheduling or retries
- UI dashboards
- Complex dependency graphs
- Autonomous long-term learning

The focus is clarity, correctness, and rapid experimentation.

---

## Conclusion

This project demonstrates how multi-agent systems can be made easier to define, reason about, and experiment with by moving orchestration logic from code into configuration.

It provides a strong foundation for future extensions such as hierarchal agents, advanced memory strategies, execution graphs, and richer tool integration.

---
