import sys

from engine.agent import Agent
from engine.parser import load_config
from engine.workflow import WorkflowEngine


def build_agents(agent_configs):
    """
    Create Agent objects from YAML agent definitions.
    Returns a dict of agent_id -> Agent instance.
    """
    agents = {}

    for agent_cfg in agent_configs:
        agent = Agent(
            agent_id=agent_cfg["id"],
            role=agent_cfg["role"],
            goal=agent_cfg["goal"],
            tools=agent_cfg.get("tools"),
        )
        agents[agent.id] = agent

        # Create subagents if present
        for sub_cfg in agent_cfg.get("subagents", []):
            subagent = Agent(
                agent_id=sub_cfg["id"],
                role=sub_cfg["role"],
                goal=sub_cfg["goal"],
            )
            agent.subagents.append(subagent)

    return agents


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <path_to_yaml_config>")
        sys.exit(1)

    config_path = sys.argv[1]

    print(f"\n📄 Loading config: {config_path}")
    config = load_config(config_path)

    agents = build_agents(config["agents"])
    workflow_config = config["workflow"]

    engine = WorkflowEngine(agents, workflow_config)
    final_context = engine.run()

    print("\n📦 Final Shared Context:\n")
    for agent_id, output in final_context.items():
        print(f"--- {agent_id} ---")
        print(output)
        print()


if __name__ == "__main__":
    main()
