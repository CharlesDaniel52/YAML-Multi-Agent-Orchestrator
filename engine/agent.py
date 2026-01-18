import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class Agent:
    def __init__(self, agent_id, role, goal, tools=None):
        self.id = agent_id
        self.role = role
        self.goal = goal
        self.tools = tools or []
        self.subagents = []

    def run(self, context):
        """
        Execute agent logic.
        Uses OpenAI if API key is present, otherwise falls back to mock output.
        """

        prompt = f"""
Role: {self.role}
Goal: {self.goal}

Context:
{context}
"""

        api_key = os.getenv("OPENAI_API_KEY")

        # --- OpenAI path (optional) ---
        if api_key and OpenAI:
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful autonomous agent."},
                    {"role": "user", "content": prompt},
                ],
            )

            return (
                f"[{self.role} | {self.id}]\n"
                f"{response.choices[0].message.content.strip()}"
            )

        # --- Mock fallback (default / safe) ---
        output = (
            f"[{self.role} | {self.id}]\n"
            f"Goal: {self.goal}\n"
        )

        if context:
            context_summary = "\n".join(
                f"- From {k}: {v}" for k, v in context.items()
            )
            output += f"Context received:\n{context_summary}\n"

        if self.tools:
            output += "Tools used:\n"
            for tool in self.tools:
                output += f"- {tool}\n"

        # --- Run subagents (hierarchical execution) ---
        if self.subagents:
            output += "\nSubagents executed:\n"
            for subagent in self.subagents:
                subagent.run(context)
                output += f"- {subagent.id} completed\n"

        output += "Result: Task completed successfully."
        return output
