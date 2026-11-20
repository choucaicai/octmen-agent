"""Llama 2 action prompts shared by training and evaluation."""

SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)


def build_action_prompt(instruction, action_text=None):
    question = f"What action should the robot take to {instruction.lower()}?"
    question = question.replace("<image>", "").strip()
    prompt = f"[INST] <<SYS>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{question} [/INST] "
    if action_text is not None:
        action_text = action_text.replace("<image>", "").strip() or " "
        prompt += f"{action_text}</s>"
    return prompt.rstrip()
