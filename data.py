import numpy as np

_TOOL_TEMPLATES = [
    {
        "patterns": [
            "I need to {verb} information about {obj}",
            "Can you {verb} for details on {obj}?",
            "Please {verb} the latest news about {obj}",
            "I want to {verb} records related to {obj}",
            "Help me {verb} data on {obj}",
        ],
        "verbs": ["search", "find", "look up", "query", "lookup"],
        "objs": [
            "climate policy",
            "stock market",
            "health trends",
            "space missions",
            "economics",
        ],
    },
    {
        "patterns": [
            "I need to {verb} the value of {obj}",
            "Can you {verb} {obj} for me?",
            "Please {verb} the result of {obj}",
            "I want to {verb} how much {obj} equals",
            "Help me {verb} the total for {obj}",
        ],
        "verbs": ["calculate", "compute", "sum up", "figure out", "determine"],
        "objs": [
            "42 times 7 plus 3",
            "100 divided by 4 minus 5",
            "12 squared plus 8",
            "256 divided by 16 times 3",
            "99 minus 17 times 2",
        ],
    },
    {
        "patterns": [
            "What is the {verb} for {obj}?",
            "I need the {verb} in {obj}",
            "Can you check the {verb} for {obj}?",
            "Please tell me about the {verb} in {obj}",
            "I want to know the {verb} for {obj}",
        ],
        "verbs": [
            "forecast",
            "temperature",
            "weather outlook",
            "rain probability",
            "wind speed",
        ],
        "objs": ["Paris", "Tokyo", "London", "New York", "Sydney"],
    },
    {
        "patterns": [
            "I need to {verb} a {obj} for {day}",
            "Can you {verb} a {obj} on {day}?",
            "Please {verb} the {obj} scheduled for {day}",
            "I want to {verb} a {obj} on {day}",
            "Help me {verb} the {obj} for {day}",
        ],
        "verbs": ["schedule", "book", "set up", "arrange", "plan"],
        "objs": ["meeting", "appointment", "event", "deadline", "reminder"],
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    },
    {
        "patterns": [
            "I need to {verb} an email to {person} about {obj}",
            "Can you {verb} a message to {person} regarding {obj}?",
            "Please {verb} the {obj} to {person}",
            "I want to {verb} {person} about {obj}",
            "Help me {verb} a note to {person} on {obj}",
        ],
        "verbs": ["send", "compose", "draft", "write", "reply"],
        "objs": [
            "the project update",
            "the meeting notes",
            "the invoice",
            "the proposal",
            "the report",
        ],
        "people": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    },
]

TOOL_KEYWORDS = {
    0: ["search", "find", "look up", "query", "lookup"],
    1: ["calculate", "compute", "sum", "average", "multiply", "math"],
    2: ["weather", "forecast", "temperature", "rain", "sunny", "wind"],
    3: ["schedule", "meeting", "appointment", "event", "remind", "calendar"],
    4: ["email", "send", "message", "reply", "compose", "mail"],
}

N_TOOLS = len(_TOOL_TEMPLATES)


def _fill_pattern(pattern: str, tool: dict, rng: np.random.Generator) -> str:
    text = pattern
    for key in ("verb", "obj", "day", "person"):
        placeholder = "{" + key + "}"
        if placeholder not in text:
            continue
        if key == "day":
            pool = tool["days"]
        elif key == "person":
            pool = tool["people"]
        elif key == "obj":
            pool = tool["objs"]
        else:
            pool = tool["verbs"]
        idx = int(rng.integers(0, len(pool)))
        text = text.replace(placeholder, pool[idx])
    return text


def make_dataset(seed=42, n_samples=256):
    """Generate synthetic agent-task descriptions and their correct tool labels.

    Parameters
    ----------
    seed : int
        Random seed controlling which template/fill-in combinations are chosen.
    n_samples : int
        Number of samples to generate. Must be >= 32.

    Returns
    -------
    X : numpy.ndarray of shape (n_samples,) with dtype object (strings)
    y : numpy.ndarray of shape (n_samples,) with dtype int64 (tool index 0-4)

    Raises
    ------
    ValueError
        If n_samples < 32.
    """
    if n_samples < 32:
        raise ValueError(f"n_samples must be >= 32, got {n_samples}")

    rng = np.random.default_rng(seed)
    X = np.empty(n_samples, dtype=object)
    y = np.empty(n_samples, dtype=np.int64)

    for i in range(n_samples):
        tool_idx = i % N_TOOLS
        tool = _TOOL_TEMPLATES[tool_idx]
        pattern = tool["patterns"][int(rng.integers(0, len(tool["patterns"])))]
        X[i] = _fill_pattern(pattern, tool, rng)
        y[i] = tool_idx

    return X, y
