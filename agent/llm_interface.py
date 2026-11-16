from agent.model_manager import ask_with_fallback


def ask_llm(prompt, model=None, num_predict=None):

    return ask_with_fallback(prompt, model_override=model, num_predict=num_predict)
