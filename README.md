# System2Engine
A multi-phase context pipeline built with Python, Gradio, and LiteLLM. This engine mitigates Large Language Model (LLM) hallucinations by forcing the model through a structured, three-phase cognitive workflow (System-2 thinking) before generating a final calculation or answer.

## Features

* **Multi-Phase Inference Pipeline:** Forcibly breaks complex problems down into Rule Definition, Fact Extraction, and Step-by-Step Execution.
* **Model Agnostic:** Powered by `litellm`, allowing seamless hot-swapping between OpenAI, Anthropic, Google, and local models.
* **Dynamic System Priming:** Pre-configured profiles adjust the model's attention mechanisms for Logical Deduction, Mathematical/Quant, or Code Debugging.
