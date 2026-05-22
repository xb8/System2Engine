import os
import gradio as gr
from litellm import completion

# =====================================================================
# CONFIGURATION CONSTANTS
# =====================================================================
DEFAULT_API_KEY = ""
DEFAULT_API_BASE = ""
DEFAULT_MODEL = ""

# Custom CSS to give the interface a clean, modern, high-contrast look
CUSTOM_CSS = """
footer {visibility: hidden}
.gradio-container {background-color: #0f172a;}
.sidebar {background-color: #1e293b; border-right: 1px solid #334155; padding: 20px; border-radius: 8px;}
.output-box {background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 15px;}
"""

def run_structured_reasoning_engine(user_problem, profile_type, api_key, api_base, model_name):
    """
    Executes a 3-phase sequential reasoning loop over LiteLLM.
    Forces the model to isolate rules, extract facts, and calculate cleanly 
    on the text surface to prevent logic-jumping hallucinations.
    """
    if not api_key:
        yield "❌ Error: Please provide a valid API Key in the Configuration panel."
        return

    custom_kwargs = {
        "api_key": api_key,
        "api_base": api_base
    }
    
    # Adjust systemic focus based on user-selected profile
    profile_prompts = {
        "Logical Deduction": "Focus heavily on semantic contradictions, constraints, and conditional rules.",
        "Mathematical / Quant": "Focus heavily on numeric variables, algebraic relations, and arithmetic boundaries.",
        "Code / Systems Debugging": "Focus heavily on state changes, edge cases, input/output structures, and dependencies."
    }
    selected_focus = profile_prompts.get(profile_type, "")

    try:
        # -------------------------------------------------------------
        # PHASE 1: BOUNDARY & RULE DEFINITION
        # -------------------------------------------------------------
        yield "⏳ Phase 1: Framing problem boundaries and operational rules...\n"
        
        p1_prompt = (
            f"Problem: {user_problem}\n\n"
            f"SYSTEM CONSTRAINT: You are a Sequential Reasoning Engine. {selected_focus} "
            "Your first task is strictly to map out the explicit structural rules of this problem. "
            "Do not attempt to solve it yet. You MUST begin your response exactly with the tag: "
            "'[SYSTEM_RULES] Context Domain:'"
        )
        
        messages = [{"role": "user", "content": p1_prompt}]
        response_p1 = completion(model=model_name, messages=messages, **custom_kwargs)
        p1_output = response_p1.choices[0].message.content
        
        current_display = f"### 🧩 [Phase 1: Scope & Rules]\n{p1_output}\n\n"
        yield current_display

        # -------------------------------------------------------------
        # PHASE 2: FACT EXTRACTION & VARIABLE ISOLATION
        # -------------------------------------------------------------
        current_display += "⏳ Phase 2: Isolating raw premises from surface text...\n"
        yield current_display
        
        # Guide the context window by pre-setting the assistant's trajectory
        context_history = [
            {"role": "user", "content": f"Analyze this problem: {user_problem}"},
            {"role": "assistant", "content": f"{p1_output}\n\n[EXTRACTED_FACTS]\nHere are the isolated, verified variables directly from the text:\n"}
        ]
        
        response_p2 = completion(model=model_name, messages=context_history, **custom_kwargs)
        p2_output = response_p2.choices[0].message.content
        
        # Strip the transition text out and update UI
        current_display = current_display.split("⏳ Phase 2")[0]
        current_display += f"### 🔍 [Phase 2: Fact Extraction]\n* [EXTRACTED_FACTS]\n{p2_output}\n\n"
        yield current_display

        # -------------------------------------------------------------
        # PHASE 3: STEP-BY-STEP CALCULATION & SYNTHESIS
        # -------------------------------------------------------------
        current_display += "⏳ Phase 3: Executing final step-by-step logic processing...\n"
        yield current_display
        
        final_history = [
            {"role": "user", "content": f"Analyze this problem: {user_problem}"},
            {"role": "assistant", "content": f"{p1_output}\n\n[EXTRACTED_FACTS]\n{p2_output}\n\n[EXECUTION_LOG]\nProcessing the sequential calculations based entirely on verified facts:\n"}
        ]
        
        response_p3 = completion(model=model_name, messages=final_history, **custom_kwargs)
        p3_output = response_p3.choices[0].message.content
        
        current_display = current_display.split("⏳ Phase 3")[0]
        current_display += f"### 🎯 [Phase 3: Execution & Final Solution]\n* [EXECUTION_LOG]\n{p3_output}"
        yield current_display

    except Exception as e:
        yield (f"❌ Pipeline Execution Error: {str(e)}\n\n"
               f"**Troubleshooting Checklist:**\n"
               f"* Verify your API configuration in the left panel.\n"
               f"* Confirm that the endpoint proxy is reachable and the model identifier is correct.")


# =====================================================================
# UI RENDERING LAYOUT (Gradio Engine Layout)
# =====================================================================
with gr.Blocks(theme=gr.themes.Ocean(), css=CUSTOM_CSS) as demo:
    
    gr.Markdown("# 🌐 Sequential Logic Reasoning Engine")
    gr.Markdown("A multi-phase context pipeline designed to eliminate LLM logic jumps by forcing structured, text-surface reasoning.")
    
    with gr.Row():
        # Left Panel: User Input & Advanced Settings
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### ⚙️ Engine Parameters")
            
            profile_dropdown = gr.Dropdown(
                choices=["Logical Deduction", "Mathematical / Quant", "Code / Systems Debugging"],
                value="Logical Deduction",
                label="Reasoning Profile Optimization"
            )
            
            with gr.Accordion("🔌 API Gateway Settings", open=False):
                api_key_input = gr.Textbox(
                    label="LiteLLM Virtual Key", 
                    value=DEFAULT_API_KEY, 
                    type="password"
                )
                api_base_input = gr.Textbox(
                    label="Proxy Base URL", 
                    value=DEFAULT_API_BASE
                )
                model_name_input = gr.Textbox(
                    label="Model Identifier", 
                    value=DEFAULT_MODEL
                )
            
            gr.Markdown("---")
            problem_input = gr.Textbox(
                label="Input Problem / Constraint Set",
                placeholder="Paste the raw logical text or complex math constraints here...",
                lines=5
            )
            submit_btn = gr.Button("Execute Multi-Phase Inference", variant="primary")
        
        # Right Panel: Output Workspace
        with gr.Column(scale=2, elem_classes="output-box"):
            gr.Markdown("### 🖥️ Live Streaming Inference Log")
            output_markdown = gr.Markdown(value="*Awaiting logical problem input...*")

    # Wire up UI event listener
    submit_btn.click(
        fn=run_structured_reasoning_engine, 
        inputs=[problem_input, profile_dropdown, api_key_input, api_base_input, model_name_input], 
        outputs=[output_markdown]
    )

if __name__ == "__main__":
    demo.queue().launch()
