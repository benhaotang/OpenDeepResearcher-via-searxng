import gradio as gr
import requests
import json

def research(system_message, query, max_iterations, base_url):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": query})
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-xxx"
    }
    
    data = {
        "model": "deep_researcher",
        "messages": messages,
        "max_iterations": max_iterations,
        "stream": True
    }
    
    agent_thinking = []
    final_report = ""
    in_thinking = False
    current_think = ""

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            
            if line.startswith(b"data: "):
                try:
                    chunk = json.loads(line[6:])  # Skip "data: " prefix
                    if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                        content = chunk["choices"][0]["delta"]["content"]
                        
                        if "<think>" in content:
                            in_thinking = True
                        elif "</think>" in content:
                            in_thinking = False
                            agent_thinking.append(current_think)
                            current_think = ""
                            yield "\n".join(agent_thinking), final_report
                        elif in_thinking:
                            current_think += content
                            yield "\n".join(agent_thinking + [current_think]), final_report
                        else:
                            final_report += content
                            yield "\n".join(agent_thinking), final_report
                except json.JSONDecodeError:
                    continue
                
    except Exception as e:
        yield f"Error: {str(e)}", "An error occurred during research"

def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# OpenDeepResearcher Interface", latex_delimiters=[{ "left": "$$", "right": "$$", "display": True }])
        with gr.Row():
            # Left column for settings (1/4 width)
            with gr.Column(scale=1):
                system_msg = gr.Textbox(
                    label="System Message (Optional)", 
                    placeholder="Enter system message here...",
                    lines=3
                )
                query = gr.Textbox(
                    label="Research Query",
                    placeholder="What would you like to research?",
                    lines=3
                )
                max_iter = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Max Iterations"
                )
                base_url = gr.Textbox(
                    label="API Base URL",
                    value="http://localhost:8000/v1",
                    placeholder="Enter API base URL here..."
                )
                btn = gr.Button("Start Research", variant="primary")

            # Right column for outputs (3/4 width)
            with gr.Column(scale=3):
                thinking_output = gr.Textbox(
                    label="Agent Thinking",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True
                )
                final_output = gr.Textbox(
                    label="Final Report",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )
        
        btn.click(
            fn=research,
            inputs=[system_msg, query, max_iter, base_url],
            outputs=[thinking_output, final_output],
            api_name="research"
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch()