import gradio as gr
from ultralytics import YOLO
from collections import Counter
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

# 1. Initialize the Eyes (YOLO Vision Node)
print("Loading Vision Node...")
vision_model = YOLO("yolov8n.pt")

# 2. Initialize the Brain (Llama 3 Local AI)
print("Loading AI Orchestrator...")
# Note: Ensure this filename perfectly matches the .gguf file you downloaded!
llm = LlamaCpp(
    model_path="/Users/x/Desktop/siva_env/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    temperature=0.1, # Low temperature keeps it strictly professional and accurate
    max_tokens=300,
    n_ctx=2048, # The memory window for the prompt
    verbose=False
)

# 3. Set up the LangChain Orchestrator
# We use Llama 3's specific prompt formatting here so it knows when to stop talking
template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly efficient supply chain AI. 
The vision system has detected the following items currently in stock:
{inventory_list}

The target quota for all items is 5. 
Calculate exactly what is missing and draft a brief, professional email to our supplier (TechSupply Co.) ordering the deficit. Do not include any extra pleasantries.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Draft the email.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

prompt = PromptTemplate.from_template(template)
llm_chain = prompt | llm

def process_inventory(image):
    if image is None:
        return "⚠️ Please upload an image to begin."
    
    # --- Step 1: Vision Inference ---
    results = vision_model(image)
    detected_items = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = vision_model.names[class_id]
            detected_items.append(class_name)
    
    if not detected_items:
        return "SIVA Vision Node active: No recognizable objects detected in this image."
        
    item_counts = Counter(detected_items)
    
    # Format for the LLM prompt
    inventory_text = ""
    for item, count in item_counts.items():
        inventory_text += f"- {count}x {item}\n"
        
    # --- Step 2: AI Orchestration ---
    # Pass the vision text to LangChain and invoke Llama 3
    generated_email = llm_chain.invoke({"inventory_list": inventory_text})
    
    # --- Step 3: Format the Output ---
    final_output = f"👁️ **SIVA Vision Node Detected:**\n{inventory_text}\n"
    final_output += f"🤖 **SIVA Drafted Purchase Order:**\n{generated_email.strip()}"
    
    return final_output

# 4. Build the UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📦 SIVA: Smart Inventory & Vision Agent")
    gr.Markdown("Upload an image of your stockroom or shelf to generate an automated inventory report and purchase order.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload Stockroom Image")
            analyze_button = gr.Button("Analyze Inventory", variant="primary")
        
        with gr.Column():
            text_output = gr.Textbox(label="System Analysis & Drafted PO", lines=15)
            
    analyze_button.click(fn=process_inventory, inputs=image_input, outputs=text_output)

if __name__ == "__main__":
    demo.launch()
