import gradio as gr
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from models.model_vlm import VLMConfig, VLM
import torch
from torch.nn import functional as F

config = VLMConfig(vision_model_path='./models/vision_model/siglip-base-patch16-224', image_pad_num=49)

device = "cuda"
processor = AutoProcessor.from_pretrained(config.vision_model_path)
tokenizer = AutoTokenizer.from_pretrained('./checkpoints')
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

pretrain_model = AutoModelForCausalLM.from_pretrained('./save/sft')
pretrain_model.to(device)


pretrain_model.eval()

def generate(mode, image_input, text_input, max_new_tokens = 512, temperature = 0.0, top_k = None):
    q_text = tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":f'{text_input}\n<image>'}], \
            tokenize=False, \
            add_generation_prompt=True).replace('<image>', '<|image_pad|>'*49)
    input_ids = tokenizer(q_text, return_tensors='pt')['input_ids']
    input_ids = input_ids.to(device)
    # image = Image.open(image_input).convert("RGB")
    # pixel_values = processor(text=None, images=image_input).pixel_values
    pixel_values = processor(images=image_input, return_tensors="pt")['pixel_values']
    pixel_values = pixel_values.to(device)
    eos = tokenizer.eos_token_id
    s = input_ids.shape[1]
    while input_ids.shape[1] < s + max_new_tokens - 1:  
        model = pretrain_model
        inference_res = model(input_ids, None, pixel_values)  
        logits = inference_res.logits 
        logits = logits[:, -1, :] 

        for token in set(input_ids.tolist()[0]):  
            logits[:, token] /= 1.0

        if temperature == 0.0: 
            _, idx_next = torch.topk(logits, k=1, dim=-1)
        else:
            logits = logits / temperature  
            if top_k is not None:  
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') 

            probs = F.softmax(logits, dim=-1)  
            idx_next = torch.multinomial(probs, num_samples=1, generator=None)  

        if idx_next == eos:  
            break

        input_ids = torch.cat((input_ids, idx_next), dim=1)  
    return tokenizer.decode(input_ids[:, s:][0])

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="选择图片")
        with gr.Column(scale=1):
            mode = gr.Radio(["pretrain", "sft"], label="选择模型")
            text_input = gr.Textbox(label="输入文本")
            text_output = gr.Textbox(label="输出文本", interactive=False, lines=20)
            generate_button = gr.Button("生成")
            generate_button.click(generate, inputs=[mode, image_input, text_input], outputs=text_output)
            

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7891)