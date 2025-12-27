# from PIL import Image
# import requests
# from modelscope import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
# import torch

# model = AutoModel.from_pretrained(".models/vision_model/siglip-base-patch16-224")
# processor = AutoProcessor.from_pretrained(".models/vision_model/siglip-base-patch16-224")

# # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
# # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# default_image = Image.new('RGB', (224, 224), color='white')
# pixel_values = processor(text=None, images=default_image, return_tensors="pt")["pixel_values"]
# # print(pixel_values.shape)
# # 获取图像特征（[batch_size, hidden_size]）
# image_embeds = model.vision_model(pixel_values).last_hidden_state
# print(image_embeds.size())  # 应该是 torch.Size[1, 196, 768]

from modelscope import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoProcessor
from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self, llm_model_path = "Qwen/Qwen3-0.6B", vision_model_path = ".models/vision_model/siglip-base-patch16-224", 
                freeze_vision_model = True,
                image_pad_num = 49,
                **kwargs):
        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)

class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)

        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size*4, self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)

        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state

        b, s, d = image_embeds.shape
        image_embeds = image_embeds.view(b, -1, d*4)  # (b, 196, d) --> (b, 49, d*4) 压缩图片tokens
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))
        
        text_embeds = text_embeds.to(image_features.dtype)
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])

        num_images, num_image_patches, embed_dim = image_features.shape
        text_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)
        inputs_embeds = text_embeds

        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
             