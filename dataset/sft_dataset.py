import io
import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

# def find_assistant_tokens(tokenizer, target):
#     result = []
#     start_index =0
#     end_index = 0
#     while start_index <= len(target)-1:
#         if target[start_index]!=tokenizer('assistant')['input_ids'][0]:
#             start_index+=1
#             end_index+=1
#         else:
#             end_index+=1
#             if target[end_index]==tokenizer('<|im_end|>')['input_ids'][0]:
#                 result.append((start_index+1,end_index+1))
#                 start_index=end_index+1
#     return result

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, images_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = jsonl_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.samples = self.load_data(jsonl_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['image']
        conversations = sample['conversations']

        q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":conversations[0]['content']}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
        a_text = conversations[1]['content'] + self.tokenizer.eos_token
        q_input_ids = self.tokenizer(q_text)['input_ids']
        a_input_ids = self.tokenizer(a_text)['input_ids']
        input_ids = q_input_ids + a_input_ids

        labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
        input_ids = input_ids[:-1]
        labels = labels[1:]


        # messages = [{"role":"system", "content":'You are a helpful assistant.'}]
        # for conversation in conversations:
        #     if conversation['from'] == 'human':
        #         messages.append({"role":"user", "content":conversation['value']})
        #     else:
        #         messages.append({"role":"assistant", "content":conversation['value']})
        # text = self.tokenizer.apply_chat_template(messages, \
        #     tokenize=False, \
        #     ).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
        # # print(text)
        # input_ids = self.tokenizer(text)['input_ids']

        # indexs = find_assistant_tokens(self.tokenizer, input_ids)
        # labels = len(input_ids) * [self.tokenizer.pad_token_id]
        # for index in indexs:
        #     labels[index[0]:index[1]] = input_ids[index[0]:index[1]]
        # input_ids = input_ids[:-1]
        # labels = labels[1:]
    
        image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values']

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }   

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}
