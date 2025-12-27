from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

from modelscope import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig

from train import VLMConfig, VLM

from dataset.sft_dataset import SFTDataset, MyDataCollator


config = VLMConfig()
processor = AutoProcessor.from_pretrained(config.vision_model_path)
tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)

AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

model = AutoModelForCausalLM.from_pretrained('/home/zhangxichen/dl2/vlm_from_scratch/save/pretrain')

# for name, param in model.named_parameters():
#     if 'linear' in name or 'vision_model':
#         param.requires_grad = False
#     if 'llm_model' in name:
#         param.requires_grad = True

for name, param in model.named_parameters():
    if name.startswith("llm_model"):
        param.requires_grad = True
    else:
        param.requires_grad = False

print(f'模型参数量为：{sum(p.numel() for p in model.parameters())}') 
print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}') 

images_path = '/home/zhangxichen/dl2/vlm_from_scratch/dataset/sft_images'
jsonl_path = '/home/zhangxichen/dl2/vlm_from_scratch/dataset/sft_data.jsonl'
output_dir = 'save/sft' 

args = TrainingArguments(
    output_dir=output_dir,
    do_train=True,
    per_device_train_batch_size=2,
    learning_rate=1e-4,
    ddp_find_unused_parameters=False,
    num_train_epochs=5,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=8,
    logging_steps=100,
    report_to='tensorboard',
    dataloader_pin_memory=True,
    dataloader_num_workers=8
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=SFTDataset(jsonl_path, images_path, tokenizer, processor, config),
    data_collator=MyDataCollator(tokenizer)  
)

trainer.train(resume_from_checkpoint=False)
trainer.save_model('save/sft')
trainer.save_state()