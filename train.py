import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataset.vlm_dataset import MyDataset, MyDataCollator
from models.model_vlm import VLMConfig, VLM
from modelscope import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding


config = VLMConfig(vision_model_path='.models/vision_model/siglip-base-patch16-224', image_pad_num=49)
model = VLM(config).cuda()
print(model)
print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

images_path = '.dataset/pretrain_images'
jsonl_path = '.dataset/pretrain_data.jsonl'
tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
processor = AutoProcessor.from_pretrained(config.vision_model_path)

output_dir = 'checkpoints' 

args = TrainingArguments(
    output_dir=output_dir,
    do_train=True,
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    ddp_find_unused_parameters=False,
    num_train_epochs=5,
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    gradient_accumulation_steps=8,
    logging_steps=100,
    report_to='tensorboard',
    dataloader_pin_memory=True,
    dataloader_num_workers=8
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=MyDataset(jsonl_path, images_path, tokenizer, processor, config),
    data_collator=MyDataCollator(tokenizer)  
)

trainer.train(resume_from_checkpoint=False)
trainer.save_model(output_dir)
trainer.save_state()
