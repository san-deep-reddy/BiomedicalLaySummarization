# %%
import sys, os
sys.path.insert(0, "/home/ubuntu/adapters/src")
import pandas as pd
import numpy as np
from tqdm import tqdm

import adapters
from adapters import AutoAdapterModel, AdapterTrainer, SeqBnConfig, Seq2SeqAdapterTrainer
import adapters.composition as ac
from adapters.composition import Fuse
import peft, torch
from transformers import (AutoTokenizer, 
                          AutoModel,
                          AutoModelForSeq2SeqLM,
                          BartForConditionalGeneration,
                          Seq2SeqTrainer, 
                          Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq,
                          EarlyStoppingCallback,
                          set_seed,
                          Trainer
                         )
from datasets import Dataset, DatasetDict
device = 'cuda' if torch.cuda.is_available() else 'cpu'
adapters.__file__
import torch
torch.autograd.set_detect_anomaly(True)

# %%
# model = BartForConditionalGeneration.from_pretrained("/opt/dlami/nvme/biobart_finetuned/checkpoint-5162") 
# tokenizer = AutoTokenizer.from_pretrained("/opt/dlami/nvme/biobart_finetuned/checkpoint-5162")

# %%
model = AutoAdapterModel.from_pretrained("/opt/dlami/nvme/exp2_pretrained/results").to(device) #AutoAdapterModel.from_pretrained("/opt/dlami/nvme/knowledge_consolidation/checkpoint-7956")
tokenizer = AutoTokenizer.from_pretrained("mse30/bart-base-finetuned-pubmed")

# %%
# model.delete_head('knowledge_consolidation')
# adapter_setup = Fuse("adapter1", "adapter2", "adapter3")    
# model.add_seq2seq_lm_head('fine_tunning')
# model.train_adapter_fusion([adapter_setup, 'adapter1', 'adapter2', 'adapter3'], unfreeze_adapters=True, train_embeddings=True)

model.delete_head('adapter2')
model.add_seq2seq_lm_head('fine_tunning')
model.train_adapter(['adapter2'], train_embeddings=True)
print(model)

# %%
def make_all_layers_trainable(model):
    # Set all parameters to require gradients (trainable)
    for param in model.parameters():
        param.requires_grad = True

# Example usage assuming `model` is your instantiated model
make_all_layers_trainable(model)

# %%
def print_parameters(model):
    trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    untrainable_params = {name: param for name, param in model.named_parameters() if not param.requires_grad}

    print("Trainable Parameters:")
    total_trainable_params = 0
    for name, param in trainable_params.items():
        print(f"{name}: {param.size()}")
        total_trainable_params += param.numel()
    print(f"Total number of trainable parameters: {total_trainable_params}")

    print("\nUntrainable Parameters:")
    total_untrainable_params = 0
    for name, param in untrainable_params.items():
        print(f"{name}: {param.size()}")
        total_untrainable_params += param.numel()
    print(f"Total number of untrainable parameters: {total_untrainable_params}")

# Assuming your model instance is named `model`
print_parameters(model)

# %%
#data_path = "/opt/dlami/nvme/"

# train_df = pd.read_csv(data_path + 'train.csv', usecols = ['input_text', 'target_text'])
# val_df = pd.read_csv(data_path + 'val.csv', usecols = ['input_text', 'target_text'])
# test_df = pd.read_csv(data_path + 'test.csv', usecols = ['input_text', 'target_text'])


df = pd.read_excel("/opt/dlami/nvme/plos_all.xlsx")

def create_dataframe(df, split):
    selected_df = df[df["Split"] == split][["Abstract", "Summary"]].rename(columns={"Abstract": "input_text", "Summary": "target_text"})
    return selected_df

train_df, test_df, val_df = create_dataframe(df, "train"), create_dataframe(df, "test"), create_dataframe(df, "val")

train_dataset, val_dataset, test_dataset = Dataset.from_dict(train_df[:1000]), Dataset.from_dict(val_df[:100]), Dataset.from_dict(test_df[:100])

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "val": val_dataset
})

# %%
batch_size = 16
max_length = 1024 

def process_data(batch, tokenizer):
    if "T5" in tokenizer.__class__.__name__:
        model_max_length = 512  
        tokenizer.prefix = "summarize: "
    else:
        model_max_length = tokenizer.model_max_length
    inputs = tokenizer(batch["input_text"], padding="max_length", max_length=model_max_length, truncation=True)
    outputs = tokenizer(batch["target_text"], padding="max_length", max_length=model_max_length, truncation=True)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
    return batch

dataset = dataset.map(lambda batch: process_data(batch, tokenizer), batched=True, batch_size=batch_size, remove_columns=['input_text', 'target_text'])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
from transformers import TrainerCallback

def test_model(model, tokenizer, max_length, test_dataset):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Ensure the model is on the correct device

    decoded_preds = []
    for batch in test_dataset:
        input_ids = batch["input_ids"].clone().detach().unsqueeze(0).to(device)
        attention_mask = batch["attention_mask"].clone().detach().unsqueeze(0).to(device)

        with torch.no_grad():  
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, max_length=512, 
                            num_beams=5, no_repeat_ngram_size=2,length_penalty=2.0, early_stopping=True)
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
        
        output = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        decoded_preds.append(output)
    return decoded_preds


class TestAfterEpochCallback(TrainerCallback):
    def __init__(self, test_dataset, tokenizer, max_length):
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def on_epoch_end(self, args, state, control, model, **kwargs):
        print(f"Testing model after epoch {state.epoch}")
        decoded_preds = test_model(model, self.tokenizer, self.max_length, self.test_dataset)
        
        # Save the decoded predictions
        with open(f"/opt/dlami/nvme/exp2_finetuned_plos/exp2_finetuned_plos_{state.epoch}.txt", "w") as f:
            for pred in decoded_preds:
                f.write(pred + "\n")

test_dataset = Dataset.from_dict(test_df[:20])
test_dataset = test_dataset.map(lambda batch: process_data(batch, tokenizer), batched=True, batch_size=batch_size, remove_columns=['input_text', 'target_text'])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_callback = TestAfterEpochCallback(test_dataset, tokenizer, max_length)

# # %%
training_args = Seq2SeqTrainingArguments(
    output_dir="/opt/dlami/nvme/exp2_finetuned_plos",
    per_device_eval_batch_size=14,
    per_device_train_batch_size=14,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,
    remove_unused_columns=True,
    save_strategy="epoch",
    save_total_limit=1,
    num_train_epochs=5,
    gradient_accumulation_steps=5,
    eval_accumulation_steps=5,
    learning_rate=2e-5,
    fp16=True,
    fp16_full_eval=True,
    optim="adamw_bnb_8bit",
    seed=42,
    report_to="none",
    save_total_limit=1,
)

# Set Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Train model
trainer = Seq2SeqAdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['val'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), test_callback],
)

trainer.train()
trained_model = trainer.model


model_preds = test_model(model, tokenizer, max_length, dataset['test'])

with open('/opt/dlami/nvme/exp2_finetuned_plos/exp2_finetuned_plos.txt', 'w') as file:
  for string in model_preds:
    file.write(string + "\n")

# %%
# Print the first 30 predictions
# for i, pred in enumerate(model_preds):
#     print(pred)

# %%
# with open('plos_ireneli1024_bart-large-finetuned_model_preds.txt', 'w') as file:
#   for string in model_preds:
#     file.write(string + "\n")

# %%
# import torch

# # Before clearing cache
# print(torch.cuda.memory_allocated())  # Print current memory allocated

# torch.cuda.empty_cache()  # Clear CUDA cache

# # After clearing cache
# print(torch.cuda.memory_allocated())  # Print current memory allocated again

# # %% [markdown]
# # Base Model finetuning

# # %%
# model = AutoModelForSeq2SeqLM.from_pretrained("mse30/bart-base-finetuned-pubmed", gradient_checkpointing=True, use_cache=False) 
# tokenizer = AutoTokenizer.from_pretrained("mse30/bart-base-finetuned-pubmed")

# # %%
# model.config.num_beams = 2
# model.config.max_length = 512
# model.config.min_length = 100
# model.config.length_penalty = 2.0
# model.config.early_stopping = True
# model.config.no_repeat_ngram_size = 3

# # %%
# from datasets import load_dataset, load_metric
# rouge = load_metric("rouge")

# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = tokenizer.pad_token_id
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     rouge_output = rouge.compute(
#         predictions=pred_str, references=label_str, rouge_types=["rouge2"]
#     )["rouge2"].mid

#     return {
#         "rouge2_precision": round(rouge_output.precision, 4),
#         "rouge2_recall": round(rouge_output.recall, 4),
#         "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
#     }


