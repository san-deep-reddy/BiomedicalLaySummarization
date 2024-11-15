import sys, torch, json, random, copy, pickle
sys.path.insert(0, "/home/ubuntu/adapters/src")
import adapters
print(adapters.__file__)
import numpy as np
import pandas as pd
import torch.nn as nn
import adapters.composition as ac
from adapters.composition import Fuse
from adapters.heads import PredictionHead
from adapters import AutoAdapterModel, SeqBnConfig, AdapterTrainer
from transformers import AutoTokenizer, TrainingArguments, EvalPrediction, default_data_collator, Trainer
from datasets import Dataset, load_dataset

margin = 0.5
num_negative_samples = 4
num_positive_samples_per_batch = 1200
batch_size = (1 + num_negative_samples) * num_positive_samples_per_batch


model_path = "/opt/dlami/nvme/exp1_pretrained"
model = AutoAdapterModel.from_pretrained(model_path)
model.delete_head('default')
model.add_multiple_choice_head("adapter2", layers=1, num_choices=1, return_logits=False, num_negative_samples=num_negative_samples, margin=margin)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# adapter_fuse = Fuse("adapter1", "adapter2", "adapter3")
# seq_config = SeqBnConfig(reduction_factor=16, use_gating=False)
# model.add_adapter("adapter1", config=seq_config)
# model.add_adapter("adapter2", config=seq_config)
# model.add_adapter("adapter3", config=seq_config)
# model.add_adapter_fusion(adapter_setup)
model.train_adapter(['adapter2'])
# model.train_adapter(['adapter2'])

# json_file_path = "/opt/dlami/nvme/negative_samples_count64.json"
# with open(json_file_path, 'r') as file:
#     negative_samples = json.load(file)


# df_list = []

# # Define the relationships
# relationships = ["interacts with", "is related to", "interacts", "has function", "is a", "treats"]

# # Generate samples from negative_samples
# for relation in relationships:
#     for entity1, values in negative_samples.get(relation, {}).items():
#         for entity2, negative_entities in values.items():
#             samples = [f"{entity1} {relation} {entity2}"]
#             samples.extend([f"{entity1} {relation} {neg_entity}" for neg_entity in negative_entities[:num_negative_samples]])
#             df_list.append(samples)

# # Shuffle the samples before creating the DataFrame
# random.shuffle(df_list)

# # Flatten the list of lists
# flattened_list = [item for sublist in df_list for item in sublist]

# # Create a DataFrame from the flattened list
# df = pd.DataFrame(flattened_list, columns=['text'])

# # Generate labels pattern
# pattern = [1] + [0]*num_negative_samples
# labels = (pattern * (len(df) // len(pattern) + 1))[:len(df)]

# # Add labels to the DataFrame
# df['labels'] = labels

# def generate_batch_indices(start_index, num_positive_samples, num_negative_samples, batch_size, max_index):
#     indices = []
#     for i in range(num_positive_samples):
#         idx = start_index + i * (num_negative_samples + 1)
#         if idx < max_index:
#             indices.append(idx)
#     for j in range(1, num_negative_samples + 1):
#         for i in range(num_positive_samples):
#             idx = start_index + i * (num_negative_samples + 1) + j
#             if idx < max_index:
#                 indices.append(idx)
#     return indices

# # Generate indices for all batches
# all_indices = []
# max_index = len(df)
# for start_index in range(0, max_index, batch_size):
#     batch_indices = generate_batch_indices(start_index, num_positive_samples_per_batch, num_negative_samples, batch_size, max_index)
    
#     # Check if the batch has the required number of positive samples
#     if len(batch_indices) == batch_size:
#         all_indices.extend(batch_indices)

# # Reorder the DataFrame based on the generated indices
# df = df.iloc[all_indices].reset_index(drop=True)
# dataset = Dataset.from_pandas(df)


# def get_train_test_sizes(df_length, batch_size, num_negative_samples):
#     train_split = 0.8  # Desired train split
#     test_split = 0.2   # Desired test split

#     # Calculate train and test sizes
#     train_size = int(np.floor(train_split * df_length / (batch_size * (1 + num_negative_samples))) * batch_size * (1 + num_negative_samples))
#     test_size = int(np.floor(test_split * df_length / (batch_size * (1 + num_negative_samples))) * batch_size * (1 + num_negative_samples))

#     return train_size, test_size

# # Usage
# train_size, test_size = get_train_test_sizes(len(df), batch_size, num_negative_samples)
# dataset = Dataset.from_pandas(df).train_test_split(train_size=train_size, test_size=test_size, shuffle=False)

# def preprocess_dataset(dataset, tokenizer, max_token_length):
#     """
#     Preprocesses a dataset by encoding the text data using the model tokenizer
#     and padding the input sequences to a maximum token length.
    
#     Args:
#     - dataset (datasets.Dataset): The dataset to preprocess.
#     - tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for encoding.
#     - max_token_length (int): The maximum token length for padding.
    
#     Returns:
#     - datasets.Dataset: The preprocessed dataset.
#     """
#     def encode_and_pad_batch(batch):
#         """Encodes a batch of input data using the model tokenizer and pads to max_token_length."""
#         encoded_batch = tokenizer(batch["text"], padding=True, return_tensors="pt")
#         input_ids = encoded_batch["input_ids"]
#         attention_mask = encoded_batch["attention_mask"]
#         padded_input_ids = torch.nn.functional.pad(input_ids, (0, max_token_length - input_ids.shape[1]), value=0)
#         padded_attention_mask = torch.nn.functional.pad(attention_mask, (0, max_token_length - attention_mask.shape[1]), value=0)
#         return {"input_ids": padded_input_ids, "attention_mask": padded_attention_mask, "labels": batch["labels"]}
    
#     dataset = dataset.map(encode_and_pad_batch, batched=True)
#     return dataset

# max_token_length = 61
# dataset = preprocess_dataset(dataset, tokenizer, max_token_length)
# del df, df_list, flattened_list, negative_samples

path = f"/opt/dlami/nvme/train_only_{num_negative_samples}_{num_positive_samples_per_batch}_padded_dataset.pkl"
# with open(path, 'wb') as f:
#     pickle.dump(dataset, f)

with open(path, 'rb') as f:
    dataset = pickle.load(f)

training_args = TrainingArguments(
    output_dir="/opt/dlami/nvme/no_fusion_layer/exp2",
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=5,
    eval_accumulation_steps=5,
    learning_rate=2e-5,
    gradient_checkpointing=True,
    # evaluation_strategy="steps",
    save_strategy="steps",
    report_to="tensorboard",
    # logging_strategy="steps",
    # logging_steps=5,
    # eval_steps=5,
    save_steps=240,
    do_train=True,
    do_eval=False,
    bf16=True,
    bf16_full_eval=True,
    optim="adamw_bnb_8bit",
    dataloader_num_workers=8,
    overwrite_output_dir=True)

from transformers import EvalPrediction
def compute_accuracy(p: EvalPrediction):
    if len(p.predictions[0].shape) == 1:
        preds = p.predictions[0].flatten()
    else:
        preds = np.argmax(p.predictions[0], axis=1).flatten()
    return {"acc": (preds == p.label_ids).mean()}

# small_train_dataset = Dataset.from_dict({
#     "input_ids": dataset["train"]["input_ids"][:10 * batch_size],
#     "attention_mask": dataset["train"]["attention_mask"][:10 * batch_size],
#     "labels": dataset["train"]["labels"][:10 * batch_size]
# })

# small_test_dataset = Dataset.from_dict({
#     "input_ids": dataset["test"]["input_ids"][:2 * batch_size],
#     "attention_mask": dataset["test"]["attention_mask"][:2 * batch_size],
#     "labels": dataset["test"]["labels"][:2 * batch_size]
# })

# Initialize your Trainer
trainer = Trainer(
    model=model, 
    args=training_args,
    train_dataset=dataset,
    # eval_dataset=small_test_dataset,
#    compute_metrics=compute_accuracy  # Add this line to include the compute_metrics function
)
     
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable parameter: {name}")

trainer.train()