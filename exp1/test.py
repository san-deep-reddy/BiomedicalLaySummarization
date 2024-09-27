import sys
# import torch
# import copy
# import pickle, random
# import spacy, scispacy
# nlp = spacy.load("en_core_sci_sm")
# from itertools import chain
# from transformers import AutoTokenizer, TrainingArguments, Trainer, EvalPrediction, default_data_collator, EvalPrediction
# from datasets import Dataset, load_dataset
# from transformers import DataCollatorForLanguageModeling, AutoModelForMaskedLM

# Insert path to adapters module
sys.path.insert(0, "/home/ubuntu/adapters/src")
import adapters
print(adapters.__file__)

# import numpy as np
# import pandas as pd
# import torch.nn as nn
# import adapters.composition as ac
# from adapters.composition import Fuse
# from adapters.heads import PredictionHead
# from adapters import AutoAdapterModel, SeqBnConfig, AdapterTrainer

# # Configuration
# batch_size = 12
# max_seq_length = 1024
# train_file = "/opt/dlami/nvme/train.txt"
# eval_file = "/opt/dlami/nvme/eval.txt"

# # Load model and tokenizer
# model2 = AutoModelForMaskedLM.from_pretrained("mse30/bart-base-finetuned-pubmed")
# tokenizer = AutoTokenizer.from_pretrained("mse30/bart-base-finetuned-pubmed")
# model2_copy = copy.deepcopy(model2)
# adapters.init(model2_copy)

# model2_copy.add_adapter("adapter1")
# model2_copy.add_adapter("adapter2")
# model2_copy.add_adapter("adapter3")
# adapter_setup = Fuse("adapter1", "adapter2", "adapter3")
# model2_copy.add_adapter_fusion(adapter_setup)
# # model2_copy.delete_head("default")
# # model2_copy.add_causal_lm_head("adapter1")
# model2_copy.train_adapter_fusion([adapter_setup, "adapter1"], unfreeze_adapters=True)
# print(model2_copy)
# for name, module in model2_copy.heads.items():
#     if name == "lm_head":
#         for param in module.parameters():
#             param.requires_grad = True
# print(model2_copy.active_adapters)
# for name, param in model2_copy.named_parameters():
#     if param.requires_grad:
#         print(f"Trainable parameter: {name}")



# # Read train and eval files
# def read_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#     return lines

# train_sentences = read_file(train_file)[:5000]
# eval_sentences = read_file(eval_file)[5000:6000]

# def mask_hard_words_with_limit(examples):
#     new_examples = {"input_ids": [], "attention_mask": [], "special_tokens_mask": []}
#     for input_ids, attention_mask, special_tokens_mask in zip(
#         examples["input_ids"], examples["attention_mask"], examples["special_tokens_mask"]
#     ):
#         # Create a copy of the input IDs
#         new_input_ids = input_ids.copy()

#         text = tokenizer.decode(input_ids, skip_special_tokens=True)
#         doc = nlp(text)
#         hard_words = [token.text for token in doc if hasattr(token._, "is_concept") and token._.is_concept]

#         # Determine the number of tokens to mask (15% of the total tokens)
#         num_tokens_to_mask = int(0.15 * len(input_ids))
#         hard_word_indices = [i for i, token_id in enumerate(input_ids) if tokenizer.decode([token_id]) in hard_words]

#         # Randomly select up to 15% of tokens to mask
#         indices_to_mask = random.sample(hard_word_indices, min(num_tokens_to_mask, len(hard_word_indices)))

#         # Create a new tokenized sequence with hard words masked
#         for i, token_id in enumerate(input_ids):
#             if i in indices_to_mask:
#                 new_input_ids[i] = tokenizer.mask_token_id

#         new_examples["input_ids"].append(new_input_ids)
#         new_examples["attention_mask"].append(attention_mask)
#         new_examples["special_tokens_mask"].append(special_tokens_mask)

#     return new_examples

# # Tokenization function
# def tokenize_and_mask_function(examples):
#     examples["text"] = [line.replace('\t', ' ') for line in examples["text"]]
#     tokenized = tokenizer(examples["text"], return_special_tokens_mask=True)
#     masked = mask_hard_words_with_limit(tokenized)
#     return masked

# # Create datasets
# train_dataset = Dataset.from_dict({"text": train_sentences})
# eval_dataset = Dataset.from_dict({"text": eval_sentences})

# # Tokenize datasets
# tokenized_train = train_dataset.map(
#     tokenize_and_mask_function,
#     batched=True,
#     num_proc=8,
#     remove_columns=['text'],
#     load_from_cache_file=True,
#     desc="Tokenizing train dataset",
# )

# tokenized_eval = eval_dataset.map(
#     tokenize_and_mask_function,
#     batched=True,
#     num_proc=8,
#     remove_columns=['text'],
#     load_from_cache_file=True,
#     desc="Tokenizing eval dataset",
# )

# # Group texts into chunks
# def group_texts(examples):
#     pad_token_id = tokenizer.pad_token_id
#     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#     total_length = len(concatenated_examples['input_ids'])
#     special_token_mask = concatenated_examples['special_tokens_mask']
#     sentence_boundaries = [i for i, token in enumerate(special_token_mask) if token == 1]
    
#     result = {k: [] for k in examples.keys()}
#     current_chunk = {k: [] for k in examples.keys()}
#     current_length = 0
    
#     for i in range(len(sentence_boundaries) - 1):
#         start_idx = sentence_boundaries[i]
#         end_idx = sentence_boundaries[i + 1]
#         sentence_length = end_idx - start_idx
        
#         if current_length + sentence_length > max_seq_length:
#             for k in examples.keys():
#                 if k == 'input_ids':
#                     current_chunk[k].extend([pad_token_id] * (max_seq_length - current_length))
#                 else:
#                     current_chunk[k].extend([0] * (max_seq_length - current_length))
#                 result[k].append(current_chunk[k][:max_seq_length])
            
#             current_chunk = {k: [] for k in examples.keys()}
#             current_length = 0
        
#         for k in examples.keys():
#             current_chunk[k].extend(concatenated_examples[k][start_idx:end_idx])
#         current_length += sentence_length
    
#     if current_chunk['input_ids']:
#         for k in examples.keys():
#             if k == 'input_ids':
#                 current_chunk[k].extend([pad_token_id] * (max_seq_length - current_length))
#             else:
#                 current_chunk[k].extend([0] * (max_seq_length - current_length))
#             result[k].append(current_chunk[k][:max_seq_length])
    
#     return result

# # Apply grouping function to datasets
# tokenized_train = tokenized_train.map(
#     group_texts,
#     batched=True,
#     num_proc=8,
#     load_from_cache_file=True,
#     desc=f"Grouping train texts in chunks of {max_seq_length}",
# )

# tokenized_eval = tokenized_eval.map(
#     group_texts,
#     batched=True,
#     num_proc=8,
#     load_from_cache_file=True,
#     desc=f"Grouping eval texts in chunks of {max_seq_length}",
# )

# # Define data collator
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm_probability=0.15
# )

# # # Serialize grouped datasets
# path_train = "/opt/dlami/nvme/grouped_train.pkl"
# # with open(path_train, 'wb') as f:
# #     pickle.dump(tokenized_train, f)

# path_eval = "/opt/dlami/nvme/grouped_eval.pkl"
# # with open(path_eval, 'wb') as f:
# #     pickle.dump(tokenized_eval, f)

# # Load grouped datasets from pickle
# with open(path_train, 'rb') as f:
#     tokenized_train = pickle.load(f)

# with open(path_eval, 'rb') as f:
#     tokenized_eval = pickle.load(f)

# # Define metrics and evaluation functions
# def preprocess_logits_for_metrics(logits, labels):
#     if isinstance(logits, tuple):
#         logits = logits[0]
#     return logits.argmax(dim=-1)

# def compute_accuracy(p: EvalPrediction):
#     if len(p.predictions[0].shape) == 1:
#         preds = p.predictions[0].flatten()
#     else:
#         preds = np.argmax(p.predictions[0], axis=1).flatten()
#     return {"acc": (preds == p.label_ids).mean()}

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="/opt/dlami/nvme/exp4_pretrained_adapter1",
#     num_train_epochs=1,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     gradient_accumulation_steps=8,
#     eval_accumulation_steps=8,
#     learning_rate=2e-5,
#     gradient_checkpointing=True,
#     evaluation_strategy="steps",
#     save_strategy="steps",
#     report_to="tensorboard",
#     logging_strategy="steps",
#     # logging_steps=5000,
#     # eval_steps=5000,
#     # save_steps=5000,
#     load_best_model_at_end=True,
#     do_train=True,
#     do_eval=True,
#     bf16=True,
#     bf16_full_eval=True,
#     optim="adamw_bnb_8bit",
# )

# # Initialize Trainer
# trainer = AdapterTrainer(
#     model=model2_copy,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_eval,
#     data_collator=data_collator,
#     # compute_metrics=compute_accuracy,
#     # preprocess_logits_for_metrics=preprocess_logits_for_metrics
# )

# # Start training
# trainer.train()
import subprocess

# Define the command as a list of arguments
command = [
    "python3",
    "/home/ubuntu/adapters/examples/pytorch/language-modeling/run_mlm_unipelt.py",
    "--model_name_or_path", "/opt/dlami/nvme/no_fusion_layer/exp3", # "/opt/dlami/nvme/exp3_pretrained_1epoch",
    "--train_file", "/opt/dlami/nvme/grouped_file.txt",
    "--do_train", "true",
    # "--do_eval", "true",
    "--per_device_train_batch_size", "12",
    # "--per_device_eval_batch_size", "10",
    "--gradient_accumulation_steps", "5",
    # "--eval_accumulation_steps", "5",
    # "--eval_strategy", "steps",
    # "--eval_steps", "2181",  #"10000",
    "--save_steps", "2210",  #"10000",
    "--report_to", "tensorboard",
    "--learning_rate", "2e-5",
    "--num_train_epochs", "3",
    "--output_dir", "/opt/dlami/nvme/no_fusion_layer/knowledge_consolidation",
    "--overwrite_output_dir", "true",
    # "--load_best_model_at_end", "true",
    # "--optim", "adamw_8bit",
    # "--gradient_checkpointing", "true",
    "--adapter_type", 'knowledge_consolidation',
    "--bf16", "true", 
    "--bf16_full_eval", "true",
    "--preprocessing_num_workers", "16",
    "--weight_decay", "0.01",
    "--adam_beta1", "0.9",
    "--adam_beta2", "0.999",
    "--adam_epsilon", "1e-8",
    "--max_grad_norm", "0.5",
    "--lr_scheduler_type", "cosine",
    "--warmup_ratio", "0.1",
]

# Print the command to debug
print("Running command:", " ".join(command))

# Execute the command using subprocess
subprocess.run(command)

# gradient checkpointing - RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [10, 1, 1024, 1024]] is at version 7; expected version 5 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
# fp16 - RuntimeError: Function 'ScaledDotProductFlashAttentionBackward0' returned nan values in its 0th output.