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
                          TrainerCallback,
                          set_seed,
                          Trainer
                         )
from datasets import Dataset, DatasetDict
device = 'cuda' if torch.cuda.is_available() else 'cpu'
adapters.__file__
import torch
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #Load and prepare data (common for both experiments)
# df = pd.read_excel("/opt/dlami/nvme/plos_all.xlsx")

# def create_dataframe(df, split):
#     selected_df = df[df["Split"] == split][["Abstract", "Summary"]].rename(columns={"Abstract": "input_text", "Summary": "target_text"})
#     return selected_df

# train_df, test_df, val_df = create_dataframe(df, "train"), create_dataframe(df, "test"), create_dataframe(df, "val")
# train_dataset, val_dataset, test_dataset = Dataset.from_dict(train_df), Dataset.from_dict(val_df), Dataset.from_dict(test_df)

# data_path = "/opt/dlami/nvme/"

# train_df = pd.read_csv(data_path + 'train.csv', usecols = ['input_text', 'target_text'])
# val_df = pd.read_csv(data_path + 'val.csv', usecols = ['input_text', 'target_text'])
# test_df = pd.read_csv(data_path + 'test.csv', usecols = ['input_text', 'target_text'])

# train_dataset, val_dataset, test_dataset = Dataset.from_dict(train_df), Dataset.from_dict(val_df), Dataset.from_dict(test_df)
# dataset = DatasetDict({
#     "train": train_dataset,
#     "test": test_dataset,
#     "val": val_dataset
# })

# # Process data
max_length = 1024

# def process_data(batch, tokenizer):
#     model_max_length = tokenizer.model_max_length
#     inputs = tokenizer(batch["input_text"], padding="max_length", max_length=model_max_length, truncation=True)
#     outputs = tokenizer(batch["target_text"], padding="max_length", max_length=model_max_length, truncation=True)
#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask
#     batch["labels"] = outputs.input_ids
#     batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
#     return batch

output_file1 = "/opt/dlami/nvme/plos_bart_processed_dataset.pkl"
output_file2 = "/opt/dlami/nvme/plaba_bart_processed_dataset.pkl"

import pickle
with open(output_file1, 'rb') as file:
    dataset1 = pickle.load(file)
    
with open(output_file2, 'rb') as file:
    dataset2 = pickle.load(file)

def run_experiment(exp_number, pretrained_path, adapter_name, output_dir, processed_dataset):
    # Load model and tokenizer
    model = AutoAdapterModel.from_pretrained(pretrained_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained("mse30/bart-base-finetuned-pubmed")
    print(model)

    # Set up model for fine-tuning
    adapter_setup = Fuse("adapter1", "adapter2", "adapter3")
    # model.delete_head(adapter_name)
    # model.add_seq2seq_lm_head('fine_tunning')
    model.train_adapter_fusion([adapter_setup, "adapter1", "adapter2", "adapter3"], unfreeze_adapters=True, train_embeddings=True)
    #model.train_adapter([adapter_name], train_embeddings=True)
    # print(model)

    # Make all layers trainable
    for param in model.parameters():
        param.requires_grad = True

    # Process dataset with the current tokenizer
    # processed_dataset = dataset.map(lambda batch: process_data(batch, tokenizer), batched=True, remove_columns=['input_text', 'target_text'])
    # processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Define test function
    def test_model(model, tokenizer, max_length, test_dataset):
        model.eval()
        model.to(device)

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

    # Define callback
    # class TestAfterEpochCallback(TrainerCallback):
    #     def __init__(self, test_dataset, tokenizer, max_length, output_dir):
    #         self.test_dataset = test_dataset
    #         self.tokenizer = tokenizer
    #         self.max_length = max_length
    #         self.output_dir = output_dir

    #     def on_epoch_end(self, args, state, control, model, **kwargs):
    #         print(f"Testing model after epoch {state.epoch}")
    #         decoded_preds = test_model(model, self.tokenizer, self.max_length, self.test_dataset)
            
    #         with open(f"{self.output_dir}/exp{exp_number}_finetuned_plos_{state.epoch}.txt", "w") as f:
    #             for pred in decoded_preds:
    #                 f.write(pred + "\n")

    # test_dataset = Dataset.from_dict(test_df[:20])
    # test_dataset = test_dataset.map(lambda batch: process_data(batch, tokenizer), batched=True, remove_columns=['input_text', 'target_text'])
    # test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # test_callback = TestAfterEpochCallback(test_dataset, tokenizer, max_length, output_dir)

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=10,
        per_device_train_batch_size=10,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        remove_unused_columns=True,
        save_strategy="epoch",
        save_total_limit=1,
        num_train_epochs=5,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        learning_rate=2e-5,
        fp16=True,
        fp16_full_eval=True,
        optim="adamw_bnb_8bit",
        seed=42,
        report_to="none",
    )

    # Set Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Train model
    trainer = Seq2SeqAdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['val'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3), test_callback],
    )

    trainer.train()
    trained_model = trainer.model

    # Test model and save predictions
    model_preds = test_model(model, tokenizer, max_length, processed_dataset['test'])

    with open(f'{output_dir}/exp{exp_number}_finetuned_plaba.txt', 'w') as file:
        for string in model_preds:
            file.write(string + "\n")

# Run both experiments
experiments = [
    {
        "exp_number": 1,
        "pretrained_path": "/opt/dlami/nvme/exp1_finetuned_plaba",
        "adapter_name": "adapter1",
        "output_dir": "/opt/dlami/nvme/exp1_finetuned_plaba"
    },
    {
        "exp_number": 2,
        "pretrained_path": "/opt/dlami/nvme/exp2_pretrained/results",
        "adapter_name": "adapter2",
        "output_dir": "/opt/dlami/nvme/exp2_finetuned_plaba"
    },
    {
        "exp_number": 3,
        "pretrained_path": "/opt/dlami/nvme/exp3_pretrained/results",
        "adapter_name": "umls-synonyms",
        "output_dir": "/opt/dlami/nvme/exp3_finetuned_plaba"
    },
    {
        "exp_number": 4,
        "pretrained_path": "/opt/dlami/nvme/knowledge_consolidation",
        "adapter_name": "knowledge_consolidation",
        #"output_dir": "/opt/dlami/nvme/exp4_finetuned_plos"
    },
]

datasets = [
    #("plos", dataset1, "/opt/dlami/nvme/exp4_defintions_finetuned_plos"),
    ("plaba", dataset2, "/opt/dlami/nvme/exp4_defintions_finetuned_plaba")
]

max_length = 1024

for dataset_name, dataset, output_base_dir in datasets:
    for exp in experiments[3:]:
        run_experiment(**exp, output_dir=output_base_dir, processed_dataset=dataset)
        print(f"Experiment {exp['exp_number']} on {dataset_name} dataset completed")

print("All experiments completed")