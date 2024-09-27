import torch
import os
import time
import math
import json
import argparse
import bitsandbytes as bnb
from transformers import AutoAdapterModel, AutoTokenizer
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner
from tqdm import tqdm
import torch.cuda.amp as amp
from argparse import ArgumentParser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# Define utility function
def create_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Define class for the adapter model
class AdapterModel:
    def __init__(self, configs):
        self.configs = configs

        # Load pre-trained model
        self.model = AutoAdapterModel.from_pretrained(configs['model_path'])
        task_name = 'adapter3'
        
        # Add and train the new adapter
        self.model.delete_head('adapter2')
        # As return_logits is True,  num_negative_samples and margin are redundant
        self.model.add_multiple_choice_head(task_name, layers=1, num_choices=1, num_negative_samples=4, margin=0.5,
                                            activation_function="tanh", overwrite_ok=False, id2label=None,
                                            use_pooler=False, return_logits=True)
        self.model.train_adapter(['adapter3'])

        # Enable gradient checkpointing if required
        if self.configs['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("mse30/bart-base-finetuned-pubmed")
     
        # Initialize Loss Function and Miner
        self.loss_fct = MultiSimilarityLoss(alpha=configs['loss_scale_pos'],
                                            beta=configs['loss_scale_neg'],
                                            base=configs['loss_thresh'])
        self.miner_fct = MultiSimilarityMiner(epsilon=configs['loss_lambda'])

    def compute_loss(self, all_reps, all_labels):
        # Use the miner to get hard positive and hard negative examples
        miner_output = self.miner_fct(all_reps, all_labels)
        
        # Compute the loss using the MultiSimilarityLoss function
        loss = self.loss_fct(all_reps, all_labels, miner_output)
        
        return loss

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the model
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        all_reps = outputs
        all_labels = labels

        # Compute the loss if labels are provided
        if labels is not None:
            loss = self.compute_loss(all_reps, all_labels)
            return loss
        
        return outputs


def create_batches(file_path, batch_size):
    # Read the JSON file
    with open(file_path, 'r') as file:
        umls_dict = json.load(file)
    
    batches = []
    current_batch = []
    
    for key, synonyms in umls_dict.items():
        # Add only the definitions (synonym[1]) and their corresponding labels (key)
        for synonym in synonyms:
            current_batch.append((synonym[1], key))  # synonym[1] is the definition, key is the label
        
        # Check if the current batch size exceeds the batch size
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    
    # Append the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches


def pretrain(configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdapterModel(configs)
    model.model.to(device)

    if configs['optim'] == 'adamw_bnb_8bit':
        optimizer = bnb.optim.AdamW8bit(model.model.parameters(), lr=configs['learning_rate'])
    else:
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=configs['learning_rate'])

    batches = create_batches(configs['ontology_path'], configs['batch_size'])
    num_epoch_steps = len(batches)
    print('Prepared the optimizer and the scheduler')

    print(f"Batch size: {configs['batch_size']}")
    print(f"Epochs: {configs['epochs']}")
    iters = 0
    gradient_accumulation_steps = configs['gradient_accumulation_steps']

    best_loss = float('inf')
    create_dir_if_not_exist(configs['save_dir'])
    scaler = amp.GradScaler(enabled=configs['mixed_precision'])

    total_steps = configs['epochs'] * num_epoch_steps
    start_time = time.time()
    with tqdm(total=total_steps, desc='Training', unit='step') as pbar:
        for epoch in range(configs['epochs']):
            epoch_loss = 0
            for step in range(num_epoch_steps):
                iters += 1
                batch = batches[step]
                definitions = [item[0] for item in batch]
                labels = [item[1] for item in batch]
                
                inputs = model.tokenizer(definitions, padding=True, truncation=True, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                labels = torch.tensor(labels).to(device)

                with amp.autocast(enabled=configs['mixed_precision']):
                    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
                epoch_loss += loss.item() * gradient_accumulation_steps

                if iters % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), configs['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                pbar.update(1)
                pbar.set_postfix_str(f'Loss: {loss.item():.4f}')

            avg_epoch_loss = epoch_loss / num_epoch_steps
            print(f"Epoch {epoch+1}/{configs['epochs']}, Average Loss: {avg_epoch_loss:.4f}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                model.model.save_pretrained(configs['save_dir'])
                torch.save(model.model.state_dict(), os.path.join(configs['save_dir'], 'best_model.pth'))
                print(f'New best model saved with average loss: {best_loss:.4f}')
    
    total_training_time = time.time() - start_time
    print(f'Total training time: {total_training_time:.2f} seconds')

    
def create_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == "__main__":
    # Parse argument
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--gradient_accumulation_steps', type=int, default=10)
    parser.add_argument('--gradient_checkpointing', type=str2bool, default=False)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--use_synthetic_train', type=str2bool, default=False)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=150)  #50
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--loss_scale_pos', type=int, default=2)
    parser.add_argument('--loss_scale_neg', type=int, default=50)
    parser.add_argument('--loss_thresh', type=float, default=0.5)
    parser.add_argument('--loss_lambda', type=float, default=0.2)
    parser.add_argument('--hard_negatives_training', type=str2bool, default=False)
    parser.add_argument('--model_path', type=str, default='/opt/dlami/nvme/no_fusion_layer/exp2/checkpoint-261') 
    parser.add_argument('--dataset', type=str, default='UMLS-2020AA-Full')
    parser.add_argument('--save_dir', type=str, default='/opt/dlami/nvme/no_fusion_layer/exp3')
    parser.add_argument('--ontology_path', type=str, default="/opt/dlami/nvme/umls_dict.json") 
    parser.add_argument('--mixed_precision', type=str2bool, default=True)
    parser.add_argument('--optim', type=str, default='adamw', choices=['adamw', 'adamw_bnb_8bit'])
    parser.add_argument('--learning_rate', type=float, default=2e-5)

    args = parser.parse_args()

    # Convert args to a dictionary
    configs = vars(args)

    # Train
    pretrain(configs)