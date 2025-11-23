import copy, os
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss, BCEWithLogitsLoss, CosineEmbeddingLoss
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr

from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup

import sys
sys.path.append('../')
import utils

from tdc.benchmark_group import admet_group
from tdc.single_pred import ADME, Tox


def to_canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None

class DistillationDataset(Dataset):
    def __init__(self, df, tokenizer, smiles_col, target_col=None, ref_embeddings=None):
        self.df = df
        self.tokenizer = tokenizer
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.ref_embeddings = ref_embeddings # Tensor [N, 768]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row[self.smiles_col]
        target = row[self.target_col] if self.target_col is not None else 0.0
        # Return ref embedding if it exists (for training), else empty
        ref_emb = self.ref_embeddings[idx] if self.ref_embeddings is not None else torch.empty(0)
        return smiles, target, ref_emb
    
def create_collator(tokenizer):
    def collator(batch):
        try:
            smiles_list, targets_list, ref_emb_list = zip(*batch)
            encodings = tokenizer(list(smiles_list), padding=True, truncation=True, return_tensors='pt', max_length=512)
            
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
            targets = torch.tensor(targets_list, dtype=torch.float)
            
            if ref_emb_list[0].numel() > 0:
                ref_embs = torch.stack(ref_emb_list)
            else:
                ref_embs = None
                
            return input_ids, attention_mask, targets, ref_embs
        except Exception as e:
            print(f'Error in collator: {e}')
            return None
    return collator

def precompute_teacher(df, tokenizer):
    print("  -> Precomputing Frozen Teacher Embeddings...")
    model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
    model.to(DEVICE)
    model.eval()
    
    # Simple dataset just for extraction
    ds = DistillationDataset(df, tokenizer, 'smiles', None, None)
    col = create_collator(tokenizer)
    loader = DataLoader(ds, batch_size=64, collate_fn=col, shuffle=False, drop_last=False)
    
    all_embs = []
    with torch.no_grad():
        for input_ids, attention_mask, _, _ in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            out = model(input_ids, attention_mask, output_hidden_states=True)
            # Extract last layer
            last_hidden = out.hidden_states[-1]
            emb = (attention_mask.unsqueeze(2) * last_hidden).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(1)
            all_embs.append(emb.cpu())
    
    del model
    torch.cuda.empty_cache()
    return torch.cat(all_embs, dim=0)

class RegularizedTransformer(nn.Module):
    def __init__(self, reg_drop_rate=0.1, intermediate_size=256, num_targets=1):
        super(RegularizedTransformer, self).__init__()
        self.hidden_size = 768
        self.transformer = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
        
        # Head connects to Penultimate Layer (-2)
        self.regressor = nn.Sequential(
            nn.Dropout(reg_drop_rate),
            nn.Linear(self.hidden_size, intermediate_size),
            nn.SiLU(),
            nn.Dropout(reg_drop_rate),
            nn.Linear(intermediate_size, num_targets)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask, output_hidden_states=True)
        
        # --- Task Branch: Penultimate Layer (-2) ---
        penultimate = outputs.hidden_states[-2]
        task_emb = (attention_mask.unsqueeze(2) * penultimate).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(1)
        pred = self.regressor(task_emb)
        
        # --- Regularization Branch: Last Layer (-1) ---
        last = outputs.hidden_states[-1]
        last_emb = (attention_mask.unsqueeze(2) * last).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(1)
        
        return pred, last_emb


tasks = ['caco2_wang', 'bioavailabilityma']
REG_METHOD = 'mse' #cosine
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


dfs = []

# Define the Grid for Lambda (Regularization Strength)
lambda_grid = [0.0, 1.0] #, 0.1, 0.5, 1.0, 2.0]

for task in tasks:
    if task.startswith('.'): continue
    print(f"\n=== Processing Task: {task} ===")

    # --- 1. Task Metadata Setup ---
    prefix = 'tdcommons/'
    if prefix+task in utils.tdc_mae_tasks:
        metric = 'mae'
    elif prefix+task in utils.tdc_spearman_task:
        metric = 'spearman'
    elif prefix+task in utils.tdc_auroc_tasks:
        metric = 'auc'
    elif prefix+task in utils.tdc_aucpr_tasks:
        metric = 'aucpr'
    elif prefix+task in utils.tdc_aucpr2_tasks:
        metric = 'aucpr'
    else:
        # Fallback or error
        print(f"Warning: {task} metric not found, defaulting to MAE")
        metric = 'mae'

    try:
        data = ADME(name = task)
    except:
        data = Tox(name = task)

    split = data.get_split(method = 'scaffold')
    
    train_df = split['valid'].rename({'Drug': 'smiles', 'Y': 'target'}, axis=1).drop('Drug_ID', axis=1)
    val_df = split['valid'].rename({'Drug': 'smiles', 'Y': 'target'}, axis=1).drop('Drug_ID', axis=1)
    test_df = split['test'].rename({'Drug': 'smiles', 'Y': 'target'}, axis=1).drop('Drug_ID', axis=1)
    
    train_df['smiles'] = train_df['smiles'].apply(to_canonical)
    val_df['smiles'] = val_df['smiles'].apply(to_canonical)
    test_df['smiles'] = test_df['smiles'].apply(to_canonical)

    
    # --- 2. Scaling ---
    scaler = None
    if metric in ('mae', 'spearman', 'pearson'):
        scaler = StandardScaler()
        train_vals = train_df[['target']].values
        scaler.fit(train_vals)
        train_df['target_scaled'] = scaler.transform(train_vals)
        val_df['target_scaled']   = scaler.transform(val_df[['target']].values)
        test_targets_orig = test_df['target'].values
        test_df['target_scaled']  = scaler.transform(test_df[['target']].values)
        target_col = 'target_scaled'
    else:
        target_col = 'target'

    # --- 3. Precompute Teacher Embeddings ---
    # We only need reference embeddings for the TRAIN set
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    collator = create_collator(tokenizer)
    
    # THIS IS THE NEW STEP
    train_ref_embeddings = precompute_teacher(train_df, tokenizer)

    # --- 4. Dataloaders ---
    # Train loader gets the teacher embeddings
    train_dataset = DistillationDataset(train_df, tokenizer, 'smiles', target_col, ref_embeddings=train_ref_embeddings)
    train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn=collator, shuffle=True, drop_last=True)

    # Val/Test do not need teacher embeddings
    val_dataset = DistillationDataset(val_df, tokenizer, 'smiles', target_col, ref_embeddings=None)
    val_dataloader = DataLoader(val_dataset, batch_size=512, collate_fn=collator, shuffle=False, drop_last=False)
    
    test_dataset = DistillationDataset(test_df, tokenizer, 'smiles', target_col, ref_embeddings=None)
    test_dataloader = DataLoader(test_dataset, batch_size=512, collate_fn=collator, shuffle=False, drop_last=False)

    # --- 5. Regularization Loop ---
    test_metrics = []
    
    for reg_lambda in lambda_grid:
        print(f'\n--- Lambda: {reg_lambda} ---')
        
        # Best model tracking for this lambda
        best_lambda_score = -float('inf') if metric != 'mae' else float('inf')
        best_lambda_state = None

        # LR Sweep (Inner Loop)
        for lr in [5e-5, 1e-4]: #[1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
            print(f'  LR={lr}...')
            
            model = RegularizedTransformer().to(DEVICE)
            optimizer = AdamW(model.parameters(), lr=lr)
            
            # Loss Functions
            if metric in ('mae', 'spearman', 'pearson'):
                task_loss_fn = MSELoss()
            else:
                task_loss_fn = BCEWithLogitsLoss()
            
            # Reg Loss
            if REG_METHOD == 'mse':
                reg_loss_fn = MSELoss()
            else:
                reg_loss_fn = CosineEmbeddingLoss()

            epochs = 5
            num_training_steps = len(train_dataloader) * epochs
            num_warmup_steps = int(0.05 * num_training_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

            # Tracking per LR
            best_val = float('inf') if metric == 'mae' else -float('inf')
            best_state = None

            for epoch in range(epochs):
                model.train()
                for input_ids, attention_mask, targets, ref_embs in train_dataloader:
                    input_ids = input_ids.to(DEVICE)
                    attention_mask = attention_mask.to(DEVICE)
                    targets = targets.to(DEVICE).float()
                    ref_embs = ref_embs.to(DEVICE) # Teacher embeddings

                    optimizer.zero_grad()
                    
                    # Forward pass returns (Prediction, Current_Last_Layer_Emb)
                    preds, current_last_emb = model(input_ids, attention_mask)
                    preds = preds.squeeze()
                    
                    # 1. Task Loss
                    loss_task = task_loss_fn(preds, targets)
                    
                    # 2. Regularization Loss
                    if reg_lambda > 0.0:
                        if REG_METHOD == 'mse':
                            loss_reg = reg_loss_fn(current_last_emb, ref_embs)
                        else:
                            # Target 1.0 means "Make them similar"
                            target_ones = torch.ones(current_last_emb.size(0)).to(DEVICE)
                            loss_reg = reg_loss_fn(current_last_emb, ref_embs, target_ones)
                        
                        total_loss = loss_task + (reg_lambda * loss_reg)
                    else:
                        total_loss = loss_task

                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()

                # --- Validation ---
                model.eval()
                val_preds, val_targs = [], []
                with torch.no_grad():
                    for input_ids, attention_mask, targets, _ in val_dataloader:
                        input_ids = input_ids.to(DEVICE)
                        attention_mask = attention_mask.to(DEVICE)
                        # Note: We discard the embedding output during inference
                        p, _ = model(input_ids, attention_mask)
                        p = p.squeeze(-1).cpu().numpy()
                        val_preds.extend(p.tolist())
                        val_targs.extend(targets.numpy().tolist())

                # Compute metric
                if metric == 'mae':
                    val_score = np.mean(np.abs(np.array(val_preds) - np.array(val_targs)))
                elif metric == 'spearman':
                    val_score = spearmanr(val_targs, val_preds)[0]
                elif metric == 'pearson':
                    val_score = pearsonr(val_targs, val_preds)[0]
                elif metric == 'auc':
                    try: val_score = roc_auc_score(val_targs, val_preds)
                    except: val_score = 0.5
                elif metric == 'aucpr':
                    try: val_score = average_precision_score(val_targs, val_preds)
                    except: val_score = 0.0

                improved = (metric == 'mae' and val_score < best_val) or (metric != 'mae' and val_score > best_val)
                if improved:
                    best_val = val_score
                    best_state = copy.deepcopy(model.state_dict())
            
            # End of Epochs for this LR -> Check if this was the best LR so far for this Lambda
            lr_improved = (metric == 'mae' and best_val < best_lambda_score) or (metric != 'mae' and best_val > best_lambda_score)
            if lr_improved:
                best_lambda_score = best_val
                best_lambda_state = best_state

        # --- Testing for this Lambda ---
        # Load best model across all LRs for this specific Lambda
        model = RegularizedTransformer().to(DEVICE)
        model.load_state_dict(best_lambda_state)
        model.eval()
        
        test_preds, test_targs = [], []
        with torch.no_grad():
            for input_ids, attention_mask, targets, _ in test_dataloader:
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                p, _ = model(input_ids, attention_mask)
                p = p.squeeze(-1).cpu().numpy()
                test_preds.extend(p.tolist())
                test_targs.extend(targets.numpy().tolist())

        # Inverse Transform if Regression
        if metric in ('mae', 'spearman', 'pearson'):
            test_preds = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten()
            test_targs = test_targets_orig

        # Final Metric Calculation
        if metric == 'mae':
            test_score = np.mean(np.abs(np.array(test_preds) - np.array(test_targs)))
        elif metric == 'spearman':
            test_score = spearmanr(test_targs, test_preds)[0]
        elif metric == 'pearson':
            test_score = pearsonr(test_targs, test_preds)[0]
        elif metric == 'auc':
            test_score = roc_auc_score(test_targs, test_preds)
        elif metric == 'aucpr':
            test_score = average_precision_score(test_targs, test_preds)

        print(f"Result for Lambda {reg_lambda}: {test_score}")
        test_metrics.append(test_score)

    # Save Results for this Task
    # Columns will be tasks, Index will be Lambdas
    results_df = pd.DataFrame({task: test_metrics}, index=lambda_grid)
    results_df.index.name = f'lambda_{REG_METHOD}'
    dfs.append(results_df)

    # Ensure tmp directory exists
    os.makedirs('tmp', exist_ok=True)

    results_df.to_csv(f"tmp/molf_{task}_lambda_results.csv", index=True)

# Final Aggregation
dfs = pd.concat(dfs, axis=1)
dfs.to_csv(f'./results_molf_finetune_regularized_{REG_METHOD}.csv', index=True)
print("Done.")