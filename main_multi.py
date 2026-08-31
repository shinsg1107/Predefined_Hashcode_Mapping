import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import datetime
import json
import random
import itertools

from copy import deepcopy
from itertools import cycle, islice
from tqdm import tqdm
from pathlib import Path
from sconf import Config
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning.samplers import MPerClassSampler

from models import DNAEncoderWithClassifier, DNAEncoderWithClassifierEva, DNAEncoderWithClassifierSwint
from loss import MaskedOtLoss, MaskedOtLoss_jc
from tools import (set_seed,
                   log,
                   perform_mapping,
                   perform_soft_mapping,
                   perform_multilabel_mapping,
                   build_inputs_for_mapping,
                   calculate_pr_curve_multi,
                   )
from utils.data_utils import (save_config_file,
                              data_load,
                              DnaDataset,
                              ImgfeatureDataset)
from utils.evaluation_metric import calculate_map_at_k, hash_distribution_concentration, calculate_map_at_k_multi
from utils.cosinesche import CosineAnnealingWarmupRestarts

seed = 42
set_seed(seed)

def main(config, mode):
    if mode == "train":
        train(config)
    elif mode == "test":
        test(config)

def train(config):
    device = config.device
    # Override with config.seed when present (default 42) — for multi-seed experiments
    try:
        _seed = int(config.seed)
    except (KeyError, AttributeError):
        _seed = 42
    set_seed(_seed)
    num_label = config.num_label
    embedding_size, hash_train_path, hash_test_path ,img_train_path, img_query_path, img_db_path = data_load(config)
    logging.basicConfig(filename=Path(config.encoder_save_path) / "train.log", level=logging.INFO, format="%(message)s")
    summary_writer = SummaryWriter(log_dir=config.encoder_save_path)

    #data
    dna_dataset = DnaDataset(hash_train_path, img_train_path, device, num_label)

    if config.dataset in ["imagenet", "imagenet_swin", "imagenet_eva"]:
        hash_labels = [sample[1] for sample in dna_dataset]
        hash_sampler = MPerClassSampler(hash_labels, 
                                        m=config.m, 
                                        length_before_new_iter=len(hash_labels), 
                                        batch_size=config.batch_size)
        dna_train_loader = DataLoader(dna_dataset, 
                                      sampler=hash_sampler, 
                                      batch_size=config.batch_size)
    else:
        dna_train_loader = DataLoader(dna_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    img_train_dataset = ImgfeatureDataset(img_train_path)
       
    if config.dataset in ["imagenet", "imagenet_swin", "imagenet_eva"]:
        img_labels = [sample[1] for sample in img_train_dataset]
        img_sampler = MPerClassSampler(img_labels,
                                   m=config.m,
                                   length_before_new_iter=len(img_labels),
                                   batch_size=config.batch_size)
        img_train_loader = DataLoader(img_train_dataset, sampler=img_sampler, batch_size=config.batch_size)
    else: 
        img_train_loader = DataLoader(img_train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    
    #model
    if embedding_size == 768:
        dna_model = DNAEncoderWithClassifier(embedding_size, num_label, config.bit * 2).to(device)
    elif embedding_size == 1408:
        dna_model = DNAEncoderWithClassifierEva(embedding_size, num_label, config.bit * 2).to(device)
    elif embedding_size == 1536:
        dna_model = DNAEncoderWithClassifierSwint(embedding_size, num_label, config.bit * 2).to(device)
    else:
        raise ValueError("Unsupported embedding size.")

    #loss
    try:
        _mismatch_penalty = config.mismatch_penalty
    except (KeyError, AttributeError):
        _mismatch_penalty = 1e9
    try:
        _log_domain = config.log_domain
    except (KeyError, AttributeError):
        _log_domain = False
    clsloss = nn.CrossEntropyLoss()
    masked_otloss = MaskedOtLoss(reg=config.ot_reg, max_iter=config.ot_maxiter,
                                 mismatch_penalty=_mismatch_penalty, log_domain=_log_domain)

    clsloss_multilabel = nn.BCEWithLogitsLoss()
    masked_otloss_jc = MaskedOtLoss_jc(reg=config.ot_reg, max_iter=config.ot_maxiter,
                                       mismatch_penalty=_mismatch_penalty)

    #loss parameter
    a = config.alpha
    b = config.beta

    optimizer = optim.Adam(dna_model.parameters(), lr=0.00001)    
    best_map_score = 0.0

    for epoch in range(config.epochs):
        dna_model.train()
        epoch_loss = 0
        total_correct = 0
        total_samples = 0
        batches_done = 0
        map_all = map_10k = map_5k = map_k = 0.0

        long_is_img = len(img_train_loader) >= len(dna_train_loader)
        steps = max(len(img_train_loader), len(dna_train_loader))

        it_img = iter(img_train_loader) if long_is_img else cycle(img_train_loader)
        it_dna = cycle(dna_train_loader) if long_is_img else iter(dna_train_loader)
        paired = islice(zip(it_img, it_dna), steps)

        all_true = []
        all_pred = []

        with tqdm(paired, total=steps, desc=f"Epoch {epoch+1}/{config.epochs}") as t:
            for (images, img_labels), (code, anchor_labels) in t:
                code = code.to(device, dtype=torch.float32) 
                #anchor_labels = anchor_labels.to(device)

                images = images.to(device, dtype=torch.float32) 
                img_labels = img_labels.to(device)
                hash_embedding, logits = dna_model(code) 
                if config.dataset in ['nuswide','nuswide_eva', 'nuswide_swin']:
                    anchor_labels = torch.eye(num_label)[anchor_labels].to(device)

                    loss_a, _ = masked_otloss_jc(hash_embedding, images, anchor_labels, img_labels)
                    loss_b = clsloss_multilabel(logits, anchor_labels)

                    loss =  a* loss_a + b *loss_b
                    predictions = torch.softmax(logits, dim=1) 
                    final_preds = (predictions > 0.5).float()
                    all_true.append(img_labels.cpu()) 
                    all_pred.append(final_preds.cpu())
                    
                else:
                    loss_a, _ = masked_otloss(hash_embedding, images, anchor_labels, img_labels)
                    loss_b = clsloss(logits, anchor_labels)

                    loss =  a* loss_a + b *loss_b

                    predictions = torch.argmax(logits, dim=1)
                    correct = (predictions == anchor_labels).sum().item()
                    total_correct += correct
                    total_samples += anchor_labels.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches_done += 1
                t.set_postfix(loss=loss.item())

        if config.dataset in ['nuswide','nuswide_eva', 'nuswide_swin']:
            all_true = torch.cat(all_true, dim=0).numpy()
            all_pred = torch.cat(all_pred, dim=0).numpy()

            intersection = np.logical_and(all_true, all_pred).sum()
            union = np.logical_or(all_true, all_pred).sum()
            epoch_accuracy = intersection / union if union != 0 else 0.0
            print(f"[Epoch {epoch+1}]| Accuracy: {epoch_accuracy:.4f} | Loss: {epoch_loss/len(dna_train_loader):.4f}")

        else:
            epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            if batches_done > 0:
                epoch_accuracy = total_correct / total_samples
                avg_loss = epoch_loss / batches_done
            else:
                epoch_accuracy, avg_loss = 0.0, 0.0
            print(f"[Epoch {epoch+1}] Accuracy: {epoch_accuracy:.4f} | Loss: {avg_loss:.4f}")
        dna_model.eval()

        # --- save-every-epoch mode (for offline eval): save each epoch + skip in-training eval ---
        try:
            _save_every = config.save_every_epoch
        except (KeyError, AttributeError):
            _save_every = False
        if _save_every:
            sp = Path(config.encoder_save_path) / f"model_epoch{epoch+1}.pt"
            torch.save(dna_model.state_dict(), sp)
            continue

        # map
        try:
            eval_every = config.eval_every
        except (KeyError, AttributeError):
            eval_every = 1
        if (epoch % eval_every) == 0:
            num_classes = config.num_label
            hash_embeddings, anchor_labels, all_binary_codes = build_inputs_for_mapping(
                dna_train_loader, dna_model, device, num_classes, config.dataset
            )

            mapping_result = perform_multilabel_mapping(
                img_query_path,
                img_db_path,
                hash_embeddings,
                anchor_labels,      # (K,) or (K,C) — multi-label supported
                all_binary_codes,   # (K,B) — converted to uint8 internally with a >0 threshold
                device=device
            )
            
            map_all = calculate_map_at_k_multi(mapping_result, topk=None, device=device)
            map_10k = calculate_map_at_k_multi(mapping_result, topk=10000, device=device)
            map_5k = calculate_map_at_k_multi(mapping_result, topk=5000, device=device)
            map_1k = calculate_map_at_k_multi(mapping_result, topk=1000, device=device)

            print(f"\n{'='*15} Epoch: {epoch+1} mAP Results {'='*15}")
            print(f"| mAP @ All:  {map_all:.4f}")
            print(f"| mAP @ 10k:  {map_10k:.4f}")
            print(f"| mAP @ 5k:   {map_5k:.4f}")
            print(f"| mAP @ 1k:   {map_1k:.4f}")
            print(f"{'='*51}\n")
    
        #logging
        log(summary_writer, "loss/ot", loss_a, epoch)
        log(summary_writer, "loss/cls", loss_b, epoch)
        log(summary_writer, "acc/train", epoch_accuracy, epoch)
        if all(x != 0.0 for x in [map_all, map_10k, map_5k, map_k]):
            log(summary_writer, "map/all", map_all, epoch)
            log(summary_writer, "map/10k", map_10k, epoch)
            log(summary_writer, "map/5k", map_5k, epoch)
            log(summary_writer, "map/1k", map_1k, epoch) 

        #save model
        current_map_score = 0.0
        if config.dataset == "cifar-10":
            current_map_score = map_all
        elif config.dataset in ["imagenet", "imagenet_swin", "imagenet_eva"] :
            current_map_score = map_1k
        else:
            current_map_score = map_5k

        if current_map_score > best_map_score:
            print(f"New best mAP score found! {best_map_score:.4f} -> {current_map_score:.4f}")
            best_map_score = current_map_score 
            save_path = Path(config.encoder_save_path) / f"model_epoch{epoch+1}_map{best_map_score:.4f}.pt"
            torch.save(dna_model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    summary_writer.flush()
    summary_writer.close()
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
def _labels_to_numpy(labels_t, num_classes):
    """
    labels_t: torch.Tensor
      - (B,) int/long  => converted to one-hot
      - (B, C) float/bool/int => returned as-is in multi-hot form (assumed to be 0/1)
    return: np.ndarray shape (B, C)
    """
    if labels_t.dim() == 1:
        return torch.nn.functional.one_hot(labels_t.long(), num_classes=num_classes).cpu().numpy()
    elif labels_t.dim() == 2:
        return labels_t.float().cpu().numpy()
    else:
        raise ValueError(f"Unexpected label shape: {labels_t.shape}")


def _encode_hash_set(dna_loader, dna_model, device, num_classes, is_multilabel=False):
    """
    Collect hash embeddings / labels / raw (binary) codes from dna_loader and return them as numpy.
    If is_multilabel=True, labels are returned as 2D (np.float32, 0/1).
    """
    embeddings, labels_np_list, raw_codes_list = [], [], []
    with torch.no_grad():
        for code, label in tqdm(dna_loader, desc="Encoding Hash"):
            raw_codes_list.append(code.numpy())
            code = code.to(device, dtype=torch.float32)
            emb, _ = dna_model(code)
            embeddings.append(emb.cpu().numpy())

            if is_multilabel:
                labels_np_list.append(_labels_to_numpy(label, num_classes))
            else:
                labels_np_list.append(label.cpu().numpy())

    hash_embeddings = np.concatenate(embeddings, axis=0)
    all_binary_codes = np.concatenate(raw_codes_list, axis=0)
    hash_labels = np.concatenate(labels_np_list, axis=0)
    return hash_embeddings, hash_labels, all_binary_codes


def _eval_map_block(img_query_path, img_db_path,
                    hash_embeddings, hash_labels, all_binary_codes,
                    device, is_multilabel=False):
    """
    mAP computation block: picks the mapping/evaluation function based on multi-label or not.
    """
    if is_multilabel:
        # Multi-label version
        mapping_result = perform_multilabel_mapping(
            img_query_path, img_db_path,
            hash_embeddings,  # (K, D)
            hash_labels,      # (K, C)  multi-hot (0/1)
            all_binary_codes, # (K, B)  uint8 expected. can be handled internally via >0
            device=device
        )
        map_all = calculate_map_at_k_multi(mapping_result, topk=None, device=device)
        map_10k = calculate_map_at_k_multi(mapping_result, topk=10000, device=device)
        map_5k  = calculate_map_at_k_multi(mapping_result, topk=5000, device=device)
        map_1k  = calculate_map_at_k_multi(mapping_result, topk=1000, device=device)
    else:
        # Single-label version
        mapping_result = perform_mapping(
            img_query_path, img_db_path,
            hash_embeddings,  # (K, D)
            hash_labels,      # (K,)
            all_binary_codes, # (K, B)
            device
        )
        map_all = calculate_map_at_k(mapping_result, topk=None, device=device)
        map_10k = calculate_map_at_k(mapping_result, topk=10000, device=device)
        map_5k  = calculate_map_at_k(mapping_result, topk=5000, device=device)
        map_1k  = calculate_map_at_k(mapping_result, topk=1000, device=device)

    return map_all, map_10k, map_5k, map_1k, mapping_result



def test(config):
    if config.model_path is None:
        raise ValueError("Model path must be specified for testing.")
    device = config.device
    num_label = config.num_label
    (embedding_size,
     hash_train_path, hash_test_path,
     img_train_path, img_query_path, img_db_path) = data_load(config)

    # Whether the dataset is multi-label
    MULTISETS = ['nuswide', 'nuswide_eva', 'nuswide_swin']
    is_multilabel = (config.dataset in MULTISETS)

    # Prepare loaders
    dna_dataset = DnaDataset(hash_train_path, img_train_path, device, num_label)  # set1
    dna_test_dataset = DnaDataset(hash_test_path, img_train_path, device, num_label)  # set2
    dna_loader = DataLoader(dna_dataset, batch_size=config.batch_size, shuffle=False)
    dna_test_loader = DataLoader(dna_test_dataset, batch_size=config.batch_size, shuffle=False)

    # Model
    if embedding_size == 768:
        dna_model = DNAEncoderWithClassifier(embedding_size, num_label, config.bit * 2).to(device)
    elif embedding_size == 1408:
        dna_model = DNAEncoderWithClassifierEva(embedding_size, num_label, config.bit * 2).to(device)
    elif embedding_size == 1536:
        dna_model = DNAEncoderWithClassifierSwint(embedding_size, num_label, config.bit * 2).to(device)
    else:
        raise ValueError("Unsupported embedding size.")

    dna_model.load_state_dict(torch.load(config.model_path, map_location=device))
    dna_model.eval()

    #----- Set 1 -----
    hash_embeddings, hash_labels, all_binary_codes = _encode_hash_set(
        dna_loader, dna_model, device, num_classes=num_label, is_multilabel=is_multilabel
    )
    map_all, map_10k, map_5k, map_1k, mapping_result = _eval_map_block(
        img_query_path, img_db_path,
        hash_embeddings, hash_labels, all_binary_codes,
        device=device, is_multilabel=is_multilabel
    )
    print(f"\n{'='*15} Set1 mAP Results {'='*15}")
    print(f"| mAP @ All:  {map_all:.4f}")
    print(f"| mAP @ 10k:  {map_10k:.4f}")
    print(f"| mAP @ 5k:   {map_5k:.4f}")
    print(f"| mAP @ 1k:   {map_1k:.4f}")
    print(f"{'='*45}\n")

    P, R, index = calculate_pr_curve_multi(mapping_result, interval=10, multilabel=True)
    
    # 2. Build the data structure to save as JSON
    pr_log_data = {
        "index": index,
        "P": P,
        "R": R
    }
    
    # 3. Define the output file name (built dynamically from the config values)
    log_filename = f"pr_log_new{config.dataset}_{config.bit}nt.json"
    
    # 4. Save the JSON file
    with open(log_filename, 'w') as f:
        json.dump(pr_log_data, f, indent=4)
        
    print(f"P-R log has been successfully saved to: {log_filename}")

    print("Clearing CUDA cache after test map...")



    # ----- Set 2 -----
    hash_embeddings, hash_labels, all_binary_codes = _encode_hash_set(
        dna_test_loader, dna_model, device, num_classes=num_label, is_multilabel=is_multilabel
    )
    map_all, map_10k, map_5k, map_1k, _= _eval_map_block(
        img_query_path, img_db_path,
        hash_embeddings, hash_labels, all_binary_codes,
        device=device, is_multilabel=is_multilabel
    )
    print(f"\n{'='*15} Set2 mAP Results {'='*15}")
    print(f"| mAP @ All:  {map_all:.4f}")
    print(f"| mAP @ 10k:  {map_10k:.4f}")
    print(f"| mAP @ 5k:   {map_5k:.4f}")
    print(f"| mAP @ 1k:   {map_1k:.4f}")
    print(f"{'='*51}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=False)
    args, left_argv = parser.parse_known_args()
    config = Config(args.config)
    config.argv_update(left_argv)
    config.exp_name = os.path.basename(args.config).split(".")[0]
    config.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config.past_epochs = 0
    config.encoder_save_path = Path(config.encoder_save_path) / config.exp_name / config.exp_version
    embedding_size, hash_train_path, hash_test_path ,img_train_path, img_query_path, img_db_path = data_load(config)
    config.train_data = hash_train_path
    config.test_data = hash_test_path
    config.img_train_data = img_train_path
    
    if args.model_path:
        config.model_path = args.model_path
    if args.mode == 'train':
        save_config_file(config, config.encoder_save_path)
    main(config, args.mode)