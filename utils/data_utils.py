import os
import torch
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from sconf import Config
from typing import Union, Optional, Any
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset

from PIL import Image
from tools import (set_seed,
                   one_hot_encode,
                   generate_anchor_codes,
                   code_ot_mapping,
                   )

set_seed(1107)

def save_config_file(config: Config, path: Union[str, os.PathLike], verbose: Optional[bool] = True):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    if verbose:
        print(config.dumps())

    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")

def data_load(config):
    embedding_size = PATH_CONFIG[config.dataset]["emb_size"]
    img_train_path = PATH_CONFIG[config.dataset]["train"]
    img_query_path = PATH_CONFIG[config.dataset]["query"]
    img_db_path = PATH_CONFIG[config.dataset]["db"]
    hash_train_path = PATH_CONFIG['codeset']["train"][config.bit]
    hash_test_path = PATH_CONFIG['codeset']["test"][config.bit]

    # --- experiment #1 (codebook sweep) override ---
    # If config specifies codeset_path / codeset_test_path, use them instead of PATH_CONFIG.
    # Otherwise keep the existing behavior (backward compatible).
    try:
        _train_override = config.codeset_path
    except Exception:
        _train_override = None
    if _train_override:
        hash_train_path = _train_override

    try:
        _test_override = config.codeset_test_path
    except Exception:
        _test_override = None
    if _test_override:
        hash_test_path = _test_override

    return embedding_size, hash_train_path, hash_test_path ,img_train_path, img_query_path, img_db_path

class DnaDataset(Dataset):
    def __init__(self, datapath, img_pkl_path, device, num_label=10, anchor_codeset = None):
        with open(img_pkl_path, "rb") as f:
            self.img_data = pickle.load(f)
        self.code_data = pd.read_csv(datapath)
        self.code = self.code_data['code'].tolist()
        self.sequence_length = len(self.code[0]) 
        self.onehot_code = np.array([one_hot_encode(seq) for seq in self.code], dtype=np.float32)

        if 'label' in self.code_data.columns:
            print("Using provided labels from CSV.")
            self.labels = self.code_data['label'].values
            if self.labels.max() >= num_label:
                print(f"Warning: Provided labels have max {self.labels.max()}, which exceeds num_label={num_label}.")
        else:
            if anchor_codeset is None:
                self.anchor_codes = generate_anchor_codes(seq_length=self.sequence_length, k_select=num_label)
                self.onehot_anchor_code = np.array([one_hot_encode(seq) for seq in self.anchor_codes], dtype=np.float32)
            else:
                self.anchor_codes = pd.read_csv(anchor_codeset)
                self.onehot_anchor_code = self.anchor_codes['code'].tolist()
            self.t_matrix, self.labels = code_ot_mapping(self.onehot_code , self.onehot_anchor_code, device)

        print("\n" + "="*30)
        print("Label Distribution in DnaDataset:")
        labels_np = np.array(self.labels)
        counts = np.bincount(labels_np, minlength=num_label)
        for label_idx, count in enumerate(counts):
            print(f"  Label {label_idx}: {count} samples")
        print("="*30 + "\n")

    def __len__(self):
        return len(self.onehot_code)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, np.ndarray)):
            onehot_code = torch.stack([self.onehot_code[i] for i in idx])
            anchor_label = torch.tensor([self.labels[i] for i in idx], dtype=torch.long)
            return onehot_code, anchor_label  # onehot_code: (B, ...), anchor_label: (B,)
        else:
            onehot_code = self.onehot_code[idx]
            anchor_label = torch.tensor(self.labels[idx], dtype=torch.long)  
            return onehot_code, anchor_label  # anchor_label: scalar tensor

class ImgfeatureDataset(Dataset):
    def __init__(self, datapath):
        with open(datapath, "rb") as f:
            img_data = pickle.load(f)
        self.images = np.array(img_data["features"])
        self.labels = np.array(img_data["labels"])   

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, np.ndarray)):
            images = torch.stack([self.images[i] for i in idx])
            labels = torch.tensor([self.labels[i] for i in idx], dtype=torch.long)

            return images, labels
        else:
            images = torch.tensor(self.images[idx])
            labels = torch.tensor(self.labels[idx], dtype=torch.long)  

            return images, labels
        

def get_idx_for_uniform_sampling(dataset, n_class, n_samples_per_class,offset=0):
    L = np.array(dataset.targets)
    first = True

    for label in range(n_class):
        index = np.where(L == label)[0]
        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]
        index = index[offset:offset+n_samples_per_class]

        if first:
            index_array = index
        else:
            index_array = np.concatenate((index_array, index))

        first = False

    return index_array

class ImageDataset(Dataset):
    def __init__(self, txt_file, processor):
        self.samples = []
        self.processor = processor

        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_path = parts[0]
                label_vector = list(map(int, parts[1:]))
                if sum(label_vector) != 1:
                    continue  # skip invalid
                label = label_vector.index(1)
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # (3, 224, 224)
        return pixel_values, label


# ── Loaders for encoder fine-tuning / feature extraction ─────────────────────
# These take a torchvision/timm `transform` (unlike ImageDataset, which is tied to a
# HuggingFace processor, and ImgRtvCIFAR10, which always applies its default transform),
# so the same code path works for any backbone: the transform is derived per model via
# timm.data.resolve_model_data_config.

class ListImageDataset(Dataset):
    """Image-list file ("<relative path> <l1> <l2> ...") -> (image tensor, label).

    multilabel=False: the one-hot row is converted to a class index; rows whose labels
        do not sum to 1 are skipped (matches the ImageNet-100 list convention).
    multilabel=True:  the row is kept as a float multi-hot vector (NUS-WIDE).
    Paths are resolved against `root`; entries whose image file is missing are skipped,
    so a partially downloaded dataset still loads.
    """

    def __init__(self, txt_file, transform, multilabel=False, root=""):
        self.transform = transform
        self.multilabel = multilabel
        self.samples = []
        self.n_listed = 0        # non-empty rows in the list file
        self.n_skipped_label = 0  # single-label rows whose one-hot did not sum to 1
        self.n_missing = 0        # rows whose image file is absent
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                self.n_listed += 1
                path = os.path.join(root, parts[0]) if root else parts[0]
                vec = list(map(int, parts[1:]))
                if multilabel:
                    label = torch.tensor(vec, dtype=torch.float32)
                else:
                    if sum(vec) != 1:
                        self.n_skipped_label += 1
                        continue
                    label = vec.index(1)
                if not os.path.exists(path):
                    self.n_missing += 1
                    continue
                self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError(
                f"No usable images found for {txt_file} (root={root!r}). "
                "Check that the image files exist at the listed paths.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CIFAR10Split(CIFAR10):
    """CIFAR-10 with the retrieval splits used by PHM, honoring `transform`.

    mode="train"    : 500 images per class (5,000)
    mode="test"     : 100 images per class (1,000)   -- the query set
    mode="database" : the full 60,000 images, train set followed by test set

    `train` and `test` are drawn per class with the RNG re-seeded to `split_seed` for each
    class, and the selected indices are kept grouped by class. This reproduces the sample
    order of the released feature files exactly; changing it changes which images are in
    each split and therefore the reported metrics.
    """

    def __init__(self, root, mode="train", transform=None, download=True, split_seed=1107):
        super().__init__(root, train=True, download=download)
        train_data, train_targets = self.data, np.array(self.targets)
        test_set = CIFAR10(root, train=False, download=download)
        test_data, test_targets = test_set.data, np.array(test_set.targets)

        if mode == "train":
            idx = self._draw(train_targets, 500, split_seed)
            self.data, targets = train_data[idx], train_targets[idx]
        elif mode == "test":
            idx = self._draw(test_targets, 100, split_seed)
            self.data, targets = test_data[idx], test_targets[idx]
        elif mode == "database":
            self.data = np.concatenate((train_data, test_data))
            targets = np.concatenate((train_targets, test_targets))
        else:
            raise ValueError(f"unknown mode: {mode}")

        self.targets = targets.tolist()
        self.transform = transform

    @staticmethod
    def _draw(targets, n_per_class, seed, offset=0, n_class=10):
        idx = []
        for c in range(n_class):
            pool = np.where(targets == c)[0]
            np.random.seed(seed)                       # re-seeded per class, as in the original
            perm = np.random.permutation(pool.shape[0])
            idx.extend(pool[perm][offset:offset + n_per_class])
        return np.array(idx)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class ImgRtvCIFAR10(CIFAR10):
    def __init__(self, root, img_transform=None, target_transform=None, mode="train", setting_name="setting1", return_index=False, return_paired_aug_img=False):
        train_set = CIFAR10(root, train=True, download=True)
        test_set = CIFAR10(root, train=False, download=True)

        if setting_name == "setting1":
            if mode == "train":
                self.data = train_set.data
                self.targets = np.array(train_set.targets)
                idx = self._get_indices(self.targets, n_samples_per_class=500, seed=42)
                self.data = self.data[idx]
                self.targets = self.targets[idx]
            elif mode == "test":
                self.data = test_set.data
                self.targets = np.array(test_set.targets)
                idx = self._get_indices(self.targets, n_samples_per_class=100, seed=42)
                self.data = self.data[idx]
                self.targets = self.targets[idx]
            elif mode == "database":
                test_indices_to_exclude = self._get_indices(np.array(test_set.targets), n_samples_per_class=100, seed=42)
                test_mask = np.ones(len(test_set), dtype=bool)
                test_mask[test_indices_to_exclude] = False
                db_test_data = test_set.data[test_mask]
                db_test_targets = np.array(test_set.targets)[test_mask]               
                self.data = np.concatenate((train_set.data, db_test_data))
                self.targets = np.concatenate((np.array(train_set.targets), db_test_targets))
            else:
                raise ValueError(f"Invalid mode '{mode}' for setting1")

        elif setting_name == "setting2":
            if mode == "train" or mode == "database":
                self.data = train_set.data
                self.targets = np.array(train_set.targets)
            elif mode == "test":
                self.data = test_set.data
                self.targets = np.array(test_set.targets)
            else:
                raise ValueError(f"Invalid mode '{mode}' for setting2")
        else:
            raise ValueError(f"Invalid setting_name '{setting_name}'")

        self.root = root
        self.transform = img_transform
        self.target_transform = target_transform
        self.default_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.return_index = return_index
        if self.return_index:
            self.index_table = torch.arange(0, len(self))
            
        self.return_paired_aug_img = return_paired_aug_img
        if self.return_paired_aug_img:
            assert self.transform is not None

    def _get_indices(self, targets, n_samples_per_class, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        indices = []
        for class_idx in range(10): # CIFAR-10 has 10 classes
            class_indices = np.where(targets == class_idx)[0]

            replace = len(class_indices) < n_samples_per_class
            sampled_indices = np.random.choice(class_indices, n_samples_per_class, replace=replace)
            indices.extend(sampled_indices)
        return np.array(indices)

    def __getitem__(self, index: int) -> Any:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img_i = self.default_transform(img)

        if self.transform:
            pass

        if self.target_transform:
            target = self.target_transform(target)

        return img_i, target
    
PATH_CONFIG = {
    "codeset": {
        "train": {
        16 : "./codebooks/set1_imagenet_16nt.csv", #imagenet best (paper default)
        #16 : "./codebooks/set1_imagenet_16nt_tau6.csv", #imagenet, stricter tau_in=6 variant

        #16 : "./codebooks/set1_cifar10_16nt.csv", #cifar_10 vit best
        #16 : "./codebooks/set1_cifar10_16nt_swin.csv", #cifar_10 best for the non-ViT backbones
        #16 : "./codebooks/set1_nuswide_16nt.csv", #nuswide best
        #32 : "./codebooks/set1_imagenet_32nt.csv", #best imagenet
        #32 : "./codebooks/set1_cifar10_32nt.csv", #cifar best
        32 : "./codebooks/set1_nuswide_32nt.csv", #nuswide best
        20 : "./dataset/codeset/final_bio_20nt_20k_hashcodes.csv"
        },
        "test": {
        16 : "./dataset/codeset/set2_16nt_20k.csv",
        32 : "./dataset/codeset/set2_32nt_20k.csv",
        20 : "./dataset/codeset/20bit_20k_hashcodes.csv"
        }
    },
    # "cifar-10": {
    #     "train" : "./dataset/cifar-10/cifar10_train.pkl",
    #     "query" : "./dataset/cifar-10/cifar10_test.pkl",
    #     "db" : "./dataset/cifar-10/cifar10_db.pkl",
    #     "emb_size" : 768
    # },
    # "imagenet": {
    #     "train" : "./dataset/imagenet/imagenet_train.pkl",
    #     "query" : "./dataset/imagenet/imagenet_test.pkl",
    #     "db" : "./dataset/imagenet/imagenet_db.pkl",
    #     "emb_size" : 768
    # },
    "cifar-10": {
        "train" : "./data/features/cifar10/train_img768.pkl",
        "query" : "./data/features/cifar10/test_img768.pkl",
        "db" : "./data/features/cifar10/newdb_img768.pkl",
        "emb_size" : 768
    },
    "imagenet": {
        "train" : "./data/features/imagenet/imagenet_train768.pkl",
        "query" : "./data/features/imagenet/imagenet_test768.pkl",
        "db" : "./data/features/imagenet/imagenet_db768.pkl",
        "emb_size" : 768
    },
    "cifar_swin": {
            "train": "./data/features/cifar10/swint_cifar10_train.pkl",
            "query": "./data/features/cifar10/swint_cifar10_test.pkl",
            "db": "./data/features/cifar10/swint_cifar10_db.pkl",
            "emb_size" : 1536
    },
    "cifar_eva": {
            "train": "./data/features/cifar10/eva_cifar10_train.pkl",
            "query": "./data/features/cifar10/eva_cifar10_test.pkl",
            "db": "./data/features/cifar10/eva_cifar10_db.pkl",
            "emb_size" : 1408
    },
    "imagenet_swin": {
            "train": "./data/features/imagenet/imagenet_swint_train.pkl",
            "query": "./data/features/imagenet/imagenet_swint_test.pkl" ,
            "db": "./data/features/imagenet/imagenet_swint_db.pkl",
            "emb_size" : 1536
    },
    "imagenet_eva": {
            "train": "./data/features/imagenet/imagenet_eva_train.pkl",
            "query": "./data/features/imagenet/imagenet_eva_test.pkl" ,
            "db": "./data/features/imagenet/imagenet_eva_db.pkl",
            "emb_size" : 1408
    },
    "nuswide": {
            "train": "./data/features/nuswide/nuswide_train.pkl",
            "query": "./data/features/nuswide/nuswide_test.pkl" ,
            "db": "./data/features/nuswide/nuswide_db.pkl",
            "emb_size" : 768
    },
    "nuswide_swin": {
            "train": "./data/features/nuswide/nuswide_swin_train_p.pkl",
            "query": "./data/features/nuswide/nuswide_swin_test_p.pkl" ,
            "db": "./data/features/nuswide/nuswide_swin_db_p.pkl",
            "emb_size" : 1536
    },
    "nuswide_eva": {
            "train": "./data/features/nuswide/nuswide_eva_train_c.pkl",
            "query": "./data/features/nuswide/nuswide_eva_test_c.pkl" ,
            "db": "./data/features/nuswide/nuswide_eva_db_c.pkl",
            "emb_size" : 1408
    }
}

