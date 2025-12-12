"""
AI for Conservation Project: DINO V3
by Tripp Lyons
"""

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from transformers import AutoModel, pipeline, AutoImageProcessor
import numpy as np
import io
import os
from PIL import Image
import torch
import torch.nn as nn


model_id = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

num_features = 1280


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_feature_extractor(model_id, device):
    torch.set_default_device(device)
    image_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id).to(device)
    return pipeline(
        model=model,
        task="image-feature-extraction",
        image_processor=image_processor,
        use_fast=True,
        framework="pt",
        device_map="auto",
    )


def get_features(feature_extractor, images, device):
    features = feature_extractor(
        images,
        batch_size=len(images),
        return_tensors="pt",
        device=device,
    )
    features = (
        torch.concat(features, dim=0).mean(-2).to(device)
    )  # average pooling over tokens
    return features.flatten(start_dim=1)


def load_dataset_df():
    import os

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    from datasets import load_dataset

    dataset_id = "imageomics/invasive_plants_hawaii"

    # splits are "dorsal", "ventral", "both"
    both_dataset = load_dataset(dataset_id, split="both")
    both_df = both_dataset.to_pandas()
    del both_dataset
    return both_df


class PlantsDataloader:
    def __init__(self, df, feature_extractor, device, batch_size=64, cache_path=None):
        self.device = device
        self.df = df
        self.damage_types = [
            "healthy",
            "rust",
            "leaf_miner",
            "other_insect",
            "mechanical_damage",
        ]
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.features = None

        if cache_path:
            if os.path.exists(cache_path):
                print(f"Loading features from {cache_path}")
                self.features = torch.load(cache_path, map_location="cpu")
            else:
                print(f"Precomputing features and saving to {cache_path}")
                self.features = self.precompute_features()
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(self.features, cache_path)

    def precompute_features(self):
        all_features = []
        batch_images = []
        for idx in tqdm(range(len(self.df)), desc="Precomputing features"):
            row = self.df.iloc[idx]
            image = self.binary_string_to_image(row["image"]["bytes"])
            batch_images.append(image)

            if len(batch_images) == self.batch_size:
                with torch.no_grad():
                    feats = get_features(
                        self.feature_extractor, batch_images, self.device
                    )
                all_features.append(feats.cpu())
                batch_images = []

        if len(batch_images) > 0:
            with torch.no_grad():
                feats = get_features(self.feature_extractor, batch_images, self.device)
            all_features.append(feats.cpu())

        if not all_features:
            return torch.empty(0, num_features)

        return torch.cat(all_features, dim=0)

    def binary_string_to_image(self, binary_string) -> torch.Tensor:
        file = io.BytesIO(binary_string)
        return Image.open(file)

    def epoch_iterator(self, shuffle=True):
        if shuffle:
            order = np.random.permutation(range(len(self.df)))
        else:
            order = range(len(self.df))

        if self.features is not None:
            num_samples = len(self.df)
            for i in range(0, num_samples, self.batch_size):
                batch_indices = order[i : i + self.batch_size]
                batch_feats = self.features[batch_indices].to(self.device)

                batch_labels = []
                for idx in batch_indices:
                    row = self.df.iloc[idx]
                    batch_labels.append(
                        torch.tensor(
                            row[self.damage_types]
                            .apply(lambda x: x == "Yes")
                            .values.astype(np.float32),
                            device=self.device,
                        )
                    )
                yield batch_feats, torch.stack(batch_labels)
        else:
            batch_images = []
            batch_labels = []
            for idx in order:
                row = self.df.iloc[idx]
                image = self.binary_string_to_image(row["image"]["bytes"])
                batch_images.append(image)
                batch_labels.append(
                    torch.tensor(
                        row[self.damage_types]
                        .apply(lambda x: x == "Yes")
                        .values.astype(np.float32),
                        device=self.device,
                    )
                )
                if len(batch_images) == self.batch_size:
                    yield (
                        get_features(self.feature_extractor, batch_images, self.device),
                        torch.stack(batch_labels),
                    )
                    batch_images = []
                    batch_labels = []
            if len(batch_images) > 0:
                yield (
                    get_features(self.feature_extractor, batch_images, self.device),
                    torch.stack(batch_labels),
                )


def main():
    seed = 0
    batch_size = 64
    learning_rate = 1e-3
    epochs = 30
    eval_every = 1
    num_classes = 5
    device = get_device()

    both_df = load_dataset_df()
    # 0.8 for training
    train_df, non_train_df = train_test_split(both_df, test_size=0.2, random_state=seed)
    test_start = len(non_train_df) // 2
    # 0.1 for validation
    val_df = non_train_df.head(test_start)
    # 0.1 for testing
    test_df = non_train_df.tail(len(non_train_df) - test_start)

    feature_extractor = get_feature_extractor(model_id, device)
    train_dataloader = PlantsDataloader(
        train_df,
        feature_extractor,
        device,
        batch_size=batch_size,
        cache_path="dino_features/train_features.pt",
    )
    val_dataloader = PlantsDataloader(
        val_df,
        feature_extractor,
        device,
        batch_size=batch_size,
        cache_path="dino_features/val_features.pt",
    )
    test_dataloader = PlantsDataloader(
        test_df,
        feature_extractor,
        device,
        batch_size=batch_size,
        cache_path="dino_features/test_features.pt",
    )

    # model = nn.Linear(num_features, num_classes).to(device)
    def make_layer():
        return nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    model = nn.Sequential(
        make_layer(),
        make_layer(),
        make_layer(),
        nn.Linear(num_features, num_classes),
    ).to(device)

    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        losses = []
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        for features, labels in tqdm(
            train_dataloader.epoch_iterator(shuffle=True), desc="Training"
        ):
            optimizer.zero_grad()
            logits = model(features.to(device))
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Average Loss: {np.mean(losses):.4f}")
        if (epoch + 1) % eval_every == 0:
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for features, labels in tqdm(
                    val_dataloader.epoch_iterator(shuffle=False), desc="Evaluating"
                ):
                    logits = model(features)
                    loss = loss_fn(logits, labels.to(device))
                    eval_losses.append(loss.item())
                print(f"Average Eval Loss: {np.mean(eval_losses):.4f}")

    model.eval()

    all_predictions = []
    all_labels = []

    final_dataloader = test_dataloader

    with torch.no_grad():
        for features, labels in tqdm(
            final_dataloader.epoch_iterator(shuffle=False), desc="Evaluating"
        ):
            logits = model(features)
            all_predictions.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_labels = all_labels.int()

        max_f1s = []
        roc_aucs = []
        for i in range(num_classes):
            thresholds = np.linspace(0, 1, 100)[1:-1]
            precision = []
            recall = []
            f1 = []
            for threshold in thresholds:
                preds = (all_predictions[:, i] > threshold).int()
                predicted_positive = preds == 1
                label_positive = all_labels[:, i] == 1
                if predicted_positive.sum() == 0:
                    precision.append(1)
                else:
                    precision.append(
                        (
                            (preds == all_labels[:, i])[predicted_positive].int().sum(0)
                            / predicted_positive.sum()
                        )
                        .cpu()
                        .item()
                    )
                if label_positive.sum() == 0:
                    recall.append(1)
                else:
                    recall.append(
                        (
                            (preds == all_labels[:, i])[label_positive].int().sum(0)
                            / label_positive.sum()
                        )
                        .cpu()
                        .item()
                    )
                p = precision[-1]
                r = recall[-1]
                f1.append(2 * p * r / (p + r) if p + r > 0 else 0)
            max_f1s.append(max(f1))
            fpr_curve, tpr_curve, _ = roc_curve(all_labels[:, i], all_predictions[:, i])
            roc_aucs.append(auc(fpr_curve, tpr_curve))
        print(f"Damage types: {train_dataloader.damage_types}")
        print(f"Max F1s: {max_f1s}")
        print(f"ROC AUCs: {roc_aucs}")


if __name__ == "__main__":
    main()
