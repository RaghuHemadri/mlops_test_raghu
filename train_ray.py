from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset as TorchDataset, DataLoader
import lightning as L
import torch
import litgpt
from litgpt import LLM
from litgpt.lora import GPT, merge_lora_weights
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy, DDPStrategy
from peft import get_peft_model, LoraConfig, TaskType
# from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning.loggers.mlflow import MLFlowLogger
import mlflow
import mlflow.pytorch
import os
import json
import pandas as pd

# 🔁 New Ray imports
import ray
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, FailureConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer, RayTrainReportCallback
from ray import train

floating_ip = os.getenv("FLOATING_IP", "")

# --------------------------
# Ray-compatible Train Function
# --------------------------
def train_func(config):
    print("Training with config:", config)

    import mlflow

    # Setup mlflow logging
    # mlflow.set_tracking_uri(f"http://{floating_ip}:8000/")
    # mlflow.set_experiment("medical-qa-tinyllama")
    mlflow_logger = MLFlowLogger(experiment_name="medical-qa-tinyllama", tracking_uri=f"http://{floating_ip}:8000")

    # # Start MLflow run
    # with mlflow.start_run():
    #     # Log config parameters
    #     for key, value in config.items():
    #         mlflow.log_param(key, value)

    #     # Log entire config as a JSON artifact
    #     with open("config.json", "w") as f:
    #         json.dump(config, f, indent=2)
    #     mlflow.log_artifact("config.json")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # Load paths from environment
    train_path = os.getenv("TRAINING_JSON_PATH", "/mnt/object/training.json")
    val_path = os.getenv("VALIDATION_JSON_PATH", "/mnt/object/validation.json")
    artifact_dir = os.getenv("ARTIFACT_PATH", "/mnt/object/artifacts")

    # Load dataset and sample 100
    df = pd.read_json(train_path,lines=True)
    df = df.sample(100, random_state=42).reset_index(drop=True)
    print(" First 5 samples from training data:")
    print(df.head())
    hf_dataset = Dataset.from_pandas(df)
    train_dataset = hf_dataset.select(range(80))
    val_dataset = hf_dataset.select(range(80, 100))


    # PyTorch Dataset
    class MedicalQADataset(TorchDataset):
        def __init__(self, dataset, tokenizer, max_length=512):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self): return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            prompt = f"Question: {item['question']}\nAnswer: {item['answer']}"
            encoding = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

    class MedicalQADataModule(L.LightningDataModule):
        def __init__(self, train_data, val_data, batch_size=8):
            super().__init__()
            self.train_data = train_data
            self.val_data = val_data
            self.batch_size = batch_size

        def train_dataloader(self):
            return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.val_data, batch_size=self.batch_size)

    train_data = MedicalQADataset(train_dataset, tokenizer)
    val_data = MedicalQADataset(val_dataset, tokenizer)
    data_module = MedicalQADataModule(train_data, val_data)

    class LitLLM(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = GPT.from_name(
                name=config["model_name"],
                lora_r=2, lora_alpha=4, lora_dropout=0.05,
                lora_query=True, lora_key=False, lora_value=True,
            )
            litgpt.lora.mark_only_lora_as_trainable(self.model)

        def training_step(self, batch, batch_idx):
            logits = self.model(batch["input_ids"])
            loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], batch["labels"][..., 1:])
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            logits = self.model(batch["input_ids"])
            loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], batch["labels"][..., 1:])
            self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / 10))
            return [optimizer], [scheduler]

    model = LitLLM()

    # ⚡ Ray-compatible trainer
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        logger=mlflow_logger,
        log_every_n_steps=5,
        callbacks=[RayTrainReportCallback()],
    )

    trainer = prepare_trainer(trainer)

    # 👇 Optional fault-tolerant resume
    ckpt = train.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as ckpt_dir:
            trainer.fit(model, data_module, ckpt_path=os.path.join(ckpt_dir, "checkpoint.ckpt"))
    else:
        trainer.fit(model, data_module)

    #  Save final model 
    merge_lora_weights(model.model)
    torch.save(model.model.state_dict(), "model.pth")
    print(f"Model saved")

# --------------------------
# Launch with Ray
# --------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Connecting to Ray...")
    ray.init(address="auto")
    logging.info("Connected to Ray.")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            "lr": 2e-4,
            "epochs": 1,
        },
        run_config=RunConfig(
            name="ray-medical-qa",
            storage_path="s3://mlflow-artifacts",
            checkpoint_config=CheckpointConfig(num_to_keep=1),
            failure_config=FailureConfig(max_failures=1)
        ),
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=True,
            resources_per_worker={"CPU": 8, "GPU": 1}
        )
    )

    logging.info("Starting Ray training job...")
    results = trainer.fit()
