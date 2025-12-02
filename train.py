import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import shutil

MODEL_NAME = "distilbert-base-uncased"
SAVE_DIRECTORY = "./tiny_bert_sentiment"

train_texts = [
    "I love this product", "This is the best day", "Amazing service", 
    "I am very happy", "Fantastic result", "Good job", "Excellent work",
    "I hate this", "This is terrible", "Worst experience ever", 
    "I am sad", "Disgusting food", "Bad quality", "Very disappointing"
]
train_labels = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

class SimpleDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    if os.path.exists(SAVE_DIRECTORY):
        shutil.rmtree(SAVE_DIRECTORY)
    
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
    dataset = SimpleDataset(train_encodings, train_labels)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,           
        per_device_train_batch_size=4,
        logging_dir='./logs',
        save_strategy="no",
        use_cpu=not torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    model.save_pretrained(SAVE_DIRECTORY)
    tokenizer.save_pretrained(SAVE_DIRECTORY)
    

if __name__ == "__main__":
    import wandb
    wandb.init(mode="disabled")
    main()