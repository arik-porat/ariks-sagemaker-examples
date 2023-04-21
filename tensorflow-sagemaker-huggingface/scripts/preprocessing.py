import os
import subprocess
import sys
import tensorflow as tf

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__=='__main__':
    
    install('transformers==4.17')
    install('datasets')
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding

    model_name = 'distilbert-base-uncased'

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load DatasetDict
    dataset = load_dataset("imdb")
    #train, test, unsupervised = dataset['train'], dataset['test'], dataset['unsupervised']

    # Preprocess train dataset
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # define tokenizer_columns
    # tokenizer_columns is the list of keys from the dataset that get passed to the TensorFlow model
    tokenizer_columns = ["attention_mask", "input_ids"]

    # convert to TF datasets
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    encoded_dataset["train"] = encoded_dataset["train"].rename_column("label", "labels")
    tf_train_dataset = encoded_dataset["train"].to_tf_dataset(
        columns=tokenizer_columns,
        label_cols=["labels"],
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator,
    )
    encoded_dataset["test"] = encoded_dataset["test"].rename_column("label", "labels")
    tf_validation_dataset = encoded_dataset["test"].to_tf_dataset(
        columns=tokenizer_columns,
        label_cols=["labels"],
        shuffle=False,
        batch_size=8,
        collate_fn=data_collator,
    )
    
    tf.data.experimental.save(tf_train_dataset, "/opt/ml/processing/train")
    tf.data.experimental.save(tf_validation_dataset, "/opt/ml/processing/test")