import argparse
import logging
import os
import sys

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers.schedules import PolynomialDecay


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--learning_rate", type=float, default=3e-5)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load dataset
    tf_train_dataset = tf.data.experimental.load(args.train_dir)
    tf_test_dataset = tf.data.experimental.load(args.test_dir)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=2, label2id={'neg': '0', 'pos': '1'}, id2label={'0': 'neg', '1': 'pos'}
    )
    
    # create Adam optimizer with learning rate scheduling
    num_train_steps = len(tf_train_dataset) * args.epochs
    lr_scheduler = PolynomialDecay(
        initial_learning_rate=args.learning_rate, end_learning_rate=0.0, decay_steps=num_train_steps
    )

    # fine optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # Training
    logger.info("*** Train ***")
    train_results = model.fit(
        tf_train_dataset,
        epochs=args.epochs,
        validation_data=tf_test_dataset,
    )

    output_eval_file = os.path.join(args.output_data_dir, "train_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Train results *****")
        logger.info(train_results)
        for key, value in train_results.history.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))

    # Save result
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
