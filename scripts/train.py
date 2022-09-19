# This is the script that will be used in the training container
import argparse
import logging
import os
import sys
from unicodedata import name

from datasets import load_from_disk
from transformers import AutoTokenizer
from sentence_transformers import InputExample, SentenceTransformer, models, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def load_and_prepare_dataset(
    data_dir, text_column, target_column, num_rows=0
):
    #dataset = load_from_disk(os.path.join(data_dir, split))
    dataset = load_from_disk(data_dir)
    examples = []
    if num_rows==0:
        n_examples = dataset.num_rows    
    else:
        n_examples = num_rows

    print('Rows to collect:', n_examples)

    for i in range(n_examples):
        examples.append(InputExample(texts=[dataset[i][text_column], dataset[i][target_column]]))

    return examples



def train(args):

    model_name = args.model_name
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Loading datasets...\n")
    train_examples = load_and_prepare_dataset(args.train_data_dir,args.text_column,
        args.target_column, args.num_examples)
    validation_examples = load_and_prepare_dataset(args.val_data_dir,args.text_column,
        args.target_column, args.num_examples)
    test_examples = load_and_prepare_dataset(args.test_data_dir,args.text_column,
        args.target_column, args.num_examples)

    logger.info("Defining the model\n")
    ## Step 1: use an existing language model
    word_embedding_model = models.Transformer(model_name)

    ## Step 2: use a pool function over the token embeddings
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

    ## Join steps 1 and 2 using the modules argument
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    logger.info("Setting the loss function\n")
    # Define the loss function for our dataset
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    logger.info("Creating the DataLoader for training\n")
    # Create a DataLoader to be trained
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.train_batch_size)

    logger.info("Creating the Evaluator")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(validation_examples, name='sts-dev')

    # Set logging steps when strategy= steps
    logger.info("Defining some training parameters\n")
    warmup_steps = int(len(train_dataloader) *args.epochs * 0.1) #10% of train data

    logger.info("Starting Training")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator= evaluator,
          epochs=args.epochs,
          warmup_steps=warmup_steps,
          output_path=args.model_dir) 

    model.save(args.model_dir, args.trained_model_name, train_datasets= ["LeoCordoba/CC-NEWS-ES-titles"])
    
    logger.info("Model trained successfully")
    logger.info("Evaluate the model on the test dataset")
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name='sts-test')
    test_evaluator(model, output_path=args.model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="bertin-project/bertin-roberta-base-spanish")
    parser.add_argument(
        "--train-data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument(
        "--val-data-dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"]
    )
    parser.add_argument(
        "--test-data-dir", type=str, default=os.environ["SM_CHANNEL_TEST"]
    )

    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--target-column", type=str, default="output_text")
    parser.add_argument("--max-source", type=int, default=256)
    parser.add_argument("--max-target", type=int, default=32)
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    #parser.add_argument("--warmup-steps", type=float, default=50)
    #parser.add_argument("--lr", type=float, default=2e-5)
    #parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--trained-model-name", type=str)    
    #parser.add_argument("--log-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    #parser.add_argument("--logging-strategy", type=str, default="epoch")
    train(parser.parse_args())