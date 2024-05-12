import argparse
import ddputils

parser = argparse.ArgumentParser(description="Forecast TKGQA")
parser.add_argument(
    "--lm_model", default="distilbert", type=str, help="Pre-trained language model"
)
parser.add_argument(
    "--tkg_model_file",
    default="",
    type=str,
    help="TKG model checkpoint pre-trained on ICEWS21",
)
parser.add_argument(
    "--model", default="forecasttkgqa", type=str, help="Which QA method to run"
)
parser.add_argument(
    "--load_from", default="", type=str, help="Pre-trained QA model checkpoint"
)
parser.add_argument("--save_to", default="", type=str, help="Where to save checkpoint")
parser.add_argument(
    "--max_epochs", default=200, type=int, help="Number of maximum training epochs"
)
parser.add_argument(
    "--valid_freq",
    default=1,
    type=int,
    help="Number of epochs between each validation",
)
parser.add_argument(
    "--num_transformer_heads",
    default=8,
    type=int,
    help="Number of heads for transformers",
)
parser.add_argument(
    "--transformer_dropout", default=0.1, type=float, help="Dropout in transformers"
)
parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate")
parser.add_argument("--train", default=1, type=int, help="Whether to train")
parser.add_argument("--eval", default=1, type=int, help="Whether to evaluate")
parser.add_argument("--dataset_name", default="ICEWS21", type=str, help="Which dataset")
parser.add_argument(
    "--question_type",
    default="all",
    type=str,
    help="Which type of question are we training",
)
parser.add_argument(
    "--pct_train", default="1", type=str, help="Percentage of training data to use"
)
parser.add_argument(
    "--combine_all_ents",
    default="None",
    choices=["add", "mult", "None"],
    help="In score combination, whether to consider all entities or not",
)
# TODO: to check
parser.add_argument(
    "--num_transformer_layers",
    default=6,
    type=int,
    help="Number of layers for transformers",
)

args, unparsed = parser.parse_known_args()

if ddputils.should_execute():
    print(f"{args=}\n{unparsed=}")
