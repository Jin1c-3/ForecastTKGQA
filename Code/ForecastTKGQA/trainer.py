import os
import argparse
import torch
from qa_models import ForecastTKGQA
from qa_baselines import QABert, QABERTTComplEx, QABertTango, CronKGQATComplEx, EmbedKGQA, TempoQRTComplEx
from qa_datasets import QADatasetBert, QADatasetBaseline, QADatasetForecast, QADatasetTempoQR
from eval import eval_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import numpy as np
import random
import pickle
import torch.distributed as dist
import os

seed = 20220801
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
# if torch.cuda.is_available():  # GPU operation have separate seed
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# All types of models
# RoBERTa is specified later in LM initialization
models = ['bert', 'bert_int', 'bert_ext', 'embedkgqa', 'cronkgqa', 'tempoqr', 'forecasttkgqa']

def execute_if_rank_is_zero(func):
    def wrapper(*args, **kwargs):
        if dist.get_rank() == 0:
            return func(*args, **kwargs)
    return wrapper

@execute_if_rank_is_zero
def print_if_rank_zero(*args, **kwargs):
    print(*args, **kwargs)

@execute_if_rank_is_zero
def append_log_to_file(eval_log, epoch, filename):
    # Help function for logging
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print('Creating new log file')
    f = open(filename, 'a+')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    f.write('Log time: %s\n' % dt_string)
    f.write('Epoch %d\n' % epoch)
    for line in eval_log:
        f.write('%s\n' % line)
    f.write('\n')
    f.close()

@execute_if_rank_is_zero
def save_model(qa_model, filename):
    # Help function for saving trained models
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    print('Saving model to', filename)
    torch.save(qa_model.module.state_dict(), filename)
    print('Saved model to ', filename)
    return

def main():
    parser = argparse.ArgumentParser(description="Forecast TKGQA")
    parser.add_argument(
        '--lm_model', default='distilbert', type=str,
        help="Pre-trained language model"
    )
    parser.add_argument(
        '--tkg_model_file', default='', type=str,
        help="TKG model checkpoint pre-trained on ICEWS21"
    )
    parser.add_argument(
        '--model', default='forecasttkgqa', type=str,
        help="Which QA method to run"
    )
    parser.add_argument(
        '--load_from', default='', type=str,
        help="Pre-trained QA model checkpoint"
    )
    parser.add_argument(
        '--save_to', default='', type=str,
        help="Where to save checkpoint"
    )
    # parser.add_argument(
    #     '--use_distinguisher', default=0, type=int,
    #     help="Whether to use distinguisher, default: not use"
    # )
    parser.add_argument(
        '--max_epochs', default=200, type=int,
        help="Number of maximum training epochs"
    )
    parser.add_argument(
        '--valid_freq', default=1, type=int,
        help="Number of epochs between each validation"
    )
    parser.add_argument(
        '--num_transformer_heads', default=8, type=int,
        help="Number of heads for transformers"
    )
    parser.add_argument(
        '--transformer_dropout', default=0.1, type=float,
        help="Dropout in transformers"
    )
    parser.add_argument(
        '--batch_size', default=256, type=int,
        help="Batch size"
    )
    parser.add_argument(
        '--lr', default=2e-5, type=float,
        help="Learning rate"
    )
    parser.add_argument(
        '--train', default=1, type=int,
        help="Whether to train"
    )
    parser.add_argument(
        '--eval', default=1, type=int,
        help="Whether to evaluate"
    )
    parser.add_argument(
        '--dataset_name', default='ICEWS21', type=str,
        help="Which dataset"
    )
    parser.add_argument(
        '--question_type', default='all', type=str,
        help="Which type of question are we training"
    )
    parser.add_argument(
        '--pct_train', default='1', type=str,
        help="Percentage of training data to use"
    )
    parser.add_argument(
        '--combine_all_ents', default="None", choices=["add", "mult", "None"],
        help="In score combination, whether to consider all entities or not"
    )
    # TODO: to check
    parser.add_argument(
        '--num_transformer_layers', default=6, type=int,
        help="Number of layers for transformers"
    )

    args = parser.parse_args()
    args.batch_size = int(args.batch_size / torch.cuda.device_count())
    dist.init_process_group(backend='gloo')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Path to save logs
    log_filename = 'results/{dataset_name}/{question_type}_{model_file}.log'.format(
        dataset_name=args.dataset_name,
        question_type=args.question_type,
        model_file=args.save_to
    )
    if args.save_to == '':
        args.save_to = 'temp'

    # Checkpoint name
    checkpoint_file_name = 'models/{dataset_name}/qa_models/{question_type}_{model_file}.ckpt'.format(
        dataset_name=args.dataset_name,
        question_type=args.question_type,
        model_file=args.save_to
    )

    # Model Instantiation
    if args.model == 'bert':
        qa_model = QABert(args)
    elif args.model == 'bert_int':
        qa_model = QABERTTComplEx(args)
    elif args.model == 'bert_ext':
        qa_model = QABertTango(args)
    elif args.model == 'embedkgqa':
        qa_model = EmbedKGQA(args)
    elif args.model == 'cronkgqa':
        qa_model = CronKGQATComplEx(args)
    elif args.model == 'tempoqr':
        qa_model = TempoQRTComplEx(args)
    elif args.model == 'forecasttkgqa':
        qa_model = ForecastTKGQA(4, args)
    else:
        raise ValueError(f'Model {args.model} not implemented!')
    print_if_rank_zero('Model is', args.model)

    # Prepare data
    # train_dataloader, valid_dataloader, test_dataloader, valid_dataset, test_dataset = prepare_data(args)
    # Prepare datasets
    annotate_time = True
    if args.model == models[0]:  # LM baselines, without TKG representations
        train_dataset = QADatasetBert(split='train', dataset_name=args.dataset_name, question_type=args.question_type,
                                      pct=args.pct_train)
        valid_dataset = QADatasetBert(split="valid", dataset_name=args.dataset_name, question_type=args.question_type,
                                      pct=args.pct_train)
        test_dataset = QADatasetBert(split="test", dataset_name=args.dataset_name, question_type=args.question_type,
                                     pct=args.pct_train)
    elif args.model in models[1:5]:  # LM variants, EmbedKGQA, and CronKGQA
        train_dataset = QADatasetBaseline(split='train', dataset_name=args.dataset_name,
                                          question_type=args.question_type, pct=args.pct_train)
        valid_dataset = QADatasetBaseline(split="valid", dataset_name=args.dataset_name,
                                          question_type=args.question_type, pct=args.pct_train)
        test_dataset = QADatasetBaseline(split="test", dataset_name=args.dataset_name, question_type=args.question_type,
                                         pct=args.pct_train)
    elif args.model in models[5]:  # TempoQR
        train_dataset = QADatasetTempoQR(split='train', dataset_name=args.dataset_name,
                                         question_type=args.question_type, pct=args.pct_train,
                                         annotate_time=annotate_time)
        valid_dataset = QADatasetTempoQR(split="valid", dataset_name=args.dataset_name,
                                         question_type=args.question_type,
                                         pct=args.pct_train, annotate_time=annotate_time)
        test_dataset = QADatasetTempoQR(split="test", dataset_name=args.dataset_name, question_type=args.question_type,
                                        pct=args.pct_train, annotate_time=annotate_time)
    elif args.model == models[-1]:  # ForecastTKGQA
        train_dataset = QADatasetForecast(split='train', dataset_name=args.dataset_name,
                                          question_type=args.question_type, pct=args.pct_train)
        valid_dataset = QADatasetForecast(split="valid", dataset_name=args.dataset_name,
                                          question_type=args.question_type, pct=1)
        test_dataset = QADatasetForecast(split="test", dataset_name=args.dataset_name,
                                         question_type=args.question_type, pct=1)
    else:
        raise ValueError('Model %s not implemented!' % args.model)
    if not os.path.exists("data/ICEWS21/filter_dict.pkl"):
        if args.question_type == 'entity_prediction':
            train_dataset.create_ep_filter()
            valid_dataset.create_ep_filter()
            test_dataset.create_ep_filter()
            filter_dict = {}
            filter_dict.update(train_dataset.filter_dict)
            filter_dict.update(valid_dataset.filter_dict)
            filter_dict.update(test_dataset.filter_dict)
            with open('data/ICEWS21/filter_dict.pkl', 'wb') as f:
                pickle.dump(filter_dict, f)
    # Prepare dataloaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                  shuffle=False, num_workers=5,
                                  collate_fn=train_dataset.collate_fn)
    print_if_rank_zero(f'info for training set: {train_dataset.get_dataset_ques_info()}')
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, shuffle=False,
                                  num_workers=5,
                                  collate_fn=valid_dataset.collate_fn)
    print_if_rank_zero(f'info for valid set: {valid_dataset.get_dataset_ques_info()}')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=5,
                                  collate_fn=test_dataset.collate_fn)
    print_if_rank_zero(f'info for test set: {test_dataset.get_dataset_ques_info()}')

    # Load from checkpoint
    if args.load_from != '':
        model_filename = 'models/{dataset_name}/qa_models/{model_file}.ckpt'.format(
            dataset_name=args.dataset_name,
            model_file=args.load_from
        )
        print('Loading model from', model_filename)
        qa_model.load_state_dict(torch.load(model_filename, map_location=torch.device("cpu")))
        print_if_rank_zero('Loaded qa model from ', model_filename)
    else:
        # Make a new log file with args if not loading from any previous file
        log_args = [str(key) + '\t' + str(value) for key, value in vars(args).items()]
        append_log_to_file(log_args, 0, log_filename)
        print_if_rank_zero('Not loading from checkpoint. Starting fresh!')

    # Send model to CUDA
    qa_model.cuda(local_rank)
    qa_model = torch.nn.parallel.DistributedDataParallel(qa_model, device_ids=[local_rank], find_unused_parameters=True)

    # Training
    if args.train:
        print_if_rank_zero('Starting training')
        criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda(local_rank)

        # Optimizer Initialization
        optimizer = torch.optim.Adam(qa_model.parameters(), lr=args.lr)
        optimizer.zero_grad()
        max_mrr, max_clf_score, max_choice_score = 0, 0, 0

        for epoch in range(args.max_epochs):
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            qa_model.train()
            epoch_loss = 0
            loader = tqdm(train_dataloader, total=len(train_dataloader), unit="batches")
            running_loss = 0
            for i_batch, (
                    b_input_id, b_attention_mask, heads, tails, times, types, answers_single,
                    batch_sentences) in enumerate(loader):
                # Send data to CUDA
                question_tokenized = b_input_id.cuda(local_rank, non_blocking=True)
                question_attention_mask = b_attention_mask.cuda(local_rank, non_blocking=True)
                heads = heads.cuda(local_rank, non_blocking=True)
                tails = tails.cuda(local_rank, non_blocking=True)
                times = times.cuda(local_rank, non_blocking=True)
                types = types.cuda(local_rank, non_blocking=True)
                answers = answers_single.cuda(local_rank, non_blocking=True)
                qa_model.zero_grad()
                # Compute batch loss
                _, scores_ep, scores_yn, scores_mc = qa_model.forward(question_tokenized, question_attention_mask,
                                                                      heads, tails, times,
                                                                      types, answers)
                # 创建一个列表来存储所有的得分
                scores = [scores_ep, scores_yn, scores_mc]

                # 计算None的数量
                none_count = scores.count(None)

                # 如果None的数量小于等于1，或者全是None，引发一个异常
                if none_count <= 1 or none_count == 3:
                    raise ValueError(
                        f"得分异常，{scores_ep=}, {scores_yn=}, {scores_mc=}")

                # 如果只有一个值不为None，将其存储在score变量中
                elif none_count == 2:
                    score = next(s for s in scores if s is not None)
                    mask = types < 2 if score is scores_ep else types == 2 if score is scores_yn else types == 3
                    loss = criterion(score, answers[mask])
                else:
                    raise ValueError("所有得分都不为None，无法确定唯一的得分")
                # Loss backward
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                running_loss += loss.item()
                loader.set_postfix(Loss=running_loss / ((i_batch + 1) * args.batch_size), Epoch=epoch)
                loader.set_description('{}/{}'.format(epoch, args.max_epochs))
                loader.update()

            print_if_rank_zero('Epoch loss = ', epoch_loss)

            # Evaluation
            if (epoch + 1) % args.valid_freq == 0:
                print_if_rank_zero('Starting validation')
                mrr_score, clf_score, choice_score, eval_log = \
                    eval_model(qa_model, valid_dataloader, valid_dataset, 'valid', args.batch_size, local_rank)
                save = False

                # If the validation results increase, save checkpoint
                if mrr_score > max_mrr:
                    save = True
                    eval_log.append(f'Valid mrr score increased from {max_mrr} to {mrr_score}')
                    print_if_rank_zero(f'Valid mrr score increased from {max_mrr} to {mrr_score}')
                    max_mrr = mrr_score
                if clf_score > max_clf_score:
                    save = True
                    eval_log.append(f'Valid yes_no score increased from {max_clf_score} to {clf_score}')
                    print_if_rank_zero(f'Valid yes_no score increased from {max_clf_score} to {clf_score}')
                    max_clf_score = clf_score
                if choice_score > max_choice_score:
                    save = True
                    eval_log.append(f'Valid choice score increased from {max_choice_score} to {choice_score}')
                    print_if_rank_zero(f'Valid choice score increased from {max_choice_score} to {choice_score}')
                    max_choice_score = choice_score
                if save:
                    save_model(qa_model, checkpoint_file_name)
                # Log the validation results
                append_log_to_file(eval_log, epoch, log_filename)

                # Also do test after validation
                print_if_rank_zero('Start testing')
                _, _, choice_score_test, log = eval_model(qa_model, test_dataloader, test_dataset, 'test', args.batch_size, local_rank)
                append_log_to_file(log, epoch, log_filename)

    # Set args.train to False, args.eval to True, for evaluation only
    if args.eval:
        print_if_rank_zero("Starting evaluation")
        _, _, _, log = eval_model(qa_model, test_dataloader, test_dataset, 'test', args.batch_size, local_rank)
        append_log_to_file(log, 0, log_filename)


if __name__ == "__main__":
    main()
