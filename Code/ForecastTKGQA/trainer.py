import os
from args_config import args
import torch
from qa_models import ForecastTKGQA
from qa_baselines import (
    QABert,
    QABERTTComplEx,
    QABertTango,
    CronKGQATComplEx,
    EmbedKGQA,
    TempoQRTComplEx,
)
from qa_datasets import (
    QADatasetBert,
    QADatasetBaseline,
    QADatasetForecast,
    QADatasetTempoQR,
)
from eval import eval_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import numpy as np
import random
import pickle
from torch import distributed as dist
import os
import ddputils
from torch.utils.data.distributed import DistributedSampler

seed = 20220801
os.environ["PYTHONHASHSEED"] = str(seed)
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


@ddputils.execute_once
def append_log_to_file(eval_log, epoch, filename):
    # Help function for logging
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Creating new log file")
    with open(filename, "a+") as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"Log time: {dt_string}\n")
        f.write(f"Epoch {epoch}\n")
        for line in eval_log:
            f.write(f"{line}\n")
        f.write("\n")


@ddputils.execute_once
def save_model(qa_model, filename):
    # Help function for saving trained models
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    print("Saving model to", filename)
    torch.save(qa_model.module.state_dict(), filename)
    print("Saved model to ", filename)
    return


def prepare_data():
    assert (
        args.batch_size % ddputils.get_world_size() == 0
    ), "Batch size must be divisible by the number of GPUs"
    distributed_batch_size = args.batch_size // ddputils.get_world_size()
    # Prepare datasets
    annotate_time = True
    if args.model == models[0]:  # LM baselines, without TKG representations
        train_dataset = QADatasetBert(
            split="train",
            dataset_name=args.dataset_name,
            question_type=args.question_type,
            pct=args.pct_train,
        )
        valid_dataset = QADatasetBert(
            split="valid",
            dataset_name=args.dataset_name,
            question_type=args.question_type,
            pct=args.pct_train,
        )
        test_dataset = QADatasetBert(
            split="test",
            dataset_name=args.dataset_name,
            question_type=args.question_type,
            pct=args.pct_train,
        )
    elif args.model in models[1:5]:  # LM variants, EmbedKGQA, and CronKGQA
        train_dataset = QADatasetBaseline(
            split="train",
            dataset_name=args.dataset_name,
            question_type=args.question_type,
            pct=args.pct_train,
        )
        valid_dataset = QADatasetBaseline(
            split="valid",
            dataset_name=args.dataset_name,
            question_type=args.question_type,
            pct=args.pct_train,
        )
        test_dataset = QADatasetBaseline(
            split="test",
            dataset_name=args.dataset_name,
            question_type=args.question_type,
            pct=args.pct_train,
        )
    elif args.model in models[5]:  # TempoQR
        train_dataset = QADatasetTempoQR(
            split="train",
            dataset_name=args.dataset_name,
            question_type=args.question_type,
            pct=args.pct_train,
            annotate_time=annotate_time,
        )
        valid_dataset = QADatasetTempoQR(
            split="valid",
            dataset_name=args.dataset_name,
            question_type=args.question_type,
            pct=args.pct_train,
            annotate_time=annotate_time,
        )
        test_dataset = QADatasetTempoQR(
            split="test",
            dataset_name=args.dataset_name,
            question_type=args.question_type,
            pct=args.pct_train,
            annotate_time=annotate_time,
        )
    elif args.model == models[-1]:  # ForecastTKGQA
        train_dataset = QADatasetForecast(
            split="train",
            dataset_name=args.dataset_name,
            question_type=args.question_type,
            pct=args.pct_train,
        )
        valid_dataset = QADatasetForecast(
            split="valid",
            dataset_name=args.dataset_name,
            question_type=args.question_type,
            pct=1,
        )
        test_dataset = QADatasetForecast(
            split="test",
            dataset_name=args.dataset_name,
            question_type=args.question_type,
            pct=1,
        )
    else:
        raise ValueError(f"Model {args.model} not implemented!")
    if not os.path.exists("data/ICEWS21/filter_dict.pkl"):
        if args.question_type == "entity_prediction":
            train_dataset.create_ep_filter()
            valid_dataset.create_ep_filter()
            test_dataset.create_ep_filter()
            filter_dict = {}
            filter_dict.update(train_dataset.filter_dict)
            filter_dict.update(valid_dataset.filter_dict)
            filter_dict.update(test_dataset.filter_dict)
            with open("data/ICEWS21/filter_dict.pkl", "wb") as f:
                pickle.dump(filter_dict, f)
    # Prepare dataloaders
    train_sampler = DistributedSampler(
        train_dataset, ddputils.get_world_size(), ddputils.get_rank()
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=distributed_batch_size,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
    )
    if ddputils.should_execute():
        print(f"info for training set: {train_dataset.get_dataset_ques_info()}")
    valid_sampler = DistributedSampler(
        valid_dataset, dist.get_world_size(), ddputils.get_rank()
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=distributed_batch_size,
        sampler=valid_sampler,
        num_workers=0,
        collate_fn=valid_dataset.collate_fn,
    )
    if ddputils.should_execute():
        print(f"info for valid set: {valid_dataset.get_dataset_ques_info()}")
    test_sampler = DistributedSampler(
        test_dataset, dist.get_world_size(), ddputils.get_rank()
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=distributed_batch_size,
        sampler=test_sampler,
        num_workers=0,
        collate_fn=test_dataset.collate_fn,
    )
    if ddputils.should_execute():
        print(f"info for test set: {test_dataset.get_dataset_ques_info()}")
    return (
        train_dataset,
        train_dataloader,
        valid_dataset,
        valid_dataloader,
        test_dataset,
        test_dataloader,
    )


if __name__ == "__main__":
    ddputils.init_distributed_env(args)
    torch.cuda.set_device(ddputils.get_rank())

    # Path to save logs
    log_filename = f"results/{args.dataset_name}/{args.question_type}_{args.model}.log"

    if args.save_to == "":
        args.save_to = "temp"

    # Checkpoint name
    checkpoint_file_name = (
        f"models/{args.dataset_name}/qa_models/{args.question_type}_{args.model}.ckpt"
    )

    # Model Instantiation
    model_dict = {
        "bert": QABert,
        "bert_int": QABERTTComplEx,
        "bert_ext": QABertTango,
        "embedkgqa": EmbedKGQA,
        "cronkgqa": CronKGQATComplEx,
        "tempoqr": TempoQRTComplEx,
        "forecasttkgqa": ForecastTKGQA,
    }

    # All types of models
    # RoBERTa is specified later in LM initialization
    models = list(model_dict.keys())

    if args.model in model_dict:
        qa_model = model_dict[args.model](args)
    else:
        raise ValueError(f"Model {args.model} not implemented!")
    if ddputils.should_execute():
        print("Model is ", args.model)

    (
        train_dataset,
        train_dataloader,
        valid_dataset,
        valid_dataloader,
        test_dataset,
        test_dataloader,
    ) = prepare_data()

    # Load from checkpoint
    if args.load_from != "":
        model_filename = f"models/{args.dataset_name}/qa_models/{args.load_from}.ckpt"
        print("Loading model from", model_filename)
        qa_model.load_state_dict(
            torch.load(model_filename, map_location=torch.device("cpu"))
        )
        if ddputils.should_execute():
            print("Loaded qa model from ", model_filename)
    else:
        # Make a new log file with args if not loading from any previous file
        log_args = [str(key) + "\t" + str(value) for key, value in vars(args).items()]
        append_log_to_file(log_args, 0, log_filename)
        if ddputils.should_execute():
            print("Not loading from checkpoint. Starting fresh!")

    # Send model to CUDA
    qa_model.cuda(ddputils.get_rank())
    qa_model = torch.nn.parallel.DistributedDataParallel(
        qa_model,
        device_ids=[ddputils.get_rank()],
        find_unused_parameters=True,
        output_device=ddputils.get_rank(),
    )

    # Training
    if args.train:
        if ddputils.should_execute():
            print("Starting training")
        criterion = torch.nn.CrossEntropyLoss(reduction="mean").cuda(
            ddputils.get_rank()
        )

        # Optimizer Initialization
        optimizer = torch.optim.Adam(qa_model.parameters(), lr=args.lr)
        max_mrr, max_clf_score, max_choice_score = 0, 0, 0

        for epoch in range(args.max_epochs):
            qa_model.train()
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)
            test_dataloader.sampler.set_epoch(epoch)
            epoch_loss = 0
            loader = (
                tqdm(train_dataloader, unit="batches", total=len(train_dataloader))
                if ddputils.should_execute()
                else train_dataloader
            )
            for i_batch, (
                b_input_id,
                b_attention_mask,
                heads,
                tails,
                times,
                types,
                answers_single,
                batch_sentences,
                relations,
            ) in enumerate(loader):
                # Send data to CUDA
                question_tokenized = b_input_id.cuda(ddputils.get_rank())
                question_attention_mask = b_attention_mask.cuda(ddputils.get_rank())
                heads = heads.cuda(ddputils.get_rank())
                tails = tails.cuda(ddputils.get_rank())
                times = times.cuda(ddputils.get_rank())
                types = types.cuda(ddputils.get_rank())
                answers = answers_single.cuda(ddputils.get_rank())
                relations = relations.cuda(ddputils.get_rank())
                # Compute batch loss
                _, scores_ep, scores_yn, scores_mc = qa_model(
                    question_tokenized,
                    question_attention_mask,
                    heads,
                    tails,
                    times,
                    types,
                    answers,
                    relations,
                )
                # 创建一个列表来存储所有的得分
                scores = [scores_ep, scores_yn, scores_mc]

                # 计算None的数量
                none_count = scores.count(None)

                # 如果None的数量小于等于1，或者全是None，引发一个异常
                if none_count <= 1 or none_count == 3:
                    raise ValueError(
                        f"得分异常，{scores_ep=}, {scores_yn=}, {scores_mc=}"
                    )

                # 如果只有一个值不为None，将其存储在score变量中
                elif none_count == 2:
                    score = next(s for s in scores if s is not None)
                    mask = (
                        types < 2
                        if score is scores_ep
                        else types == 2 if score is scores_yn else types == 3
                    )
                    loss = criterion(score, answers[mask])
                else:
                    raise ValueError("所有得分都不为None，无法确定唯一的得分")
                # Loss backward
                optimizer.zero_grad()
                # loss = ddputils.reduce_scalars(loss)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if ddputils.should_execute():
                    loader.set_postfix(
                        Loss=epoch_loss / (i_batch + 1),
                        Epoch=epoch,
                    )
                    loader.set_description(f"{epoch}/{args.max_epochs}")
                    loader.update()
                ddputils.synchronize()
            ddputils.synchronize()
            if ddputils.should_execute():
                print("Epoch loss = ", epoch_loss)

            # Evaluation
            if (epoch + 1) % args.valid_freq == 0:
                if ddputils.should_execute():
                    print("Starting validation")
                mrr_score, clf_score, choice_score, eval_log = eval_model(
                    qa_model,
                    valid_dataloader,
                    valid_dataset,
                    "valid",
                    args.batch_size,
                    ddputils.get_rank(),
                )
                ddputils.synchronize()

                if ddputils.should_execute():
                    save = False

                    # If the validation results increase, save checkpoint
                    if mrr_score > max_mrr:
                        save = True
                        eval_log.append(
                            f"Valid mrr score increased from {max_mrr} to {mrr_score}"
                        )
                        print(
                            f"Valid mrr score increased from {max_mrr} to {mrr_score}"
                        )
                        max_mrr = mrr_score
                    if clf_score > max_clf_score:
                        save = True
                        eval_log.append(
                            f"Valid yes_no score increased from {max_clf_score} to {clf_score}"
                        )
                        print(
                            f"Valid yes_no score increased from {max_clf_score} to {clf_score}"
                        )
                        max_clf_score = clf_score
                    if choice_score > max_choice_score:
                        save = True
                        eval_log.append(
                            f"Valid choice score increased from {max_choice_score} to {choice_score}"
                        )
                        print(
                            f"Valid choice score increased from {max_choice_score} to {choice_score}"
                        )
                        max_choice_score = choice_score
                    if save:
                        save_model(qa_model, checkpoint_file_name)
                    # Log the validation results
                    append_log_to_file(eval_log, epoch, log_filename)
                ddputils.synchronize()

                # Also do test after validation
                if ddputils.should_execute():
                    print("Start testing")
                _, _, _, log = eval_model(
                    qa_model,
                    test_dataloader,
                    test_dataset,
                    "test",
                    args.batch_size,
                    ddputils.get_rank(),
                )
                ddputils.synchronize()
                if ddputils.should_execute():
                    append_log_to_file(log, epoch, log_filename)

    # Set args.train to False, args.eval to True, for evaluation only
    if args.eval:
        if ddputils.should_execute():
            print("Starting evaluation")
        _, _, _, log = eval_model(
            qa_model,
            test_dataloader,
            test_dataset,
            "test",
            args.batch_size,
            ddputils.get_rank(),
        )
        append_log_to_file(log, 0, log_filename)
