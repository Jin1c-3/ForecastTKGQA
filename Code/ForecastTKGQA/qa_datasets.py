from pathlib import Path
import pkg_resources
import pickle
import os
from collections import defaultdict
from typing import Dict, Tuple, List
import json
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
import utils
from tqdm import tqdm
from transformers import RobertaTokenizer
from transformers import DistilBertTokenizer
from transformers import BertTokenizer
import random
from torch.utils.data import Dataset, DataLoader

# All question types in ForecastTKGQuestions
question_types = ['entity_prediction', 'yes_unknown', 'fact_reasoning']


class QADataset(Dataset): # Base class of dataset
    def __init__(self, split, dataset_name, question_type, tokenization_needed=True, pct=1):
        self.type2count = {}
        self.split = split
        if not os.path.exists("data/ICEWS21/filter_dict.pkl"):
            self.filter_dict = {}
        questions_all = []
        if question_type in question_types:
            question_type = [question_type]
        elif question_type == 'all':
            question_type = question_types
        else:
            raise ValueError(f"question type {question_type} undefined")

        for q_type in question_type:
            # Choose questions with question type
            filename = 'data/{dataset_name}/questions/{q_type}/{split}.pickle'.format(
                dataset_name=dataset_name,
                q_type=q_type,
                split=split
            )
            to_load = open(filename, 'rb')

            # Load questions
            questions = pickle.load(to_load)
            to_load.close()
            questions = random.sample(questions, int(len(questions) * float(pct)))
            questions_all.extend(questions)
            self.type2count[q_type] = len(questions)

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.all_dicts = utils.getAllDicts(dataset_name)
        print('Total questions = ', len(questions_all))
        self.data = questions_all
        self.tokenization_needed = tokenization_needed

    # =================== preparation ===================================================

    def get_loc_ent_pairs(self, question):
        question_text = question['question']  # Question
        entities = question['entities']  # Annotated entities in the question
        ent2id = self.all_dicts['ent2id']  # Entity to id, e.g., Q0 -> 0
        loc_ent = []
        for e in entities:
            e_id = ent2id[e]
            location = question_text.find(e)
            loc_ent.append((location, e_id))
        return loc_ent

    def get_loc_time_pairs(self, question):
        question_text = question['question'] # Question
        times = question['times']  # Annotated timestamps in the question
        ts2id = self.all_dicts['ts2id']  # Timestamp to id
        loc_time = []
        for t in times:
            t_id = ts2id[(t, 0, 0)]
            location = question_text.find(str(t))
            loc_time.append((location, t_id))
        return loc_time

    def get_item_from_dict(self, dictionary, dictionary_type):
        text = dictionary['paraphrases'] # Natural language text of the question
        if dictionary_type == 'question':
            q_type = self.type_to_id(dictionary['type'])
            answer_type = dictionary['answer_type']
            if answer_type == 'entity':
                answers = self.entities_to_ids(dictionary['answers'])
            elif answer_type == 'label':
                answers = list(dictionary['answers'])
            else:
                raise ValueError(f'wrong answer type {dictionary["answer_type"]}')
        else:
            dictionary['question'] = dictionary.pop('choice')
            q_type, answers = -1, -1

        entities_list_with_locations = self.get_loc_ent_pairs(dictionary)
        entities_list_with_locations.sort()
        # Ordering necessary otherwise set->list conversion causes randomness
        entities = [idx for location, idx in entities_list_with_locations]
        head = entities[0]  # Take an entity
        if len(entities) > 1: # If more than one annotated entity in the question
            tail = entities[1]
        else:
            tail = entities[0]
        times_in_question = dictionary['times']
        if len(times_in_question) > 0:
            time = self.times_to_ids(times_in_question)[0]  # Take a timestamp
        else: # If no time raise error
            raise ValueError(f'{dictionary_type} {dictionary["uniq_id"]} contains no time')

        return text, head, tail, time, q_type, answers

    def create_ep_filter(self):
        for i, question in enumerate(self.data):
            self.data_ids_filtered.append(i)
            q_text, q_head, q_tail, q_time, q_type, answers = self.get_item_from_dict(question, 'question')
            self.data_filter(q_text, q_text, q_tail, q_time, answers)

        # with open('data/ICEWS21/filter_dict_{}.pkl'.format(self.split), 'wb') as f:
        #     pickle.dump(self.filter_dict, f)
        #     print("Dict_{} for EP created successfully.".format(self.split))

    # ====================== dictionaries ============================================
    def data_filter(self, text, head, tail, time, answers):
        if text not in self.filter_dict.keys():
            self.filter_dict.update({text: [head, tail, time, [answers]]})
        else:
            self.filter_dict[text][-1].extend([answers])
    @staticmethod
    def check_time_string(s):
        if 'Q' not in s:
            return True
        else:
            return False

    def text_to_id(self, text):
        if self.check_time_string(text):
            t = int(text)
            ts2id = self.all_dicts['ts2id']
            t_id = ts2id[(t, 0, 0)]
            return t_id
        else:
            ent2id = self.all_dicts['ent2id']
            e_id = ent2id[text]
            return e_id

    def entities_to_ids(self, entities):
        output = []
        ent2id = self.all_dicts['ent2id']
        for e in entities:
            output.append(ent2id[e])
        return output

    def entity_to_text(self, entity_wd_id): # Entity to entity text, e.g., Q0 -> Ndianghta
        return self.all_dicts['wd_id_to_text'][entity_wd_id]

    def entity_id_to_text(self, ent_id): # Entity id to entity text, e.g., 0 -> Ndianghta
        ent = self.all_dicts['id2ent'][ent_id]
        return self.entity_to_text(ent)

    def entity_id_to_entity(self, ent_id): # Entity id to entity, e.g., 0 -> Q0
        return self.all_dicts['id2ent'][ent_id]

    def times_to_ids(self, times):
        output = []
        ts2id = self.all_dicts['ts2id']
        for t in times:
            output.append(ts2id[(t, 0, 0)])
        return output

    def type_to_id(self, type_text):
        return self.all_dicts['type2id'][type_text]

    def types_to_ids(self, types):
        output = []
        for t in types:
            output.append(self.type_to_id(t))
        return output

    def ids_to_types(self, ids):
        output = []
        id2type = self.all_dicts['id2type']
        for idx in ids:
            output.append(id2type[idx])
        return output

    # ====================== get results ===================================================
    @staticmethod
    def get_answers_from_scores(scores, largest=True, k=10): # Get answers from ranking scores
        _, predict = torch.topk(scores, k, largest=largest)
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            answers.append(a_id)
        return answers

    @staticmethod
    def get_rank_from_scores(scores, answer): # Get ranks of the ground-truth entities
        _, predict = torch.sort(scores, descending=True)
        rank = (predict == answer).nonzero().item() + 1
        return 1 / rank

    @staticmethod
    def padding_tensor(sequences, max_len=-1):
        num = len(sequences)
        if max_len == -1:
            max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        mask = torch.ones((num, max_len), dtype=torch.bool)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = False
        return out_tensor, mask

    # ================== get meta info ========================================
    def __len__(self):
        return len(self.data)

    def get_dataset_ques_info(self):
        type2num = {}
        for question in self.data:
            if question["type"] not in type2num:
                type2num[question["type"]] = 0
            type2num[question["type"]] += 1
        return {"type2num": type2num, "total_num": len(self.data)}.__str__()

    def print_prepared_data(self):
        for k, v in self.prepared_data.items():
            print(k, v[:5])
            print(k, v[-5:])


class QADatasetBert(QADataset): # Dataset class for LM baselines, without TKG representations
    def __init__(self, split, dataset_name, question_type, tokenization_needed=True, pct=1):
        super().__init__(split, dataset_name, question_type, tokenization_needed, pct)
        print('Preparing data for split %s' % split)
        self.prepared_data = self.prepare_data_(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])

    def prepare_data_(self, data):
        question_text = []
        types = []
        answers_arr = []
        self.data_ids_filtered = []
        for i, question in enumerate(data):
            self.data_ids_filtered.append(i)
            q_text, q_head, q_tail, q_time, q_type, answers = self.get_item_from_dict(question, 'question')
            types.append(q_type)
            answers_arr.append(answers)
            if q_type == 3: # If question is fact reasoning
                question_text.append([])
                question_text[-1].append(q_text)
                choices = question['choices']
                for choice_dict in choices:
                    choice_text, choice_head, choice_tail, choice_time, _, _ = self.get_item_from_dict(choice_dict,
                                                                                                       'choice')
                    question_text[-1].append(choice_text) # Concatenation of question and choice
            else:
                question_text.append(q_text)
        # if not os.path.exists('data/ICEWS21/filter_dict_{}.pkl'.format(self.split)):
        #     with open('data/ICEWS21/filter_dict_{}.pkl'.format(self.split), 'wb') as f:
        #         pickle.dump(self.filter_dict, f)
        #         print("Dict_{} for EP created successfully.".format(self.split))
        self.data = [self.data[idx] for idx in self.data_ids_filtered]
        return {'question_text': question_text,
                'answers_arr': answers_arr,
                'type': types}

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        q_type = data['type'][index]
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        return question_text, q_type, answers_single

    def collate_fn(self, items):
        batch_sentences = [item[0] for item in items]
        if items[0][1] < 3:
            b = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        else:
            qc = []
            for sen in batch_sentences:
                question = sen[0]
                choices = sen[1:]
                for choice in choices:
                    qc.append([])
                    qc[-1].append(question)
                    qc[-1].append(choice)
            b = self.tokenizer(qc, padding=True, truncation=True, return_tensors="pt")
        b_input_id, b_attention_mask = b['input_ids'], b['attention_mask']
        types = torch.from_numpy(np.array([item[1] for item in items]))
        answers_single = torch.from_numpy(np.array([item[2] for item in items]))
        return b_input_id, b_attention_mask, types, answers_single, batch_sentences


class QADatasetBaseline(QADataset): # Dataset class for LM variants, EmbedKGQA, and CronKGQA
    def __init__(self, split, dataset_name, question_type, tokenization_needed=True, pct=1):
        super().__init__(split, dataset_name, question_type, tokenization_needed, pct)
        print('Preparing data for split %s' % split)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.prepared_data = self.prepare_data_(self.data)

    def prepare_data_(self, data):
        question_text = []
        heads = []
        tails = []
        times = []
        types = []
        answers_arr = []
        self.data_ids_filtered = []
        for i, question in enumerate(data):
            self.data_ids_filtered.append(i)
            q_text, q_head, q_tail, q_time, q_type, answers = self.get_item_from_dict(question, 'question')
            types.append(q_type)
            answers_arr.append(answers)
            if q_type == 3: # If question is fact reasoning
                question_text.append([])
                heads.append([])
                tails.append([])
                times.append([])
                question_text[-1].append(q_text)
                heads[-1].append(q_head)
                tails[-1].append(q_tail)
                times[-1].append(q_time)
                choices = question['choices']
                for choice_dict in choices:
                    choice_text, choice_head, choice_tail, choice_time, _, _ = self.get_item_from_dict(choice_dict,
                                                                                                       'choice')
                    question_text[-1].append(choice_text) # Concatenation of question and choice
                    heads[-1].append(choice_head)
                    tails[-1].append(choice_tail)
                    times[-1].append(choice_time)
            else:
                question_text.append(q_text)
                heads.append(q_head)
                tails.append(q_tail)
                times.append(q_time)
        # if not os.path.exists('data/ICEWS21/filter_dict_{}.pkl'.format(self.split)):
        #     with open('data/ICEWS21/filter_dict_{}.pkl'.format(self.split), 'wb') as f:
        #         pickle.dump(self.filter_dict, f)
        #         print("Dict_{} for EP created successfully.".format(self.split))
        self.data = [self.data[idx] for idx in self.data_ids_filtered]
        return {'question_text': question_text,
                'head': heads,
                'tail': tails,
                'time': times,
                'answers_arr': answers_arr,
                'type': types}

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        time = data['time'][index]
        q_type = data['type'][index]
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        return question_text, head, tail, time, q_type, answers_single

    def collate_fn(self, items):
        batch_sentences = [item[0] for item in items]
        if items[0][4] < 3:
            b = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        else:
            qc = []
            for sen in batch_sentences:
                question = sen[0]
                choices = sen[1:]
                for choice in choices:
                    qc.append([])
                    qc[-1].append(question)
                    qc[-1].append(choice)
            b = self.tokenizer(qc, padding=True, truncation=True, return_tensors="pt")
        b_input_id, b_attention_mask = b['input_ids'], b['attention_mask']
        heads = torch.from_numpy(np.array([item[1] for item in items]))
        tails = torch.from_numpy(np.array([item[2] for item in items]))
        times = torch.from_numpy(np.array([item[3] for item in items]))
        types = torch.from_numpy(np.array([item[4] for item in items]))
        answers_single = torch.from_numpy(np.array([item[5] for item in items]))
        return b_input_id, b_attention_mask, heads, tails, times, types, answers_single, batch_sentences


class QADatasetForecast(QADataset): # Dataset class for ForecastTKGQA
    def __init__(self, split, dataset_name, question_type, tokenization_needed=True, pct=1):
        super().__init__(split, dataset_name, question_type, tokenization_needed, pct)
        print('Preparing data for split %s' % split)
        self.prepared_data = self.prepare_data_(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])

    def prepare_data_(self, data):
        question_text = []
        heads = []
        tails = []
        times = []
        types = []
        answers_arr = []
        self.data_ids_filtered = []
        for i, question in enumerate(data):
            self.data_ids_filtered.append(i)
            q_text, q_head, q_tail, q_time, q_type, answers = self.get_item_from_dict(question, 'question')
            types.append(q_type)
            answers_arr.append(answers)
            if q_type == 3: # If question is fact reasoning
                question_text.append([])
                heads.append([])
                tails.append([])
                times.append([])
                question_text[-1].append(q_text)
                heads[-1].append(q_head)
                tails[-1].append(q_tail)
                times[-1].append(q_time)
                choices = question['choices']
                for choice_dict in choices:
                    choice_text, choice_head, choice_tail, choice_time, _, _ = self.get_item_from_dict(choice_dict,
                                                                                                       'choice')
                    question_text[-1].append(choice_text) # Concatenation of question and choice
                    heads[-1].append(choice_head)
                    tails[-1].append(choice_tail)
                    times[-1].append(choice_time)
            else:
                if q_type == 2:
                    question_text.append([q_text, 'unknown', 'yes', '', ''])
                else:
                    question_text.append([q_text, '', '', '', ''])
                heads.append([q_head, -1, -1, -1, -1])
                tails.append([q_tail, -1, -1, -1, -1])
                times.append([q_time, -1, -1, -1, -1])
        # if not os.path.exists('data/ICEWS21/filter_dict_{}.pkl'.format(self.split)):
        #     with open('data/ICEWS21/filter_dict_{}.pkl'.format(self.split), 'wb') as f:
        #         pickle.dump(self.filter_dict, f)
        #         print("Dict_{} for EP created successfully.".format(self.split))
        self.data = [self.data[idx] for idx in self.data_ids_filtered]
        return {'question_text': question_text,
                'head': heads,
                'tail': tails,
                'time': times,
                'answers_arr': answers_arr,
                'type': types}

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        time = data['time'][index]
        q_type = data['type'][index]
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        return question_text, head, tail, time, q_type, answers_single

    def collate_fn(self, items):
        b_input_id, b_attention_mask, batch_sentences = [], [], []
        for item in items:
            sentences = item[0]
            item_type = item[4]
            batch_sentences.append(sentences[0])
            if item_type < 3:
                tokenized = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            else:
                qc = []
                question = sentences[0]
                choices = sentences[1:]
                qc.append([question, ""])
                for choice in choices:
                    qc.append([])
                    qc[-1].append(question)
                    qc[-1].append(choice)
                tokenized = self.tokenizer(qc, padding=True, truncation=True, return_tensors="pt")
            b_input_id.append(tokenized['input_ids'].t())
            b_attention_mask.append(tokenized['attention_mask'].t())
        b_input_id = pad_sequence(b_input_id, batch_first=True)
        b_attention_mask = pad_sequence(b_attention_mask, batch_first=True)
        b_input_id = torch.transpose(b_input_id, 1, 2)
        b_attention_mask = torch.transpose(b_attention_mask, 1, 2)
        heads = torch.from_numpy(np.array([item[1] for item in items]))
        tails = torch.from_numpy(np.array([item[2] for item in items]))
        times = torch.from_numpy(np.array([item[3] for item in items]))
        types = torch.from_numpy(np.array([item[4] for item in items]))
        answers_single = torch.from_numpy(np.array([item[5] for item in items]))
        return b_input_id, b_attention_mask, heads, tails, times, types, answers_single, batch_sentences


class QADatasetTempoQR(QADataset): # Dataset class for TempoQR, adapted from TempoQR official repository
    def __init__(self, split, dataset_name, question_type, tokenization_needed=True, pct=1, annotate_time=True):
        super().__init__(split, dataset_name, question_type, tokenization_needed, pct)
        print('Preparing data for split %s' % split)
        self.annotate_time = annotate_time
        self.all_dicts['tsstr2id'] = {str(k[0]): v for k, v in self.all_dicts['ts2id'].items()}
        self.split = split

        self.data = self.addEntityAnnotation(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        if annotate_time:
            self.padding_idx = self.num_total_entities + self.num_total_times
        else:
            self.padding_idx = self.num_total_entities

        self.prepared_data = self.prepare_data_(self.data)

    def __len__(self):
        return len(self.data)

    def is_template_keyword(self, word):
        if '{' in word and '}' in word:
            return True
        else:
            return False

    def get_keyword_dict(self, template, nl_question):
        template_tokenized = self.tokenize_template(template)
        keywords = []
        for word in template_tokenized:
            if not self.is_template_keyword(word):
                # Replace only first occurence
                nl_question = nl_question.replace(word, '*', 1)
            else:
                keywords.append(word[1:-1])  # no brackets
        text_for_keywords = []
        for word in nl_question.split('*'):
            if word != '':
                text_for_keywords.append(word)
        keyword_dict = {}
        for keyword, text in zip(keywords, text_for_keywords):
            keyword_dict[keyword] = text
        return keyword_dict

    def addEntityAnnotation(self, data):
        for i in range(len(data)):
            question = data[i]
            keyword_dicts = []
            template = question['template']
            nl_question = question['paraphrases']
            keyword_dict = self.get_keyword_dict(template, nl_question)
            keyword_dicts.append(keyword_dict)
            data[i]['keyword_dicts'] = keyword_dicts
        return data

    def tokenize_template(self, template):
        output = []
        buffer = ''
        i = 0
        while i < len(template):
            c = template[i]
            if c == '{':
                if buffer != '':
                    output.append(buffer)
                    buffer = ''
                while template[i] != '}':
                    buffer += template[i]
                    i += 1
                buffer += template[i]
                output.append(buffer)
                buffer = ''
            else:
                buffer += c
            i += 1
        if buffer != '':
            output.append(buffer)
        return output

    def getEntityTimeTextIds(self, question, pp_id=0):
        keyword_dict = question['keyword_dicts'][pp_id]
        keyword_id_dict = {'source': list(question['entities'])[0],
                           'time': list(question['times'])[0]}
        output_text = []
        output_ids = []
        if self.annotate_time:
            entity_time_keywords = {'source', 'time'}
        else:
            entity_time_keywords = {'source'}
        for keyword, value in keyword_dict.items():
            if keyword in entity_time_keywords:
                wd_id_or_time = keyword_id_dict[keyword]
                output_text.append(value)
                output_ids.append(wd_id_or_time)
        return output_text, output_ids

    def get_entity_aware_tokenization(self, nl_question, ent_times, ent_times_ids):
        index_et_pairs = []
        index_et_text_pairs = []
        for e_text, e_id in zip(ent_times, ent_times_ids):
            location = nl_question.find(e_text)
            pair = (location, e_id)
            index_et_pairs.append(pair)
            pair = (location, e_text)
            index_et_text_pairs.append(pair)
        index_et_pairs.sort()
        index_et_text_pairs.sort()
        my_tokenized_question = []
        start_index = 0
        arr = []
        for pair, pair_id in zip(index_et_text_pairs, index_et_pairs):
            end_index = pair[0]
            if nl_question[start_index: end_index] != '':
                my_tokenized_question.append(nl_question[start_index: end_index])
                arr.append(self.padding_idx)
            start_index = end_index
            end_index = start_index + len(pair[1])
            my_tokenized_question.append(self.tokenizer.mask_token)
            matrix_id = self.text_to_id(str(pair_id[1]))  # get id in embedding matrix
            arr.append(matrix_id)
            start_index = end_index
        if nl_question[start_index:] != '':
            my_tokenized_question.append(nl_question[start_index:])
            arr.append(self.padding_idx)

        tokenized, valid_ids = self.tokenize(my_tokenized_question)
        entity_time_final = []
        index = 0
        for vid in valid_ids:
            if vid == 0:
                entity_time_final.append(self.padding_idx)
            else:
                entity_time_final.append(arr[index])
                index += 1
        entity_mask = []
        for x in entity_time_final:
            if x == self.padding_idx:
                entity_mask.append(1.)
            else:
                entity_mask.append(0.)

        # print(entity_time_final)
        return tokenized, entity_time_final, entity_mask

    def prepare_data_(self, data):
        heads = []
        times = []
        start_times = []
        end_times = []
        tails = []
        tails2 = []
        question_text = []
        tokenized_question = []
        entity_time_ids_tokenized_question = []
        entity_mask_tokenized_question = []
        num_total_entities = len(self.all_dicts['ent2id'])
        types = []
        answers_arr = []
        for question in tqdm(data):
            pp_id = 0
            nl_question = question['paraphrases']
            et_text, et_ids = self.getEntityTimeTextIds(question, pp_id)

            entities_list_with_locations = self.get_loc_ent_pairs(question)
            entities_list_with_locations.sort()
            entities = [id for location, id in
                        entities_list_with_locations]  # Ordering necessary otherwise set->list conversion causes randomness
            head = entities[0]  # Take an entity
            if len(entities) > 1: # If more than one annotated entity in the question
                tail = entities[1]
                if len(entities) > 2:
                    tail2 = entities[2]
                else:
                    tail2 = tail
            else:
                tail = entities[0]
                tail2 = tail
            times_in_question = question['times']
            time = self.times_to_ids(times_in_question)[0]  # Take a timestamp
            start_time = time
            end_time = time

            if self.annotate_time:
                time += num_total_entities

            heads.append(head)
            times.append(time)
            start_times.append(start_time)
            end_times.append(end_time)
            tails.append(tail)
            tails2.append(tail2)

            tokenized, entity_time_final, entity_mask = self.get_entity_aware_tokenization(nl_question, et_text, et_ids)
            assert len(tokenized) == len(entity_time_final)
            question_text.append(nl_question)
            tokenized_question.append(self.tokenizer.convert_tokens_to_ids(tokenized))
            entity_mask_tokenized_question.append(entity_mask)
            entity_time_ids_tokenized_question.append(entity_time_final)
            types.append(self.type_to_id(question['type']))
            if question['answer_type'] == 'entity':
                answers = self.entities_to_ids(question['answers'])
            else:
                answers = [x + num_total_entities for x in self.times_to_ids(question['answers'])]
            answers_arr.append(answers)
        return {'question_text': question_text,
                'tokenized_question': tokenized_question,
                'entity_time_ids': entity_time_ids_tokenized_question,
                'entity_mask': entity_mask_tokenized_question,
                'head': heads,
                'tail': tails,
                'time': times,
                'start_time': start_times,
                'end_time': end_times,
                'tail2': tails2,
                'type': types,
                'answers_arr': answers_arr}

    def tokenize(self, words):
        """ tokenize input"""
        tokens = []
        valid_positions = []
        tokens.append(self.tokenizer.cls_token)
        valid_positions.append(0)
        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        tokens.append(self.tokenizer.sep_token)
        valid_positions.append(0)
        return tokens, valid_positions

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        entity_time_ids = np.array(data['entity_time_ids'][index], dtype=np.long)
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        q_type = data['type'][index]
        # answers_khot = self.toOneHot(answers_arr, self.answer_vec_size)
        tokenized_question = data['tokenized_question'][index]
        entity_mask = data['entity_mask'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        tail2 = data['tail2'][index]
        time = data['time'][index]
        start_time = data['start_time'][index]
        end_time = data['end_time'][index]

        return question_text, tokenized_question, entity_time_ids, entity_mask, head, tail, time, start_time, end_time, tail2, q_type, answers_single

    def pad_for_batch(self, to_pad, padding_val, dtype=np.long):
        padded = np.ones([len(to_pad), len(max(to_pad, key=lambda x: len(x)))], dtype=dtype) * padding_val
        for i, j in enumerate(to_pad):
            padded[i][0:len(j)] = j
        return padded

    def get_attention_mask(self, tokenized):
        mask = np.zeros([len(tokenized), len(max(tokenized, key=lambda x: len(x)))], dtype=np.long)
        for i, j in enumerate(tokenized):
            mask[i][0:len(j)] = np.ones(len(j), dtype=np.long)
        return mask

    def collate_fn(self, items):
        batch_sentences = [item[0] for item in items]
        tokenized_questions = [item[1] for item in items]
        attention_mask = torch.from_numpy(self.get_attention_mask(tokenized_questions))
        input_ids = torch.from_numpy(self.pad_for_batch(tokenized_questions, self.tokenizer.pad_token_id, np.long))

        entity_time_ids_list = [item[2] for item in items]
        entity_time_ids_padded = self.pad_for_batch(entity_time_ids_list, self.padding_idx, np.long)
        entity_time_ids_padded = torch.from_numpy(entity_time_ids_padded)

        entity_mask = [item[3] for item in items]  # 0 if entity, 1 if not
        entity_mask_padded = self.pad_for_batch(entity_mask, 1.0,
                                                np.float32)
        entity_mask_padded = torch.from_numpy(entity_mask_padded)

        heads = torch.from_numpy(np.array([item[4] for item in items]))
        tails = torch.from_numpy(np.array([item[5] for item in items]))
        times = torch.from_numpy(np.array([item[6] for item in items]))
        start_times = torch.from_numpy(np.array([item[7] for item in items]))
        end_times = torch.from_numpy(np.array([item[8] for item in items]))

        tails2 = torch.from_numpy(np.array([item[9] for item in items]))

        types = torch.from_numpy(np.array([item[10] for item in items]))
        answers_single = torch.from_numpy(np.array([item[11] for item in items]))

        return input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times, end_times, tails2, types, answers_single, batch_sentences
