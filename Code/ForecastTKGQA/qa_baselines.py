import math
import torch
from torch import nn
import numpy as np
from tcomplex import TComplEx
from utils import load_tcomplex, load_complex, load_tango
from transformers import BertModel, RobertaModel, DistilBertModel
from torch.nn import LayerNorm


class QABaseline(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Load pre-trained language model
        self.sentence_embedding_dim = 768
        if args.lm_model == 'roberta':
            self.pretrained_weights = 'roberta-base'
            self.lm_model = RobertaModel.from_pretrained(self.pretrained_weights)
        elif args.lm_model == 'distilbert':
            self.pretrained_weights = '/data/qing/distilbert-base-uncased'
            self.lm_model = DistilBertModel.from_pretrained(self.pretrained_weights)
        else:
            raise ValueError('lm model undefined')
        for param in self.lm_model.parameters():
            param.requires_grad = False

        # TKG model
        if args.tkg_model_file:
            if 'complex' in args.tkg_model_file: # Load pre-trained TComplEx representations
                if args.model != 'embedkgqa':
                    self.tcomplex_model = load_tcomplex('models/{dataset_name}/kg_embeddings/{tkg_model_file}'.format(
                        dataset_name=args.dataset_name, tkg_model_file=args.tkg_model_file), args.device)
                else: # EmbedKGQA loads ComplEx
                    self.tcomplex_model = load_complex('models/{dataset_name}/kg_embeddings/{tkg_model_file}'.format(
                        dataset_name=args.dataset_name, tkg_model_file=args.tkg_model_file), args.device)
                self.tkg_embedding_dim = self.tcomplex_model.embeddings[0].weight.shape[1]
                self.rank = int(self.tkg_embedding_dim // 2)
                print(f'tkg embed: {self.tkg_embedding_dim}')
                self.num_entities = self.tcomplex_model.embeddings[0].weight.shape[0]
                self.num_times = self.tcomplex_model.embeddings[2].weight.shape[0]
                ent_emb_matrix = self.tcomplex_model.embeddings[0].weight.data
                time_emb_matrix = self.tcomplex_model.embeddings[2].weight.data
                full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
                self.tcomplex_embedding = nn.Embedding(self.num_entities + self.num_times + 1,
                                                       self.tkg_embedding_dim,
                                                       padding_idx=self.num_entities + self.num_times)
                self.tcomplex_embedding.weight.data[:-1, :].copy_(full_embed_matrix)
                self.tcomplex_embedding.weight.requires_grad = False
                for param in self.tcomplex_model.parameters():
                    param.requires_grad = False

            else: # Load pre-trained TANGO representations
                self.tango_model = load_tango('models/{dataset_name}/kg_embeddings/{tkg_model_file}'.format(
                    dataset_name=args.dataset_name, tkg_model_file=args.tkg_model_file))
                self.tkg_embedding_dim = list(self.tango_model.values())[0].shape[1]
                self.num_entities = list(self.tango_model.values())[0].shape[0]
                self.tango_weights = nn.Parameter(
                    torch.zeros((len(self.tango_model), self.num_entities+1, self.tkg_embedding_dim)), requires_grad=False)
                torch.nn.init.xavier_uniform_(self.tango_weights)
                self.tango_weights[:, :-1, :] = torch.stack(list(self.tango_model.values()))
                self.tango_embedding = nn.Parameter(self.tango_weights.view(-1, self.tkg_embedding_dim))
                self.rank = int(self.tkg_embedding_dim / 2)
                print(f'tkg embed: {self.tkg_embedding_dim}')
                self.tango_embedding.requires_grad = False
                self.tango_weights.requires_grad = False

            self.linear_relation = nn.Linear(768, self.tkg_embedding_dim) # To project question representation from 768 to self.tkg_embedding_dim
            self.linear_bn = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)
            self.bn = torch.nn.BatchNorm1d(self.tkg_embedding_dim)
            self.dropout = torch.nn.Dropout(0.3)

        self.loss = nn.CrossEntropyLoss(reduction='mean')
        if args.question_type == 'entity_prediction':
            self.out_dim = 20575 # Number of entity since the answer is an entity
        elif args.question_type == 'yes_no':
            self.out_dim = 2 # Answer is either yes or no
        elif args.question_type == 'fact_reasoning':
            self.out_dim = 1 # Corresponding to one choice
        else:
            raise ValueError(f'question type undefined: {args.question_type}')

    def get_question_embedding(self, question_tokenized, attention_mask):
        lm_last_hidden_states = self.lm_model(question_tokenized, attention_mask=attention_mask)[0]
        states = lm_last_hidden_states.transpose(1, 0)
        question_embedding = states[0]

        return question_embedding

    @staticmethod
    def prepare_forward(a, device):
        tokenized = a[0].cuda(device)
        attention_mask = a[1].cuda(device)
        heads = a[2].cuda(device)
        tails = a[3].cuda(device)
        times = a[4].cuda(device)
        q_type = a[5][0].item()
        answers = a[6].cuda(device)
        if q_type < 3: # Entity prediction and yes-no
            return tokenized, attention_mask, heads, tails, times, answers
        else:
            q_heads, c_heads = heads[:, 0], torch.flatten(heads[:, 1:])
            q_tails, c_tails = tails[:, 0], torch.flatten(tails[:, 1:])
            q_times, c_times = times[:, 0], torch.flatten(times[:, 1:])
            return tokenized, attention_mask, q_heads, q_tails, q_times, c_heads, c_tails, c_times, answers


class QABert(QABaseline): # Model class for LM baselines, without TKG representations
    def __init__(self, args):
        super().__init__(args)
        self.final_linear = nn.Linear(self.sentence_embedding_dim, self.out_dim)

    def forward(self, a, device):
        scores_ep, scores_yn, scores_mc = None, None, None
        q_tokenized = a[0].cuda(device)
        q_attention_mask = a[1].cuda(device)
        answers = a[3].cuda(device)
        question_embedding = self.get_question_embedding(q_tokenized, q_attention_mask)
        # Score computation with a prediction head (Appendix B.2)
        scores = self.final_linear(question_embedding)
        if self.out_dim == 2:
            scores_yn = scores
        elif self.out_dim > 2:
            scores_ep = scores
        else:
            scores = scores.view(-1, 4)
            scores_mc = scores
        loss = self.loss(scores, answers)
        return None, scores_ep, scores_yn, scores_mc, loss


class QABERTTComplEx(QABaseline): # Model class for LM variants BERT_int and RoBERTa_int
    def __init__(self, args):
        super().__init__(args)
        if self.out_dim > 2:
            self.final_linear = nn.Linear(4 * self.tkg_embedding_dim, self.tkg_embedding_dim)
        elif self.out_dim == 2:
            self.final_linear = nn.Linear(4 * self.tkg_embedding_dim, self.out_dim)
        else:
            self.linear_cat_q = nn.Linear(4 * self.tkg_embedding_dim, self.tkg_embedding_dim)
            self.linear_cat_c = nn.Linear(4 * self.tkg_embedding_dim, self.tkg_embedding_dim)
            self.final_linear = nn.Linear(2 * self.tkg_embedding_dim, self.out_dim)

    def forward(self, a, device):
        scores_ep, scores_yn, scores_mc = None, None, None
        if self.out_dim > 1:  # Entity prediction and yes-no
            q_tokenized, q_attention_mask, heads, tails, times, answers = \
                self.prepare_forward(a, device)
            question_emb = self.get_question_embedding(q_tokenized, q_attention_mask)
            relation_emb = self.linear_relation(question_emb)
            head_emb = self.tcomplex_embedding(heads)
            tail_emb = self.tcomplex_embedding(tails)
            time_emb = self.tcomplex_embedding(times + self.num_entities)
            # Score computation with a prediction head
            output = self.final_linear(torch.cat((head_emb, relation_emb, tail_emb, time_emb), dim=-1))
            if self.out_dim == 2:
                scores_yn = output
                loss = self.loss(scores_yn, answers)
            else:
                scores_ep = torch.matmul(output, self.tcomplex_embedding.weight.data[:-1, :].T)
                loss = self.loss(scores_ep, answers)
        else: # Fact reasoning
            qc_tokenized, qc_attention_mask, q_head, q_tail, q_time, c_head, c_tail, c_time, answers = \
                self.prepare_forward(a, device)
            qc_emb = self.get_question_embedding(qc_tokenized, qc_attention_mask)
            qc_relation = self.linear_relation(qc_emb)
            # Concatenate the representations of the annotated entities and timestamps in the questions
            q_head_emb = self.tcomplex_embedding(q_head)
            q_tail_emb = self.tcomplex_embedding(q_tail)
            q_time_emb = self.tcomplex_embedding(q_time + self.num_entities)
            q_head_emb = torch.repeat_interleave(q_head_emb, 4, dim=0)
            q_tail_emb = torch.repeat_interleave(q_tail_emb, 4, dim=0)
            q_time_emb = torch.repeat_interleave(q_time_emb, 4, dim=0)
            q_cat = torch.cat((q_head_emb, qc_relation, q_tail_emb, q_time_emb), dim=-1)
            q_cat = self.linear_cat_q(q_cat)
            # Concatenate the representations of the annotated entities and timestamps in the choices
            c_head_emb = self.tcomplex_embedding(c_head)
            c_tail_emb = self.tcomplex_embedding(c_tail)
            c_time_emb = self.tcomplex_embedding(c_time + self.num_entities)
            c_cat = torch.cat((c_head_emb, qc_relation, c_tail_emb, c_time_emb), dim=-1)
            c_cat = self.linear_cat_c(c_cat)
            # Concatenate question information and choice information (Equation (12) in Appendix B.2)
            qc_cat = torch.cat((q_cat, c_cat), dim=-1)
            scores = self.final_linear(qc_cat)
            scores_mc = scores.view(-1, 4)
            loss = self.loss(scores_mc, answers)
        return None, scores_ep, scores_yn, scores_mc, loss


class QABertTango(QABaseline): # Model class for LM variants BERT_ext and RoBERTa_ext
    def __init__(self, args):
        super().__init__(args)
        if self.out_dim > 4:
            self.final_linear = nn.Linear(3 * self.tkg_embedding_dim, self.tkg_embedding_dim)
        elif self.out_dim == 2:
            self.final_linear = nn.Linear(3 * self.tkg_embedding_dim, self.out_dim)
        else:
            self.linear_cat_q = nn.Linear(3 * self.tkg_embedding_dim, self.tkg_embedding_dim)
            self.linear_cat_c = nn.Linear(3 * self.tkg_embedding_dim, self.tkg_embedding_dim)
            self.final_linear = nn.Linear(2 * self.tkg_embedding_dim, self.out_dim)

    def forward(self, a, device):
        scores_ep, scores_yn, scores_mc = None, None, None
        if self.out_dim > 1:  # Entity prediction and yes-no
            q_tokenized, q_attention_mask, heads, tails, times, answers = \
                self.prepare_forward(a, device)
            question_emb = self.get_question_embedding(q_tokenized, q_attention_mask)
            relation_emb = self.linear_relation(question_emb)
            head_emb = self.tango_weights[[times, heads]]
            tail_emb = self.tango_weights[[times, tails]]
            # Score computation with a prediction head
            output = self.final_linear(torch.cat((head_emb, relation_emb, tail_emb), dim=-1))
            if self.out_dim == 2:
                scores_yn = output
                loss = self.loss(scores_yn, answers)
            else:
                right = self.tango_weights[times]
                scores_ep = torch.bmm(output.unsqueeze(1), torch.transpose(right, 1, 2)).squeeze()
                loss = self.loss(scores_ep, answers)
        else: # Fact reasoning
            qc_tokenized, qc_attention_mask, q_head, q_tail, q_time, c_head, c_tail, c_time, answers = \
                self.prepare_forward(a, device)
            qc_embedding = self.get_question_embedding(qc_tokenized, qc_attention_mask)
            qc_relation = self.linear_relation(qc_embedding)
            # Concatenate the TANGO representations of the annotated entities in the questions
            q_head_emb = self.tango_weights[[q_time, q_head]]
            q_tail_emb = self.tango_weights[[q_time, q_tail]]
            q_head_emb = torch.repeat_interleave(q_head_emb, 4, dim=0)
            q_tail_emb = torch.repeat_interleave(q_tail_emb, 4, dim=0)
            q_cat = torch.cat((q_head_emb, qc_relation, q_tail_emb), dim=-1)
            q_cat = self.linear_cat_q(q_cat)
            # Concatenate the TANGO representations of the annotated entities in the choices
            c_head_emb = self.tango_weights[[c_time, c_head]]
            c_tail_emb = self.tango_weights[[c_time, c_tail]]
            c_cat = torch.cat((c_head_emb, qc_relation, c_tail_emb), dim=-1)
            c_cat = self.linear_cat_c(c_cat)
            # Concatenate question information and choice information (Equation (13) in Appendix B.2)
            qc_cat = torch.cat((q_cat, c_cat), dim=-1)
            scores_mc = self.final_linear(qc_cat)
            scores_mc = scores_mc.view(-1, 4)
            loss = self.loss(scores_mc, answers)

        return None, scores_ep, scores_yn, scores_mc, loss


class EmbedKGQA(QABaseline): # Model class for EmbedKGQA, borrowed from the official repository
    def __init__(self, args):
        super().__init__(args)

    def score(self, head_embedding, relation_embedding): # Scoring function
        lhs = head_embedding
        rel = relation_embedding
        right = self.tcomplex_embedding.weight
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        right = right[:, :self.rank], right[:, self.rank:]
        return (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].transpose(0, 1) + \
               (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].transpose(0, 1)

    def forward(self, a, device):
        q_tokenized, q_attention_mask, heads, tails, times, answers = \
            self.prepare_forward(a, device)
        question_embedding = self.get_question_embedding(q_tokenized, q_attention_mask)
        relation_embedding = self.linear_relation(question_embedding)
        relation_embedding = self.dropout(self.bn(self.linear_bn(relation_embedding)))
        head_embedding = self.tcomplex_embedding(heads)
        scores = self.score(head_embedding, relation_embedding)
        loss = self.loss(scores, answers)
        return None, scores, None, None, loss


class CronKGQATComplEx(QABaseline): # Model class for CronKGQA, borrowed from the official repository
    def __init__(self, args):
        super().__init__(args)

    def score(self, head_embedding, relation_embedding, time_embedding): # Scoring function
        lhs = head_embedding[:, :self.rank], head_embedding[:, self.rank:]
        rel = relation_embedding[:, :self.rank], relation_embedding[:, self.rank:]
        time = time_embedding[:, :self.rank], time_embedding[:, self.rank:]

        right = self.tcomplex_model.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
        )

    def forward(self, a, device):
        q_tokenized, q_attention_mask, heads, tails, times, answers = \
            self.prepare_forward(a, device)
        head_embedding = self.tcomplex_embedding(heads)
        time_embedding = self.tcomplex_embedding(times + self.num_entities)
        question_embedding = self.get_question_embedding(q_tokenized, q_attention_mask)
        relation_embedding = self.linear_relation(question_embedding)
        relation_embedding = self.dropout(self.bn(self.linear_bn(relation_embedding)))
        scores = self.score(head_embedding, relation_embedding, time_embedding)
        loss = self.loss(scores, answers)
        return None, scores, None, None, loss


class TempoQRTComplEx(QABaseline): # Model class for TempoQR, borrowed from the official repository
    def __init__(self, args):
        super().__init__(args)

        # Transformers
        self.transformer_dim = self.tkg_embedding_dim
        self.nhead = 8
        self.num_layers = 6
        self.transformer_dropout = 0.1
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.nhead,
                                                        dropout=self.transformer_dropout)
        encoder_norm = LayerNorm(self.transformer_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers,
                                                         norm=encoder_norm)
        self.project_sentence_to_transformer_dim = nn.Linear(self.sentence_embedding_dim, self.transformer_dim)
        self.project_entity = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)

        # Position embedding for transformers
        self.max_seq_length = 100
        self.position_embedding = nn.Embedding(self.max_seq_length, self.tkg_embedding_dim)
        self.layer_norm = nn.LayerNorm(self.transformer_dim)

        self.linearT = nn.Linear(768, self.tkg_embedding_dim)  # to project question embedding

    def invert_binary_tensor(self, tensor, device):
        ones_tensor = torch.ones(tensor.shape, dtype=torch.float32).cuda(device)
        inverted = ones_tensor - tensor
        return inverted

    def score_entity(self, head_embedding, relation_embedding, time_embedding): # Scoring function
        lhs = head_embedding[:, :self.rank], head_embedding[:, self.rank:]
        rel = relation_embedding[:, :self.rank], relation_embedding[:, self.rank:]
        time = time_embedding[:, :self.rank], time_embedding[:, self.rank:]
        right = self.tcomplex_model.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
        )

    def forward(self, a, device):
        # Tokenized questions, where entities are masked from the sentence to have TKG embeddings
        question_tokenized = a[0].cuda(device)
        question_attention_mask = a[1].cuda(device)
        entities_times_padded = a[2].cuda(device)
        entity_mask_padded = a[3].cuda(device)

        # Annotated entities/timestamps
        heads = a[4].cuda(device)
        times = a[6].cuda(device)

        answers = a[-2].cuda(device)

        # TKG embeddings
        head_embedding = self.tcomplex_embedding(heads)
        time_embedding = self.tcomplex_embedding(times)

        # Entity embeddings to replace in sentence
        entity_time_embedding = self.tcomplex_embedding(entities_times_padded)

        # Context-aware step
        outputs = self.lm_model(question_tokenized, attention_mask=question_attention_mask)
        last_hidden_states = outputs.last_hidden_state

        # Entity-aware step
        question_embedding = self.project_sentence_to_transformer_dim(last_hidden_states)
        entity_mask = entity_mask_padded.unsqueeze(-1).expand(question_embedding.shape)
        masked_question_embedding = question_embedding * entity_mask  # set entity positions 0
        entity_time_embedding_projected = self.project_entity(entity_time_embedding)

        # Transformer information fusion layer
        masked_entity_time_embedding = entity_time_embedding_projected * self.invert_binary_tensor(entity_mask, device)
        combined_embed = masked_question_embedding + masked_entity_time_embedding

        # Also need to add position embedding
        sequence_length = combined_embed.shape[1]
        v = np.arange(0, sequence_length, dtype=np.long)
        indices_for_position_embedding = torch.from_numpy(v).cuda(device)
        position_embedding = self.position_embedding(indices_for_position_embedding)
        position_embedding = position_embedding.unsqueeze(0).expand(combined_embed.shape)
        combined_embed = combined_embed + position_embedding
        combined_embed = self.layer_norm(combined_embed)
        combined_embed = torch.transpose(combined_embed, 0, 1)

        # Transformers
        mask2 = ~(question_attention_mask.bool()).cuda(device)
        output = self.transformer_encoder(combined_embed, src_key_padding_mask=mask2)
        relation_embedding = output[0]  # self.linear(output[0]) #cls token embedding
        relation_embedding = self.dropout(self.bn(self.linear_bn(relation_embedding)))

        scores = self.score_entity(head_embedding, relation_embedding, time_embedding)
        loss = self.loss(scores, answers)
        return None, scores, None, None, loss

