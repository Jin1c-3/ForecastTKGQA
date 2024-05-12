import torch
from torch import nn
from transformers import DistilBertModel
from utils import load_tango, load_tipnn
import ddputils


class ForecastTKGQA(nn.Module):  # Model class for ForecastTKGQA
    def __init__(self, args):
        super().__init__()

        # Load pre-trained language model
        self.sentence_embedding_dim = 768
        self.pretrained_weights = "/data/qing/distilbert-base-uncased"
        self.lm_model = DistilBertModel.from_pretrained(self.pretrained_weights)
        for param in self.lm_model.parameters():
            param.requires_grad = False

        if "tango" in args.tkg_model_file:
            self.tkg_embedding_type = "tango"
        elif "tipnn" in args.tkg_model_file:
            self.tkg_embedding_type = "tipnn"
        else:
            raise ValueError("tkg model type undefined")

        # Load pre-trained TANGO representations
        if "tango" == self.tkg_embedding_type:
            self.tango_model = load_tango(
                f"models/{args.dataset_name}/kg_embeddings/{args.tkg_model_file}"
            )
            self.tkg_embedding_dim = list(self.tango_model.values())[0].shape[1]
            self.num_entities = list(self.tango_model.values())[0].shape[0]
            self.tango_weights = nn.Parameter(
                torch.stack(list(self.tango_model.values()))
            )
            self.tango_embedding = nn.Parameter(
                self.tango_weights.view(-1, self.tkg_embedding_dim)
            )
            self.rank = int(self.tkg_embedding_dim / 2)
            print(f"tkg embed: {self.tkg_embedding_dim}")
            self.tango_embedding.requires_grad = False
            self.tango_weights.requires_grad = False
        # Load pre-trained TiPNN representations
        elif "tipnn" == self.tkg_embedding_type:
            self.tipnn_model = load_tipnn(
                f"models/{args.dataset_name}/kg_embeddings/{args.tkg_model_file}"
            )
            # Assuming the first key in the dictionary is representative
            first_key = next(iter(self.tipnn_model.keys()))
            first_sub_key = next(iter(self.tipnn_model[first_key].keys()))
            self.tkg_embedding_dim = self.tipnn_model[first_key][first_sub_key].shape[0]
            # self.tipnn_weights = nn.Parameter(
            #     torch.stack([torch.stack(list(self.tipnn_model[t].values())) for t in self.tipnn_model.keys()])
            # )
            # self.tipnn_embedding = nn.Parameter(
            #     self.tipnn_weights.view(-1, self.tipnn_embedding_dim)
            # )
            self.rank = int(self.tkg_embedding_dim / 2)
            print(f"tipnn embed: {self.tkg_embedding_dim}")
            # self.tipnn_embedding.requires_grad = False
            # self.tipnn_weights.requires_grad = False
        else:
            raise ValueError("tkg model type undefined")

        self.linear_relation = nn.Linear(
            self.sentence_embedding_dim, self.tkg_embedding_dim
        )  # To project question representation from 768 to self.tkg_embedding_dim
        self.dropout = torch.nn.Dropout(0.3)
        self.relu = nn.ReLU()

        # For entity prediction questions, linear layers in Equation (1)
        self.linear_bn_entity = nn.Linear(
            self.tkg_embedding_dim, self.tkg_embedding_dim
        )
        self.bn_entity = torch.nn.BatchNorm1d(self.tkg_embedding_dim)
        self.ep_linear = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)

        # For yes-no questions, linear layers in Equation (2)
        self.linear_bn_yn = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)
        self.bn_yn = torch.nn.BatchNorm1d(self.tkg_embedding_dim)
        self.yn_linear = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)

        # For fact reasoning questions, linear layers in Equation (3)
        self.linear_bn_mc = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)
        self.bn_mc = torch.nn.BatchNorm1d(self.tkg_embedding_dim)
        self.mc_cat = nn.Linear(3 * self.tkg_embedding_dim, self.tkg_embedding_dim)
        self.mc_linear = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)
        # self.mc_linear_q = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)

        return

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        bert_last_hidden_states = self.lm_model(
            question_tokenized, attention_mask=attention_mask
        )[0]
        states = bert_last_hidden_states.transpose(1, 0)
        return states[0]

    def score_entity(self, head_embedding, relation_embedding, times):
        # Scoring function of entity prediction questions (Equation (1))
        head_embedding = self.ep_linear(head_embedding)
        lhs = head_embedding[:, : self.rank], head_embedding[:, self.rank :]
        rel = relation_embedding[:, : self.rank], relation_embedding[:, self.rank :]

        if "tango" == self.tkg_embedding_type:
            right = self.tango_weights[times]
        elif "tipnn" == self.tkg_embedding_type:
            right = self.tipnn_model[times].values()

        right = self.ep_linear(right)
        right = right[:, :, : self.rank], right[:, :, self.rank :]
        re_score = (lhs[0] * rel[0] - lhs[1] * rel[1]).unsqueeze(1)
        im_score = (lhs[0] * rel[1] + lhs[1] * rel[0]).unsqueeze(1)
        score = re_score * right[0] + im_score * right[1]
        score = score.sum(dim=2)
        return score

    def scores_yes_no(
        self,
        head_embedding,
        tail_embedding,
        relation_embedding,
        choice_embedding,
    ):
        # Scoring function of fact reasoning questions (Equation (2))
        head_embedding = torch.repeat_interleave(head_embedding, 2, dim=0)
        relation_embedding = torch.repeat_interleave(relation_embedding, 2, dim=0)
        tail_embedding = torch.repeat_interleave(tail_embedding, 2, dim=0)
        out = self.score_choice(
            head_embedding, relation_embedding, tail_embedding, choice_embedding
        )
        out = out.view(-1, 2)
        return out

    def scores_multiple_choice(
        self,
        heads_embed_q,
        question_embed,
        tails_embed_q,
        heads_embed_c,
        choice_embed,
        tails_embed_c,
    ):
        # Scoring function of fact reasoning questions

        # Concatenate question representation and the entity representations of the entities annotated in the question (Equation (3))
        question_cat = torch.cat((heads_embed_q, question_embed, tails_embed_q), dim=1)
        question_cat = self.mc_cat(question_cat)
        question_cat = torch.repeat_interleave(question_cat, 4, dim=0)

        # Scoring
        out = self.score_choice(
            heads_embed_c, choice_embed, tails_embed_c, question_cat
        )
        out = out.view(-1, 4)  # (batch, 4)
        return out

    def score_choice(self, head_emb, question_emb, tail_emb, choice):
        # scoring function of fact reasoning questions

        lhs = head_emb[:, : self.rank], head_emb[:, self.rank :]
        rel = question_emb[:, : self.rank], question_emb[:, self.rank :]
        rhs = tail_emb[:, : self.rank], tail_emb[:, self.rank :]
        choice = choice[:, : self.rank], choice[:, self.rank :]

        re_score = (
            lhs[0] * rel[0] * rhs[0]
            - lhs[1] * rel[1] * rhs[0]
            - lhs[1] * rel[0] * rhs[1]
            + lhs[0] * rel[1] * rhs[1]
        )
        im_score = (
            lhs[1] * rel[0] * rhs[0]
            - lhs[0] * rel[1] * rhs[0]
            + lhs[0] * rel[0] * rhs[1]
            - lhs[1] * rel[1] * rhs[1]
        )
        score = re_score * choice[0] + im_score * choice[1]
        score = score.sum(dim=1)

        return score

    def forward(
        self,
        question_tokenized,
        question_attention_mask,
        heads,
        tails,
        times,
        types,
        answers,
        relations,
    ):
        # Question representation
        question = question_tokenized[:, 0, :]
        question_mask = question_attention_mask[:, 0, :]
        question_embedding = self.getQuestionEmbedding(
            question, question_mask
        )  # Get question representation from pre-trained LM
        relation_embedding = self.linear_relation(
            question_embedding
        )  # Map to dimension of TKG representations
        mask_dis = None # 这玩意儿啥用也没有

        mask_ep = types < 2
        mask_yn = types == 2
        mask_mc = types == 3
        num_ep, num_yn, num_mc = (
            torch.sum(mask_ep).item(),
            torch.sum(mask_yn).item(),
            torch.sum(mask_mc).item(),
        )

        heads_question, heads_choice = heads[:, 0], heads[:, 1:]
        tails_question, tails_choice = tails[:, 0], tails[:, 1:]
        times_question, times_choice = times[:, 0], times[:, 1:]
        relations_question, relations_choice = relations[:, 0], relations[:, 1:]

        if self.tkg_embedding_type == "tango":
            # TANGO
            heads_embed_q = self.tango_embedding[
                times_question * self.num_entities + heads_question
            ]
            tails_embed_q = self.tango_embedding[
                times_question * self.num_entities + tails_question
            ]
        elif self.tkg_embedding_type == "tipnn":
            heads_embed_q = []
            tails_embed_q = []

            for i in range(len(heads_question)):
                head = heads_question[i].item()
                tail = tails_question[i].item()
                time = times_question[i].item()
                rel = relations_question[i]
                if len(rel) > 1:
                    if (str(head), str(rel[0])) not in self.tipnn_model[str(time)].keys():
                        head_embed = self.tipnn_model[str(time)][(str(head), str(rel[1]))]
                    else:
                        head_embed = self.tipnn_model[str(time)][(str(head), str(rel[0]))]
                    if (tail, str(rel[0])) not in self.tipnn_model[str(time)].keys():
                        tail_embed = self.tipnn_model[str(time)][(tail, str(rel[1]))]
                    else:
                        tail_embed = self.tipnn_model[str(time)][(tail, str(rel[0]))]
                else:
                    head_embed = self.tipnn_model[str(time)][(str(head), str(rel[0]))]
                    tail_embed = self.tipnn_model[str(time)][(tail, str(rel[0]))]
                heads_embed_q.append(head_embed)
                tails_embed_q.append(tail_embed)

            heads_embed_q = torch.stack(heads_embed_q)
            tails_embed_q = torch.stack(tails_embed_q)

        # Entity prediction
        if num_ep > 1:
            # Question representation
            relation_embed_ep = relation_embedding[mask_ep]
            relation_embed_ep = self.dropout(
                self.bn_entity(self.linear_bn_entity(relation_embed_ep))
            )
            times_ep = times_question[mask_ep]
            heads_embed_ep = heads_embed_q[mask_ep]
            # Score computation
            scores_ep = self.score_entity(heads_embed_ep, relation_embed_ep, times_ep)
        else:
            scores_ep = None

        # Yes-no
        if num_yn > 1:
            # Question Representation
            relation_embed_yn = relation_embedding[mask_yn]
            relation_embed_yn = self.dropout(
                self.bn_yn(self.linear_bn_yn(relation_embed_yn))
            )
            heads_embed_yn = self.yn_linear(heads_embed_q[mask_yn])
            tails_embed_yn = self.yn_linear(tails_embed_q[mask_yn])
            # Encode representations of yes and no
            choice_yn = question_tokenized[mask_yn][:, 1:3, :].reshape(
                -1, question_tokenized.size(-1)
            )
            choice_mask_yn = question_attention_mask[mask_yn][:, 1:3, :].reshape(
                -1, question_attention_mask.size(-1)
            )
            choice_embed_yn = self.getQuestionEmbedding(choice_yn, choice_mask_yn)
            choice_embed_yn = self.linear_relation(choice_embed_yn)
            choice_embed_yn = self.dropout(
                self.bn_yn(self.linear_bn_yn(choice_embed_yn))
            )
            # Score computation
            scores_yn = self.scores_yes_no(
                heads_embed_yn,
                tails_embed_yn,
                relation_embed_yn,
                choice_embed_yn,
            )
        else:
            scores_yn = None

        # Fact reasoning (multiple-choice)
        if num_mc > 1:
            # Question representation
            relation_embed_mc = relation_embedding[mask_mc]
            relation_embed_mc = self.dropout(
                self.bn_mc(self.linear_bn_mc(relation_embed_mc))
            )
            # Choice representations
            choice_mc = question_tokenized[mask_mc][:, 1:, :].reshape(
                -1, question_tokenized.size(-1)
            )
            choice_mask_mc = question_attention_mask[mask_mc][:, 1:, :].reshape(
                -1, question_attention_mask.size(-1)
            )
            choice_embed_mc = self.getQuestionEmbedding(choice_mc, choice_mask_mc)
            choice_embed_mc = self.linear_relation(choice_embed_mc)
            choice_embed_mc = self.dropout(
                self.bn_mc(self.linear_bn_mc(choice_embed_mc))
            )
            # Entity representations of the entities annotated in questions
            heads_embed_mc_q, tails_embed_mc_q = self.mc_linear(
                heads_embed_q[mask_mc]
            ), self.mc_linear(tails_embed_q[mask_mc])
            # Entity representations of the entities annotated in choices
            heads_choice = torch.flatten(heads_choice[mask_mc])
            tails_choice = torch.flatten(tails_choice[mask_mc])
            times_choice = torch.flatten(times_choice[mask_mc])
            # TODO checkout tango_embedding havn't been updated
            heads_embed_mc_c = self.mc_linear(
                self.tango_embedding[[times_choice * self.num_entities + heads_choice]]
            )
            tails_embed_mc_c = self.mc_linear(
                self.tango_embedding[[times_choice * self.num_entities + tails_choice]]
            )
            # Score computation
            scores_mc = self.scores_multiple_choice(
                heads_embed_mc_q,
                relation_embed_mc,
                tails_embed_mc_q,
                heads_embed_mc_c,
                choice_embed_mc,
                tails_embed_mc_c,
            )
        else:
            scores_mc = None

        return mask_dis, scores_ep, scores_yn, scores_mc
