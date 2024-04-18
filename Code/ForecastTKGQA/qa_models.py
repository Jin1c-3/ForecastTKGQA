import torch
from torch import nn
from transformers import DistilBertModel
from utils import load_tango, load_tcomplex


class ForecastTKGQA(nn.Module): # Model class for ForecastTKGQA
	def __init__(self, num_question_type, args):
		super().__init__()

		# Load pre-trained language model
		self.sentence_embedding_dim = 768
		self.pretrained_weights = '/data/qing/distilbert-base-uncased'
		self.lm_model = DistilBertModel.from_pretrained(self.pretrained_weights)
		for param in self.lm_model.parameters():
			param.requires_grad = False

		# Load pre-trained TANGO representations
		if 'tango' in args.tkg_model_file:
			self.tango_model = load_tango('models/{dataset_name}/kg_embeddings/{tkg_model_file}'.format(
				dataset_name=args.dataset_name, tkg_model_file=args.tkg_model_file))
			self.tkg_embedding_dim = list(self.tango_model.values())[0].shape[1]
			self.num_entities = list(self.tango_model.values())[0].shape[0]
			self.tango_weights = nn.Parameter(torch.stack(list(self.tango_model.values())))
			self.tango_embedding = nn.Parameter(self.tango_weights.view(-1, self.tkg_embedding_dim))
			self.rank = int(self.tkg_embedding_dim / 2)
			print(f'tkg embed: {self.tkg_embedding_dim}')
			self.tango_embedding.requires_grad = False
			self.tango_weights.requires_grad = False
		else:
			raise ValueError('tkg model type undefined')

		self.linear_relation = nn.Linear(768, self.tkg_embedding_dim)  # To project question representation from 768 to self.tkg_embedding_dim
		self.dropout = torch.nn.Dropout(0.3)
		self.relu = nn.ReLU()

		# # for unify
		# self.use_distinguisher = args.use_distinguisher
		# self.linear_bn_dis = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)
		# self.bn_dis = torch.nn.BatchNorm1d(self.tkg_embedding_dim)
		# self.distinguisher = nn.Linear(self.tkg_embedding_dim, num_question_type)

		# For entity prediction questions, linear layers in Equation (1)
		self.linear_bn_entity = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)
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
		bert_last_hidden_states = self.lm_model(question_tokenized, attention_mask=attention_mask)[0]
		states = bert_last_hidden_states.transpose(1, 0)
		return states[0]

	# def score_distinguisher(self, relation_embedding, types):
	# 	out = self.distinguisher(relation_embedding)
	# 	loss = self.loss(out, types)
	# 	predictions = torch.argmax(out, dim=1)
	# 	mask = predictions == types
	# 	return mask, out, loss

	def score_entity(self, head_embedding, relation_embedding, times, answers):
		# Scoring function of entity prediction questions (Equation (1))
		head_embedding = self.ep_linear(head_embedding)
		lhs = head_embedding[:, :self.rank], head_embedding[:, self.rank:]
		rel = relation_embedding[:, :self.rank], relation_embedding[:, self.rank:]

		right = self.tango_weights[times]
		right = self.ep_linear(right)
		right = right[:, :, :self.rank], right[:, :, self.rank:]
		re_score = (lhs[0] * rel[0] - lhs[1] * rel[1]).unsqueeze(1)
		im_score = (lhs[0] * rel[1] + lhs[1] * rel[0]).unsqueeze(1)
		score = re_score * right[0] + im_score * right[1]
		score = score.sum(dim=2)
		return score

	def scores_yes_no(self, head_embedding, tail_embedding, relation_embedding, choice_embedding, answers):
		# Scoring function of fact reasoning questions (Equation (2))

		head_embedding = torch.repeat_interleave(head_embedding, 2, dim=0)
		relation_embedding = torch.repeat_interleave(relation_embedding, 2, dim=0)
		tail_embedding = torch.repeat_interleave(tail_embedding, 2, dim=0)
		out = self.score_choice(head_embedding, relation_embedding, tail_embedding, choice_embedding)
		out = out.view(-1, 2)
		return out

	def scores_multiple_choice(self, heads_embed_q, question_embed, tails_embed_q,
							   heads_embed_c, choice_embed, tails_embed_c, answers):
		# Scoring function of fact reasoning questions

		# Concatenate question representation and the entity representations of the entities annotated in the question (Equation (3))
		question_cat = torch.cat((heads_embed_q, question_embed, tails_embed_q), dim=1)
		question_cat = self.mc_cat(question_cat)
		question_cat = torch.repeat_interleave(question_cat, 4, dim=0)

		# Scoring
		out = self.score_choice(heads_embed_c, choice_embed, tails_embed_c, question_cat)
		out = out.view(-1, 4)  # (batch, 4)
		return out

	def score_choice(self, head_emb, question_emb, tail_emb, choice):
		# scoring function of fact reasoning questions

		lhs = head_emb[:, :self.rank], head_emb[:, self.rank:]
		rel = question_emb[:, :self.rank], question_emb[:, self.rank:]
		rhs = tail_emb[:, :self.rank], tail_emb[:, self.rank:]
		choice = choice[:, :self.rank], choice[:, self.rank:]

		re_score = lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] - \
				   lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]
		im_score = lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] + \
				   lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]
		score = re_score * choice[0] + im_score * choice[1]
		score = score.sum(dim=1)
		
		return score

	def forward(self, question_tokenized, question_attention_mask, heads, tails, times, types, answers):
		# Question representation
		question = question_tokenized[:, 0, :]
		question_mask = question_attention_mask[:, 0, :]
		question_embedding = self.getQuestionEmbedding(question, question_mask) # Get question representation from pre-trained LM
		relation_embedding = self.linear_relation(question_embedding) # Map to dimension of TKG representations

		# if self.use_distinguisher:
		# 	relation_emb_dis = self.dropout(self.bn_dis(self.linear_bn_dis(relation_embedding)))
		# 	mask_dis, score_dis, loss_dis = self.distinguisher(relation_emb_dis)
		# 	# scores_dis, loss_dis = self.distinguisher(a, device)
		# 	# predictions = torch.argmax(scores_dis, dim=1)
		# 	# mask_dis = predictions == types
		# 	# heads = heads[mask_dis]
		# 	# tails = tails[mask_dis]
		# 	# times = times[mask_dis]
		# 	# types = types[mask_dis]
		# 	# answers = answers[mask_dis]
		# 	# relation_embedding = relation_embedding[mask_dis]
		# 	# question_tokenized = question_tokenized[mask_dis]
		# 	# question_attention_mask = question_attention_mask[mask_dis]
		# else:
		# 	mask_dis = None
		mask_dis = None

		mask_ep = types < 2
		mask_yn = types == 2
		mask_mc = types == 3
		num_ep, num_yn, num_mc = torch.sum(mask_ep).item(), torch.sum(mask_yn).item(), torch.sum(mask_mc).item()

		heads_question, heads_choice = heads[:, 0], heads[:, 1:]
		tails_question, tails_choice = tails[:, 0], tails[:, 1:]
		times_question, times_choice = times[:, 0], times[:, 1:]

		# TANGO
		heads_embed_q = self.tango_embedding[times_question*self.num_entities + heads_question]
		tails_embed_q = self.tango_embedding[times_question*self.num_entities + tails_question]

		# Entity prediction
		if num_ep > 1:
			# Question representation
			relation_embed_ep = relation_embedding[mask_ep]
			relation_embed_ep = self.dropout(self.bn_entity(self.linear_bn_entity(relation_embed_ep)))
			times_ep, answers_ep = times_question[mask_ep], answers[mask_ep]
			heads_embed_ep = heads_embed_q[mask_ep]
			# Score computation
			scores_ep = self.score_entity(heads_embed_ep, relation_embed_ep, times_ep, answers_ep)
		else:
			scores_ep = None

		# Yes-no
		if num_yn > 1:
			# Question Representation
			relation_embed_yn = relation_embedding[mask_yn]
			relation_embed_yn = self.dropout(self.bn_yn(self.linear_bn_yn(relation_embed_yn)))
			times_yn, answers_yn = times_question[mask_yn], answers[mask_yn]
			heads_embed_yn = self.yn_linear(heads_embed_q[mask_yn])
			tails_embed_yn = self.yn_linear(tails_embed_q[mask_yn])
			# Encode representations of yes and no
			choice_yn = question_tokenized[mask_yn][:, 1:3, :].reshape(-1, question_tokenized.size(-1))
			choice_mask_yn = question_attention_mask[mask_yn][:, 1:3, :].reshape(-1, question_attention_mask.size(-1))
			choice_embed_yn = self.getQuestionEmbedding(choice_yn, choice_mask_yn)
			choice_embed_yn = self.linear_relation(choice_embed_yn)
			choice_embed_yn = self.dropout(self.bn_yn(self.linear_bn_yn(choice_embed_yn)))
			# Score computation
			scores_yn = self.scores_yes_no(
				heads_embed_yn, tails_embed_yn, relation_embed_yn, choice_embed_yn, answers_yn)
		else:
			scores_yn = None

		# Fact reasoning (multiple-choice)
		if num_mc > 1:
			# Question representation
			relation_embed_mc = relation_embedding[mask_mc]
			relation_embed_mc = self.dropout(self.bn_mc(self.linear_bn_mc(relation_embed_mc)))
			# Choice representations
			choice_mc = question_tokenized[mask_mc][:, 1:, :].reshape(-1, question_tokenized.size(-1))
			choice_mask_mc = question_attention_mask[mask_mc][:, 1:, :].reshape(-1, question_attention_mask.size(-1))
			choice_embed_mc = self.getQuestionEmbedding(choice_mc, choice_mask_mc)
			choice_embed_mc = self.linear_relation(choice_embed_mc)
			choice_embed_mc = self.dropout(self.bn_mc(self.linear_bn_mc(choice_embed_mc)))
			# Entity representations of the entities annotated in questions
			heads_embed_mc_q, tails_embed_mc_q = self.mc_linear(heads_embed_q[mask_mc]), self.mc_linear(tails_embed_q[mask_mc])
			# Entity representations of the entities annotated in choices
			heads_choice = torch.flatten(heads_choice[mask_mc])
			tails_choice = torch.flatten(tails_choice[mask_mc])
			times_choice = torch.flatten(times_choice[mask_mc])
			heads_embed_mc_c = self.mc_linear(self.tango_embedding[[times_choice*self.num_entities + heads_choice]])
			tails_embed_mc_c = self.mc_linear(self.tango_embedding[[times_choice*self.num_entities + tails_choice]])
			# Score computation
			answers_mc = answers[mask_mc]
			scores_mc = self.scores_multiple_choice(heads_embed_mc_q, relation_embed_mc, tails_embed_mc_q,
															 heads_embed_mc_c, choice_embed_mc, tails_embed_mc_c,
															 answers_mc)
		else:
			scores_mc = None

		# if self.use_distinguisher:
		# 	types_num = [num_ep, num_yn, num_mc]
		# 	max_num = max(types_num)
		# 	weights = []
		# 	for n in types_num:
		# 		if n > 1:
		# 			weights.append(max_num / n)
		# 		else:
		# 			weights.append(0)
		# 	total_weights = sum(weights)
		# 	try:
		# 		weights = [w / total_weights for w in weights]
		# 		loss = weights[0] * loss_ep + weights[1] * loss_yn + weights[2] * loss_mc
		# 	except ZeroDivisionError:
		# 		loss = torch.tensor(0)
		# else:
		# 	loss = loss_ep + loss_yn + loss_mc

		return mask_dis, scores_ep, scores_yn, scores_mc


class ForecastTKGQA_MHS(nn.Module):
    def __init__(self, args, graph_ori=None):
        super().__init__()
        self.args = args
        self.tango_complex = args.tkg_model_file
        self.repea_num = []
        # self.graph_two = graph_two
        if graph_ori is not None:
            self.graph_origin = graph_ori
        else:
            raise Exception("Graph must be loaded.")
        if 'tango' in args.tkg_model_file:
            self.tango_model = load_tango('models/{dataset_name}/kg_embeddings/{tkg_model_file}'.format(
                dataset_name=args.dataset_name, tkg_model_file=args.tkg_model_file))
            self.tkg_embedding_dim = 200
            self.num_entities = 20575
            self.tkg_embedding = nn.Parameter(torch.stack(list(self.tango_model.values())))
            # self.tango_embedding = nn.Parameter(self.tango_weights.view(-1, self.tkg_embedding_dim))
            self.rank = int(self.tkg_embedding_dim / 2)
            print(f'tkg embed: {self.tkg_embedding_dim}')
        elif 'complex' in args.tkg_model_file:
            self.tcomplex_model = load_tcomplex('models/{dataset_name}/kg_embeddings/{tkg_model_file}'.format(
                dataset_name=args.dataset_name, tkg_model_file=args.tkg_model_file), args.device)
            self.num_entities = self.tcomplex_model.embeddings[0].weight.shape[0]
            self.num_relations = self.tcomplex_model.embeddings[1].weight.shape[0]
            self.num_times = self.tcomplex_model.embeddings[2].weight.shape[0]

            self.tkg_embedding = nn.Parameter(torch.cat(
                [self.tcomplex_model.embeddings[0].weight, self.tcomplex_model.embeddings[1].weight,
                 self.tcomplex_model.embeddings[2].weight]))
            self.tkg_embedding_dim = self.tcomplex_model.embeddings[0].weight.shape[1]
        else:
            raise ValueError('tkg model type undefined')


        # self.tkg_embedding = nn.Parameter(self.tango_weights.view(-1, self.tkg_embedding_dim))
        # self.rank = int(self.tkg_embedding_dim / 2)
        print(f'tkg embed: {self.tkg_embedding_dim}')
        if args.frozen == 1:
            print('Freezing entity embeddings')
            self.tkg_embedding.requires_grad = False

        else:
            print('Unfrozen entity/time embeddings')
            self.tkg_embedding.requires_grad = True

        self.sentence_embedding_dim = 768  # hardwired from roberta?
        self.pretrained_weights = 'distilbert-base-uncased'
        self.roberta_model = DistilBertModel.from_pretrained(self.pretrained_weights)

        if args.lm_frozen == 1:
            print('Freezing LM params')
            for param in self.roberta_model.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen LM params')

        self.linear_question = nn.Linear(768, self.tkg_embedding_dim)
        self.linear_entity = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)
        self.linear_time = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)
        self.linear_relation = nn.Linear(self.tkg_embedding_dim, self.tkg_embedding_dim)

        if 'tango' in args.tkg_model_file:
            self.score_cat_layer_one = nn.Linear(self.tkg_embedding_dim * 4, self.tkg_embedding_dim)
        else:
            self.score_cat_layer_one = nn.Linear(self.tkg_embedding_dim * 5, self.tkg_embedding_dim)
        self.score_cat_layer_two = nn.Linear(self.tkg_embedding_dim, 1)

        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss(reduction='sum')

        # if args.is_multigen:
        #     self.linear_multigen = nn.Linear(self.tkg_embedding_dim * 3, self.tkg_embedding_dim)
        #     self.sigmoid = nn.Sigmoid()

        return

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1, 0)
        return states[0]

    # scoring function from TComplEx
    def score_entity(self, s_embedding, r_embedding, n_embedding, q_embedding):
        rank = int(self.tkg_embedding_dim / 2)

        lhs = s_embedding[:, :rank], s_embedding[:, rank:]
        rel = r_embedding[:, :rank], r_embedding[:, rank:]
        rhs = n_embedding[:, :rank], n_embedding[:, rank:]
        question = q_embedding[:, :rank], q_embedding[:, rank:]

        im_score = lhs[0] * rel[0] * question[0] - lhs[1] * rel[1] * question[0] - \
                   lhs[1] * rel[0] * question[1] - lhs[0] * rel[1] * question[1]
        re_score = lhs[1] * rel[0] * question[0] + lhs[0] * rel[1] * question[0] + \
                   lhs[0] * rel[0] * question[1] - lhs[1] * rel[1] * question[1]

        return torch.sum(im_score * rhs[0] + re_score * rhs[1], 1, keepdim=True)

    def get_index_for_mean(self, tails, device):
        # padding value = 20575 and in score propagation the index of 20575 in score will be filtered out
        return pad_sequence(torch.tensor(tails).cuda(device).split(self.repea_num.tolist()), batch_first=True,
                            padding_value=20575).squeeze()

    def score_propagation(self, s_embedding, r_one_embedding, n_one_embedding, r_two_embedding, n_two_embedding,
                          q_embedding, device, tail_one, tail_two, head_embedding_clone, mask=None, masks_two=None,
                          gamma=0.8):
        """
        path is s -r1-> o1 -r2-> o2
        s_embedding:     embedding of subject        , size=(batch, tkg_embedding_size)
        r_one_embedding: embedding of 1-hop relations, size=(batch, tkg_embedding_size)
        n_one_embedding: embedding of 1-hop neighbors, size=(batch, tkg_embedding_size)
        r_two_embedding: embedding of 2-hop relations, size=(batch, tkg_embedding_size)
        n_one_embedding: embedding of 2-hop neighbors, size=(batch, tkg_embedding_size)
        q_embedding:     embedding of question       , size=(batch, tkg_embedding_size)
        masks_two:       mask out invaild 2 hop info , size=(1, len(tail_two))
        """
        batch_size = len(self.repea_num)
        out_init = torch.ones(batch_size, self.tkg_embedding[0].shape[0]).cuda(device)[:, :20575]

        out_one = torch.zeros(batch_size, self.tkg_embedding[0].shape[0]).cuda(device)
        out_two = torch.zeros(batch_size, self.tkg_embedding[0].shape[0]).cuda(device)

        # score for 1 hop neighbours of subject [s, r1, o1], one time step
        score_one = self.score_entity(s_embedding, r_one_embedding, n_one_embedding, q_embedding)  # 15895 * 1
        # assert 0, self.repea_num
        # self.repea_num [14505, 1390]

        # scatter_mean function asks for a tensor for index argument.
        # subjects have various number of neighbors, if only split tail_one/tail_two,
        # the result is list of lists and cant be converted into a tensor.
        # func get_index_for_mean() returns a tensor that padded with last entity in ent2id
        # score of last entity will always be 0 in our experiment because last entity is never used
        # shape of index: (batch_size, max(self.repea_num))

        index = self.get_index_for_mean(tail_one, device)
        # assert 0, index
        score_one = pad_sequence(score_one.split(self.repea_num.tolist()), batch_first=True).squeeze()  # 2 * 14505
        # assert 0, score_one.shape
        out_one = scatter_mean(score_one, index, dim=-1, out=out_one)[:, :20575]  # 2 * 21081 -> 2 * 20575
        # assert 0, [s_embedding.shape, r_one_embedding.shape, n_one_embedding.shape, r_two_embedding.shape, n_two_embedding.shape, q_embedding.shape]
        # all above 15895 * 200
        # score for 2 hop neighbours of subject [s, r2, o2, r1, o1], one time step
        score_two = self.score_entity(n_one_embedding, r_two_embedding, n_two_embedding, q_embedding)

        if masks_two is not None:
            score_two = score_two * masks_two
        score_two = pad_sequence(score_two.split(self.repea_num.tolist()), batch_first=True).squeeze()

        # assert 0, [score_one.shape, score_two.shape]
        index = self.get_index_for_mean(tail_two, device)
        out_two = scatter_mean(score_two, index, dim=-1, out=out_two)[:, :20575]

        if mask is not None:
            out_one_span = torch.zeros((head_embedding_clone.shape[0], out_one.shape[1])).cuda(device)
            ind = torch.arange(0, head_embedding_clone.shape[0]).cuda(device)
            ind = ind.masked_select(mask).long()
            out_one_span[ind] = out_one

            out_two_span = torch.zeros((head_embedding_clone.shape[0], out_two.shape[1])).cuda(device)
            out_two_span[ind] = out_two

            out_one = gamma * out_init + out_one_span
            final_score = gamma * out_one + out_two_span
        else:
            out_one = gamma * out_init + out_one
            final_score = gamma * out_one + out_two

        # pad final_score to size (batch, number of entities) with value(-inf)
        final_score[final_score == 0] = -99999

        return final_score


    def score_network_tango(self, s_embedding, r_embedding, n_embedding, q_embedding):
        cat_embedding = torch.cat((s_embedding, r_embedding, n_embedding, q_embedding), 1)  # batch_size * 600
        hidden_states = self.relu(self.score_cat_layer_one(cat_embedding))
        score = self.score_cat_layer_two(hidden_states)  # batch_size * 1

        return score

    def score_network_tcomplex(self, s_embedding, r_embedding, n_embedding, q_embedding, t_embedding):
        cat_embedding = torch.cat((s_embedding, r_embedding, n_embedding, t_embedding, q_embedding), 1)  # batch_size * 1000
        hidden_states = self.relu(self.score_cat_layer_one(cat_embedding))
        score = self.score_cat_layer_two(hidden_states)  # batch_size * 1

        return score

    def score_propagation_tcomplex(self, s_embedding, r_one_embedding, n_one_embedding, r_two_embedding, n_two_embedding,
                                   q_embedding, device, tail_one, tail_two, head_embedding_clone, t_embedding,
                                   mask=None, masks_two=None, gamma=0.8):
        """
        path is s -r1-> o1 -r2-> o2
        s_embedding:     embedding of subject        , size=(batch, tkg_embedding_size)
        r_one_embedding: embedding of 1-hop relations, size=(batch, tkg_embedding_size)
        n_one_embedding: embedding of 1-hop neighbors, size=(batch, tkg_embedding_size)
        r_two_embedding: embedding of 2-hop relations, size=(batch, tkg_embedding_size)
        n_one_embedding: embedding of 2-hop neighbors, size=(batch, tkg_embedding_size)
        q_embedding:     embedding of question       , size=(batch, tkg_embedding_size)
        masks_two:       mask out invaild 2 hop info , size=(1, len(tail_two))
        """
        batch_size = len(self.repea_num)
        # out_init = self.score_entity_complex(head_embedding_clone, tango_embedding, question_embedding_clone)
        out_init = torch.zeros(batch_size, self.tkg_embedding.shape[0]).cuda(device)[:, :self.num_entities]
        # out_init[torch.arange(batch_size), heads] = 1.

        # if self.load_static_embedding:
        out_one = torch.zeros(batch_size, self.tkg_embedding.shape[0]).cuda(device)
        out_two = torch.zeros(batch_size, self.tkg_embedding.shape[0]).cuda(device)
        # else:
        # 	out_one = torch.zeros(batch_size, self.tkg_embedding[0].shape[0]).cuda(device)
        # 	out_two = torch.zeros(batch_size, self.tkg_embedding[0].shape[0]).cuda(device)
        # score for 1 hop neighbours of subject [s, r1, o1], one time step
        # if self.args.is_multigen:  # paper Multigen
        # 	score_one = self.score_entity_cat(s_embedding, r_one_embedding, n_one_embedding, q_embedding)
        # else:
        score_one = self.score_network_tcomplex(s_embedding, r_one_embedding, n_one_embedding, q_embedding, t_embedding)
        # scatter_mean function asks for a tensor for index argument.
        # subjects have various number of neighbors, if only split tail_one/tail_two,
        # the result is list of lists and cant be converted into a tensor.
        # func get_index_for_mean() returns a tensor that padded with last entity in ent2id
        # score of last entity will always be 0 in our experiment because last entity is never used
        # shape of index: (batch_size, max(self.repea_num))

        index = self.get_index_for_mean(tail_one, device)

        score_one = pad_sequence(score_one.split(self.repea_num.tolist()), batch_first=True).squeeze()
        out_one = scatter_mean(score_one, index, dim=-1, out=out_one)[:, :self.num_entities]

        # score for 2 hop neighbours of subject [s, r2, o2, r1, o1], one time step
        # if self.args.is_multigen:
        # 	score_two = self.score_entity_cat(s_embedding, r_one_embedding, n_one_embedding, q_embedding)
        # else:
        score_two = self.score_network_tcomplex(n_one_embedding, r_two_embedding, n_two_embedding, q_embedding,
                                                t_embedding)
        if masks_two is not None:
            score_two = score_two * masks_two
        score_two = pad_sequence(score_two.split(self.repea_num.tolist()), batch_first=True).squeeze()

        index = self.get_index_for_mean(tail_two, device)
        out_two = scatter_mean(score_two, index, dim=-1, out=out_two)[:, :self.num_entities]
        if mask is not None:
            out_one_span = torch.zeros((head_embedding_clone.shape[0], out_one.shape[1])).cuda(device)
            ind = torch.arange(0, head_embedding_clone.shape[0]).cuda(device)
            ind = ind.masked_select(mask).long()
            out_one_span[ind] = out_one

            out_two_span = torch.zeros((head_embedding_clone.shape[0], out_two.shape[1])).cuda(device)
            out_two_span[ind] = out_two

            out_one = gamma * out_init + out_one_span
            final_score = gamma * out_one + out_two_span
        else:
            out_one = gamma * out_init + out_one
            final_score = gamma * out_one + out_two

        # pad final_score to size (batch, number of entities) with value(-inf)
        final_score[final_score == 0] = float('-inf')

        return final_score

    def score_propagation_mhs_ctango(self, s_embedding, r_one_embedding, n_one_embedding, r_two_embedding,
                                     n_two_embedding, device, tail_one, tail_two, head_embedding_clone, q_embedding,
                                     mask=None, masks_two=None, gamma=0.8):

        batch_size = len(self.repea_num)
        out_init = torch.ones(batch_size, self.tkg_embedding[0].shape[0]).cuda(device)[:, :20575]

        out_one = torch.zeros(batch_size, self.tkg_embedding[0].shape[0]).cuda(device)
        out_two = torch.zeros(batch_size, self.tkg_embedding[0].shape[0]).cuda(device)
        # else:
        # 	out_one = torch.zeros(batch_size, self.tkg_embedding[0].shape[0]).cuda(device)
        # 	out_two = torch.zeros(batch_size, self.tkg_embedding[0].shape[0]).cuda(device)
        # score for 1 hop neighbours of subject [s, r1, o1], one time step
        # if self.args.is_multigen:  # paper Multigen
        # 	score_one = self.score_entity_cat(s_embedding, r_one_embedding, n_one_embedding, q_embedding)
        # else:
        score_one = self.score_network_tango(s_embedding, r_one_embedding, n_one_embedding, q_embedding)
        # scatter_mean function asks for a tensor for index argument.
        # subjects have various number of neighbors, if only split tail_one/tail_two,
        # the result is list of lists and cant be converted into a tensor.
        # func get_index_for_mean() returns a tensor that padded with last entity in ent2id
        # score of last entity will always be 0 in our experiment because last entity is never used
        # shape of index: (batch_size, max(self.repea_num))

        index = self.get_index_for_mean(tail_one, device)

        # score_one = pad_sequence(score_one.split(self.repea_num.tolist()), batch_first=True).squeeze()
        score_one = pad_sequence(score_one.split(self.repea_num.tolist()), batch_first=True).squeeze()
        out_one = scatter_mean(score_one, index, dim=-1, out=out_one)[:, :20575]

        # score for 2 hop neighbours of subject [s, r2, o2, r1, o1], one time step
        # if self.args.is_multigen:
        # 	score_two = self.score_entity_cat(s_embedding, r_one_embedding, n_one_embedding, q_embedding)
        # else:
        score_two = self.score_network_tango(n_one_embedding, r_two_embedding, n_two_embedding, q_embedding)
        if masks_two is not None:
            score_two = score_two * masks_two
        score_two = pad_sequence(score_two.split(self.repea_num.tolist()), batch_first=True).squeeze()

        # assert 0, [score_one.shape, score_two.shape]
        index = self.get_index_for_mean(tail_two, device)
        out_two = scatter_mean(score_two, index, dim=-1, out=out_two)[:, :20575]

        if mask is not None:
            out_one_span = torch.zeros((head_embedding_clone.shape[0], out_one.shape[1])).cuda(device)
            ind = torch.arange(0, head_embedding_clone.shape[0]).cuda(device)
            ind = ind.masked_select(mask).long()
            out_one_span[ind] = out_one

            out_two_span = torch.zeros((head_embedding_clone.shape[0], out_two.shape[1])).cuda(device)
            out_two_span[ind] = out_two

            out_one = gamma * out_init + out_one_span
            final_score = gamma * out_one + out_two_span
        else:
            out_one = gamma * out_init + out_one
            final_score = gamma * out_one + out_two

        # pad final_score to size (batch, number of entities) with value(-inf)
        final_score[final_score == 0] = -99999

        return final_score

    def forward(self, a, device):
        question_tokenized = a[0][:, 0].cuda(device)
        question_attention_mask = a[1][:, 0].cuda(device)
        heads = a[2][:, 0]
        times = a[4][:, 0]
        answers = a[-2].cuda(device)

        tails_one, tails_two = [], []
        rels_one, rels_two = [], []
        masks_two = []
        self.repea_num = []
        for (t, h) in zip(times, heads):
            # add t info
            t, h = t.item(), h.item()
            # assert 0, (t, h)
            tail_two = np.array(self.graph_origin[t][h])[:, 3]
            tail_one = np.array(self.graph_origin[t][h])[:, 1]
            tails_one.extend(tail_one.tolist())
            tails_two.extend(tail_two.tolist())
            rel_one = np.array(self.graph_origin[t][h])[:, 0]
            rel_two = np.array(self.graph_origin[t][h])[:, 2]
            rels_one.extend(rel_one.tolist())
            rels_two.extend(rel_two.tolist())
            mask_two = np.array(self.graph_origin[t][h])[:, 4]
            masks_two.extend(mask_two.tolist())

            self.repea_num.append(len(tail_two))

        self.repea_num = torch.tensor(self.repea_num).cuda(device)
        masks_two = torch.tensor(masks_two).unsqueeze(1).cuda(device)
        if 'tango' in self.tango_complex:
            head_embedding = self.tkg_embedding[[times, heads]]
            times = np.repeat(times, self.repea_num.cpu(), 0).tolist()
            tail_one_embedding = self.tkg_embedding[[times, tails_one]]
            tail_two_embedding = self.tkg_embedding[[times, tails_two]]
        elif 'complex' in self.tango_complex:
            head_embedding = self.tkg_embedding[heads]
            tail_one_embedding = self.tkg_embedding[tails_one]
            tail_two_embedding = self.tkg_embedding[tails_two]
            time_embedding = self.tkg_embedding[times + self.num_entities + self.num_relations]
            time_embedding = torch.repeat_interleave(time_embedding, self.repea_num, 0)
        else:
            raise ValueError('tkg model type undefined. Cant load embeddings.')

        # repeat: one head in one time step has multiple neighbors. Repeat heads and times to construct tensor for training.
        head_embedding = torch.repeat_interleave(head_embedding, self.repea_num, 0)

        head_embedding = self.linear_entity(head_embedding)
        head_embedding_clone = head_embedding.clone()

        question_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)

        question_embedding = torch.repeat_interleave(question_embedding, self.repea_num, 0)
        question_embedding = self.linear_question(question_embedding)

        tail_one_embedding = self.linear_entity(tail_one_embedding)
        tail_two_embedding = self.linear_entity(tail_two_embedding)

        relation_one_embedding = torch.stack([self.tkg_embedding[0][torch.tensor(rels_one) + 20575]]).squeeze()
        relation_two_embedding = torch.stack([self.tkg_embedding[0][torch.tensor(rels_two) + 20575]]).squeeze()

        relation_one_embedding = self.linear_relation(relation_one_embedding)
        relation_two_embedding = self.linear_relation(relation_two_embedding)
        if 'tango' in self.tango_complex:
            scores = self.score_propagation_mhs_ctango(head_embedding, relation_one_embedding, tail_one_embedding,
                                                       relation_two_embedding, tail_two_embedding, device, tails_one,
                                                       tails_two, head_embedding_clone, question_embedding,
                                                       masks_two=masks_two)
        elif 'complex' in self.tango_complex:
            scores = self.score_propagation_tcomplex(head_embedding, relation_one_embedding, tail_one_embedding,
                                                     relation_two_embedding, tail_two_embedding, question_embedding,
                                                     device, tails_one, tails_two, head_embedding_clone, time_embedding,
                                                     masks_two=None)
        else:
            raise ValueError('tkg model type undefined. Cant calculate scores')

        loss = self.loss(scores, answers)

        return None, scores, None, None, loss


