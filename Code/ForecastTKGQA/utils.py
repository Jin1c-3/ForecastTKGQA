import pickle
import torch
from tcomplex import TComplEx
import io
import os
from datetime import datetime


class CPU_Unpickler(pickle.Unpickler):
	def find_class(self, module, name):
		if module == 'torch.storage' and name == '_load_from_bytes':
			return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
		else:
			return super().find_class(module, name)

def load_tcomplex(tcomplex_model_file, device):
	# Load TComplEx checkpoint pre-trained with ICEWS21
	print('Loading tcomplex model from', tcomplex_model_file)
	x = torch.load(tcomplex_model_file, map_location=torch.device("cpu"))
	num_ent = x['embeddings.0.weight'].shape[0]
	num_rel = x['embeddings.1.weight'].shape[0]
	num_ts = x['embeddings.2.weight'].shape[0]
	print('Number ent, rel, ts from loaded model:', num_ent, num_rel, num_ts)
	sizes = [num_ent, num_rel, num_ent, num_ts]
	rank = x['embeddings.0.weight'].shape[1] // 2  # complex has 2*rank embedding size
	tkbc_model = TComplEx(sizes, rank, no_time_emb=False)
	tkbc_model.load_state_dict(x)
	tkbc_model.cuda(device)
	print('Loaded tkbc model')
	return tkbc_model


def load_complex(complex_file, device):
	# Load ComplEx checkpoint pre-trained with ICEWS21
	print('Loading complex model from', complex_file)
	# Initilize with TComplEx
	tcomplex_file = 'models/ICEWS21/kg_embeddings/tcomplex_05_27_03.ckpt' # Use a TComplEx to provide a template
	tcomplex_params = torch.load(tcomplex_file, map_location=torch.device("cpu"))
	complex_params = torch.load(complex_file, map_location=torch.device("cpu"))
	num_ent = tcomplex_params['embeddings.0.weight'].shape[0]
	num_rel = tcomplex_params['embeddings.1.weight'].shape[0]
	num_ts = tcomplex_params['embeddings.2.weight'].shape[0]
	print('Number ent,rel,ts from loaded model:', num_ent, num_rel, num_ts)
	sizes = [num_ent, num_rel, num_ent, num_ts]
	rank = tcomplex_params['embeddings.0.weight'].shape[1] // 2 # complex has 2*rank embedding size

	# Now put ComplEx params in TcomplEx model
	# Time embeddings will not be used in score computation
	tcomplex_params['embeddings.0.weight'] = complex_params['embeddings.0.weight']
	tcomplex_params['embeddings.1.weight'] = complex_params['embeddings.1.weight']
	torch.nn.init.xavier_uniform_(tcomplex_params['embeddings.2.weight']) # randomize time embeddings

	tkbc_model = TComplEx(sizes, rank, no_time_emb=False)
	tkbc_model.load_state_dict(tcomplex_params)
	tkbc_model.cuda(device)
	print('Loaded complex tkbc model')
	return tkbc_model


def load_tango(tango_file):
	# Load TANGO checkpoint pre-trained with ICEWS21
	print('Loading tango model from', tango_file)
	in_file = open(tango_file, 'rb')
	# time2emb = CPU_Unpickler(in_file).load()
	time2emb = pickle.load(in_file)
	for k, v in time2emb.items():
		time2emb[k] = torch.tensor(v[:20575], requires_grad=False, device='cpu')
	in_file.close()
	return time2emb


def getAllDicts(dataset_name):
	base_path = 'data/{dataset_name}/kg/tkbc_processed_data/{dataset_name}/'.format(
		dataset_name=dataset_name
	)
	dicts = {}
	for f in ['ent_id', 'rel_id', 'ts_id']:
		in_file = open(str(base_path + f), 'rb')
		dicts[f] = pickle.load(in_file)
	rel2id = dicts['rel_id']
	ent2id = dicts['ent_id']
	ts2id = dicts['ts_id']
	file_ent = 'data/{dataset_name}/kg/wd_id2entity_text.txt'.format(
		dataset_name=dataset_name
	)
	file_rel = 'data/{dataset_name}/kg/wd_id2relation_text.txt'.format(
		dataset_name=dataset_name
	)
	type2id = {'1-hop': 0, '2-hop': 1, 'yes_no': 2, 'multiple_choice': 3}

	def readDict(filename):
		f = open(filename, 'r')
		d = {}
		for line in f:
			line = line.strip().split('\t')
			if len(line) == 1:
				line.append('')  # in case literal was blank or whitespace
			d[line[0]] = line[1]
		f.close()
		return d

	e = readDict(file_ent)
	r = readDict(file_rel)
	wd_id_to_text = dict(list(e.items()) + list(r.items()))

	def getReverseDict(d):
		return {value: key for key, value in d.items()}

	id2rel = getReverseDict(rel2id)
	id2ent = getReverseDict(ent2id)
	id2ts = getReverseDict(ts2id)
	id2type = getReverseDict(type2id)

	all_dicts = {'rel2id': rel2id,
				 'id2rel': id2rel,
				 'ent2id': ent2id,
				 'id2ent': id2ent,
				 'ts2id': ts2id,
				 'id2ts': id2ts,
				 'type2id': type2id,
				 'id2type': id2type,
				 'wd_id_to_text': wd_id_to_text
				 }

	return all_dicts


def dataIdsToLiterals(d, all_dicts):
	new_datapoint = []
	id2rel = all_dicts['id2rel']
	id2ent = all_dicts['id2ent']
	id2ts = all_dicts['id2ts']
	wd_id_to_text = all_dicts['wd_id_to_text']
	new_datapoint.append(wd_id_to_text[id2ent[d[0]]])
	new_datapoint.append(wd_id_to_text[id2rel[d[1]]])
	new_datapoint.append(wd_id_to_text[id2ent[d[2]]])
	new_datapoint.append(id2ts[d[3]])
	new_datapoint.append(id2ts[d[4]])
	return new_datapoint

