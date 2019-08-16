import torch

model = torch.load('transE/model/WN11/l_0.001_es_0_L_1_em_300_nb_100_n_1000_m_1.0_f_1_mo_0.9_s_0_op_1_lo_0_TransE.ckpt')
ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy()

f = open('transE/model/WN11/ent_embeddings.txt', 'a+', encoding='utf-8')
for line in ent_embeddings:
    string = ''
    for item in line.tolist():
        string += str(item) + ' '

    f.write(string + '\n')
f.close()
