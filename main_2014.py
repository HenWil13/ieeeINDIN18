import sys
sys.path.append('/Users/henningwilms/GitEL/WorkingArea-Henning/tf_models')
# from data_set import DataSet
from data_set import DataPreparation
import seq2seq_ex_vars


num_attribs_feat = ['w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','w11','w12','w13','w14','w15','w16',
                    'w17','w18','w19','w20','w21','w22','w23','w24','w25']
cat_attribs_feat = ['yr', 'mn', 'dy', 'hr']

num_attribs_lab = ['LOAD']

ds_path = '/Users/henningwilms/GitEL/WorkingArea-Henning/INDIN_paper'

base_path = '/Users/henningwilms/GitEL/WorkingArea-Henning/INDIN_paper/load14'

ds = DataPreparation(ds_path=ds_path, num_attribs_feat=num_attribs_feat,
                               cat_attribs_feat=cat_attribs_feat, num_attribs_lab=num_attribs_lab)

ds.create_fit_pipeline_feat()

ds.create_fit_pipeline_lab(scaler="min-max")

ds.reduce_dimensions(explained_variance=0.98, method='PCA', apply=True)

arch_search_params = {"n_layers": [1, 2, 3, 4],
                      "n_neurons": [10, 20, 30, 40],
                      "batch_size": [20, 30, 40],
                      "cell_type": ["lstm", "gru", "lstm-layer-norm"],
                      "learning_rate": [0.001, 0.01, 0.1],
                      "gradient_clip": [1, 3, 5],
                      "input_keep_prob": [0.6, 0.7, 0.8, 0.9],
                      "output_keep_prob": [0.6, 0.7, 0.8, 0.9],
                      "state_keep_prob": [0.9, 1]
                     }

train_search_params = {"learning_rate": [0.001, 0.01, 0.1],
                       "batch_size": [20, 30, 40, 50],
                       "gradient_clip": [1, 3, 5],
                       "input_keep_prob": [0.6, 0.7, 0.8, 0.9],
                       "output_keep_prob": [0.6, 0.7, 0.8, 0.9],
                       "state_keep_prob": [0.8, 0.9, 1]}

search = seq2seq_ex_vars.seq2seqSearch(arch_search_params=arch_search_params, train_search_params=train_search_params,
                                       data=ds, encoder_length=36, decoder_length=12)

search.search(logging=True, base_path=base_path, n_iter=10, dec_n_iter=5, cv=2,
              epochs=1500, max_checks_wo_progress=175)
