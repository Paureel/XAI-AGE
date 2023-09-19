import sys
from os.path import join, dirname, realpath
#current_dir = dirname(realpath(__file__))
current_dir = "train"
from preprocessing import pre
import subprocess
sys.path.insert(0, dirname(current_dir))
import os
import imp
import logging
import random
import timeit
import datetime
import numpy as np
import tensorflow as tf
from utils.logs import set_logging, DebugFolder
import yaml
from pipeline.train_validate import TrainValidatePipeline
from pipeline.one_split import OneSplitPipeline
from pipeline.crossvalidation_pipeline import CrossvalidationPipeline
from pipeline.LeaveOneOut_pipeline import LeaveOneOutPipeline
import networkx as nx
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
#from data.data_access import Data
from data.prostate_paper.data_reader import ProstateDataPaper
from copy import deepcopy
import logging

from sklearn import svm, linear_model
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import Ridge, ElasticNet, Lasso, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from analysis.figure_3.data_extraction_utils import get_node_importance, get_link_weights_df_, \
    get_data, get_degrees, adjust_coef_with_graph_degree, get_pathway_names
from model.coef_weights_utils import get_deep_explain_scores
from os import makedirs
from os.path import dirname, realpath, exists
import pickle
from model.model_utils import get_coef_importance
from model import nn
from analysis.figure_3.data_extraction_utils import get_node_importance, get_link_weights_df_, \
    get_data, get_degrees, adjust_coef_with_graph_degree, get_pathway_names
from utils.loading_utils import DataModelLoader
def transform_prediction(pred_list, age_adult = 20):
    return [(1+age_adult)*np.exp(pred)-1 if pred < 0 else (1+age_adult)*pred+age_adult for pred in pred_list]

def extract_features( x_train, x_test):
        if self.features_params == {}:
            return x_train, x_test
        logging.info('feature extraction ....')

        proc = feature_extraction.get_processor(self.features_params)
        if proc:
            x_train = proc.transform(x_train)
            x_test = proc.transform(x_test)

            if scipy.sparse.issparse(x_train):
                x_train = x_train.todense()
                x_test = x_test.todense()
        return x_train, x_test
def predict( model, x_test, y_test):
        logging.info('predicitng ...')
        y_pred_test = model.predict(x_test)
        if hasattr(model, 'predict_proba'):
            y_pred_test_scores = model.predict_proba(x_test)[:, 1]
        else:
            y_pred_test_scores = y_pred_test

        print 'y_pred_test', y_pred_test.shape, y_pred_test_scores.shape
        return y_pred_test, y_pred_test_scores
    
def preprocess( x_train, x_test):
        logging.info('preprocessing....')
        proc = pre.get_processor(self.pre_params)
        if proc:
            proc.fit(x_train)
            x_train = proc.transform(x_train)
            x_test = proc.transform(x_test)

            if scipy.sparse.issparse(x_train):
                x_train = x_train.todense()
                x_test = x_test.todense()
        return x_train, x_test
    
class Data():
    def __init__(self, id, type, params, test_size=0.3, stratify=True):

        self.test_size = test_size
        self.stratify = stratify
        self.data_type = type
        self.data_params = params
        if self.data_type == 'prostate_paper':
            self.data_reader = ProstateDataPaper(**params)
        else:
            logging.error('unsupported data type')
            raise ValueError('unsupported data type')

    def get_train_validate_test(self):
        return self.data_reader.get_train_validate_test()

    def get_train_test(self):
        x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = self.data_reader.get_train_validate_test()
        # combine training and validation datasets
        x_train = np.concatenate((x_train, x_validate))
        y_train = np.concatenate((y_train, y_validate))
        info_train = list(info_train) + list(info_validate)
        return x_train, x_test, y_train, y_test, info_train, info_test, columns

    def get_data(self):
        x = self.data_reader.x
        y = self.data_reader.y
        info = self.data_reader.info
        columns = self.data_reader.columns
        return x, y, info, columns

    def get_relevant_features(self):
        if hasattr(self.data_reader, 'relevant_features'):
            return self.data_reader.get_relevant_features()
        else:
            return None
        
def get_train_validate_test(self):
        info = self.info
        x = self.x
        y = self.y
        columns = self.columns
        splits_path = join(PROSTATE_DATA_PATH, 'splits')

        training_file = 'training_set_{}.csv'.format(self.training_split)
        training_set = pd.read_csv(join(splits_path, training_file))

        validation_set = pd.read_csv(join(splits_path, 'validation_set.csv'))
        testing_set = pd.read_csv(join(splits_path, 'test_set.csv'))

        info_train = list(set(info).intersection(training_set.id))
        info_validate = list(set(info).intersection(validation_set.id))
        info_test = list(set(info).intersection(testing_set.id))

        ind_train = info.isin(info_train)
        ind_validate = info.isin(info_validate)
        ind_test = info.isin(info_test)

        x_train = x[ind_train]
        x_test = x[ind_test]
        x_validate = x[ind_validate]

        y_train = y[ind_train]
        y_test = y[ind_test]
        y_validate = y[ind_validate]

        info_train = info[ind_train]
        info_test = info[ind_test]
        info_validate = info[ind_validate]

        return x_train, x_validate, x_test, y_train, y_validate, y_test, info_train.copy(), info_validate, info_test.copy(), columns
class DataModelLoader():
    def __init__(self, params_file):
        self.dir_path = os.path.dirname(os.path.realpath(params_file))
        model_parmas, data_parmas = self.load_parmas(params_file)
        data_reader = Data(**data_parmas)
        self.model = None
        x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data_reader.get_train_validate_test()

        self.x_train = x_train
        self.x_test = np.concatenate([x_validate_, x_test_], axis=0)

        self.y_train = y_train
        self.y_test = np.concatenate([y_validate_, y_test_], axis=0)

        self.info_train = info_train
        self.info_test = list(info_validate_) + list(info_test_)
        self.columns = cols

    def get_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test, self.info_train, self.info_test, self.columns

    def get_model(self, model_name='P-net_params.yml'):
        # if self.model is None:
        self.model = self.load_model(self.dir_path, model_name)
        return self.model

    def load_model(self, model_dir_, model_name):
        # 1 - load architecture
        params_filename = join(model_dir_, model_name + '_params.yml')
        stream = file(params_filename, 'r')
        params = yaml.load(stream)
        # print params
        # fs_model = model_factory.get_model(params['model_params'][0])
        fs_model = model_factory.get_model(params['model_params'])
        # 2 -compile model and load weights (link weights)
        weights_file = join(model_dir_, 'fs/{}.h5'.format(model_name))
        model = fs_model.load_model(weights_file)
        return model

    def load_parmas(self, params_filename):
        stream = file(params_filename, 'r')
        params = yaml.load(stream, Loader=yaml.UnsafeLoader)
        model_parmas = params['model_params']
        data_parmas = params['data_params']
        return model_parmas, data_parmas



# get a model object from a dictionary
# the params is in the format of {'type': 'model_type', 'params' {}}
# an example is params = {'type': 'svr', 'parmas': {'C': 0.025} }

def construct_model(model_params_dict):
    model_type = model_params_dict['type']
    p = model_params_dict['params']
    # logging.info ('model type: ', str(model_type))
    # logging.info('model paramters: {}'.format(p))

    if model_type == 'svr':
        model = svm.SVR(max_iter=5000, **p)

    if model_type == 'knn':
        model = KNeighborsClassifier(**p)

    if model_type == 'svc':
        model = svm.SVC(max_iter=5000, **p)

    if model_type == 'linear_svc':
        model = LinearSVC(max_iter=5000, **p)

    if model_type == 'multinomial':
        model = MultinomialNB(**p)

    if model_type == 'nearest_centroid':
        model = NearestCentroid(**p)

    if model_type == 'bernoulli':
        model = BernoulliNB(**p)

    if model_type == 'sgd':
        model = SGDClassifier(**p)

    if model_type == 'gaussian_process':
        model = GaussianProcessClassifier(**p)

    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(**p)

    if model_type == 'random_forest':
        model = RandomForestClassifier(**p)

    if model_type == 'adaboost':
        model = AdaBoostClassifier(**p)

    if model_type == 'svr':
        model = svm.SVR(max_iter=5000, **p)
    # elif model_type == 'dt':
    #     # from sklearn.tree import DecisionTreeClassifier
    #     # model = DecisionTreeClassifier(**p)
    #     model = ModelWrapper(model)
    # elif model_type == 'rf':
    #     # from sklearn.ensemble import RandomForestClassifier
    #     model = RandomForestClassifier(**p)
    #     model = ModelWrapper(model)

    if model_type == 'ridge_classifier':
        model = RidgeClassifier(**p)

    elif model_type == 'ridge':
        model = Ridge(**p)


    elif model_type == 'elastic':
        model = ElasticNet(**p)
    elif model_type == 'lasso':
        model = Lasso(**p)
    elif model_type == 'randomforest':
        model = DecisionTreeRegressor(**p)

    elif model_type == 'extratrees':
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(**p)
        # print model

    elif model_type == 'randomizedLR':
        from sklearn.linear_model import RandomizedLogisticRegression
        model = RandomizedLogisticRegression(**p)

    elif model_type == 'AdaBoostDecisionTree':
        DT_params = params['DT_params']
        model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(**DT_params), **p)
    elif model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(**p)
    elif model_type == 'ranksvm':
        model = RankSVMKernel()
    elif model_type == 'logistic':
        logging.info('model class {}'.format(model_type))
        model = linear_model.LogisticRegression()

    elif model_type == 'nn':
        model = nn.Model(**p)

    return model


def get_model(params):
    if type(params['params']) == dict:
        model = construct_model(params)
    else:
        model = params['params']
    return model
def train_predict_crossvalidation(self, model_params, X, y, info, cols, model_name):
        logging.info('model_params: {}'.format(model_params))
        n_splits = self.pipeline_params['params']['n_splits']
        skf = StratifiedKFold(n_splits=n_splits, random_state=123, shuffle=True)
        i = 0
        scores = []
        model_list = []
        for train_index, test_index in skf.split(X, y.ravel().astype(int)):
            model = get_model(model_params)
            logging.info('fold # ----------------%d---------' % i)
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            info_train = pd.DataFrame(index=info[train_index])
            info_test = pd.DataFrame(index=info[test_index])
            x_train, x_test = self.preprocess(x_train, x_test)
            # feature extraction
            logging.info('feature extraction....')
            x_train, x_test = self.extract_features(x_train, x_test)

            model = model.fit(x_train, y_train)

            y_pred_test, y_pred_test_scores = self.predict(model, x_test, y_test)
            score_test = self.evaluate(y_test, y_pred_test, y_pred_test_scores)
            logging.info('model {} -- Test score {}'.format(model_name, score_test))
            self.save_prediction(info_test, y_pred_test, y_pred_test_scores, y_test, i, model_name)

            if hasattr(model, 'save_model'):
                logging.info('saving coef')
                save_model(model, model_name + '_' + str(i), self.directory)

            if self.save_train:
                logging.info('predicting training ...')
                y_pred_train, y_pred_train_scores = self.predict(model, x_train, y_train)
                self.save_prediction(info_train, y_pred_train, y_pred_train_scores, y_train, i, model_name,
                                     training=True)

            scores.append(score_test)

            fs_parmas = deepcopy(model_params)
            if hasattr(fs_parmas, 'id'):
                fs_parmas['id'] = fs_parmas['id'] + '_fold_' + str(i)
            else:
                fs_parmas['id'] = fs_parmas['type'] + '_fold_' + str(i)

            model_list.append((model, fs_parmas))
            i += 1
        self.save_coef(model_list, cols)
        logging.info(scores)
        return scores
def save_gradient_importance(node_weights_, node_weights_samples_dfs, info, samplename):
    for i, k in enumerate(layers[:-1]):
        n = node_weights_[k]
        filename = join(saving_dir, 'gradient_importance_{}_'+samplename+'.csv'.format(i))
        n.to_csv(filename)

    for i, k in enumerate(layers[:-1]):
        n = node_weights_samples_dfs[k]
        if i > 0:
            n['ind'] = info
            n = n.set_index('ind')
            filename = join(saving_dir, 'gradient_importance_detailed_{}_'+samplename+'.csv'.format(i))
            n.to_csv(filename)


def save_link_weights(link_weights_df, layers):
    for i, l in enumerate(layers):
        link = link_weights_df[l]
        filename = join(saving_dir, 'link_weights_{}.csv'.format(i))
        link.to_csv(filename)


def save_activation(layer_outs_dict, feature_names, info):
    for l_name, l_outut in sorted(layer_outs_dict.iteritems()):
        if l_name.startswith('h'):
            print(l_name, l_outut.shape)
            l = int(l_name[1:])
            features = feature_names[l_name]
            layer_output_df = pd.DataFrame(l_outut, index=info, columns=features)
            layer_output_df = layer_output_df.round(decimals=3)
            filename = join(saving_dir, 'activation_{}.csv'.format(l + 1))
            layer_output_df.to_csv(filename)


def save_graph_stats(degrees, fan_outs, fan_ins, layers):
    i = 1

    df = pd.concat([degrees[0], fan_outs[0]], axis=1)
    df.columns = ['degree', 'fan_out']
    df['fan_in'] = 0
    filename = join(saving_dir, 'graph_stats_{}.csv'.format(i))
    df.to_csv(filename)

    for i, (d, fin, fout) in enumerate(zip(degrees[1:], fan_ins, fan_outs[1:])):
        df = pd.concat([d, fin, fout], axis=1)
        df.columns = ['degree', 'fan_in', 'fan_out']
        print(df.head())
        filename = join(saving_dir, 'graph_stats_{}.csv'.format(i + 2))
        df.to_csv(filename)
def save_gradient_importance(node_weights_, node_weights_samples_dfs, info, samplename):
    for i, k in enumerate(layers[:-1]):
        n = node_weights_[k]
        filename = join(saving_dir, 'gradient_importance_'+str(i)+'.csv'.format(i))
        print(filename)
        n.to_csv(filename)

    for i, k in enumerate(layers[:-1]):
        n = node_weights_samples_dfs[k]
        if i > 0:
            n['ind'] = info
            n = n.set_index('ind')
            filename = join(saving_dir, 'gradient_importance_detailed_'+str(i)+'.csv'.format(i))
            n.to_csv(filename)

'''
first layer
'''

def get_first_layer_df(nlargest):
    features_weights = pd.read_csv(join(module_path, './extracted/gradient_importance_0.csv'), index_col=[0, 1])
    features_weights['layer'] = 0
    nodes_per_layer0 = features_weights[['layer']]
    features_weights = features_weights[['coef']]

    all_weights = pd.read_csv(join(module_path, './extracted/node_importance_graph_adjusted.csv'), index_col=0)
    genes_weights = all_weights[all_weights.layer == 1]
    nodes_per_layer1 = genes_weights[['layer']]
    genes_weights = genes_weights[['coef_combined']]
    # genes_weights = genes_weights[[col_name]]
    genes_weights.columns = ['coef']

    nodes_per_layer_df = pd.concat([nodes_per_layer0, nodes_per_layer1])
    print(genes_weights.head())
    print('genes_weights', genes_weights)

    node_weights = [features_weights, genes_weights]

    df = get_first_layer(node_weights, number_of_best_nodes=nlargest[0], col_name='coef', include_others=True)
    # print df.head()
    # df.to_csv(join(saving_dir,'first_layer.csv'))
    first_layer_df = df[['source', 'target', 'value', 'layer']]
    return first_layer_df


'''
first layer 
df.head()
source	target	value	direction	layer

high_nodes_df
index coef	coef_combined	coef_combined2	coef_combined_zscore	coef_graph	layer	node_id

important_node_connections_df
source	target	layer	value	value_abs	child_sum_target	child_sum_source	value_normalized_by_target	value_normalized_by_source	target_importance	source_importance	A	B	value_final	value_old	source_fan_out	source_fan_out_error	target_fan_in	target_fan_in_error	value_final_corrected
'''


def encode_nodes(df):
    source = df['source']
    target = df['target']
    all_node_labels = list(np.unique(np.concatenate([source, target])))
    n_nodes = len(all_node_labels)
    node_code = range(n_nodes)
    df_encoded = df.replace(all_node_labels, node_code)
    return df_encoded, all_node_labels, node_code


def get_nlargeest_ind(S):
    # ind_source = (S - S.median()).abs() > 3. * S.std()
    ind_source = (S - S.median()).abs() > 2. * S.std()
    ret = min([10, int(sum(ind_source))])
    return ret


def get_pathway_names(all_node_ids):
    pathways_names = get_reactome_pathway_names()
    all_node_labels = pd.Series(all_node_ids).replace(list(pathways_names['id']), list(pathways_names['name']))
    return all_node_labels


def get_nodes_per_layer_filtered(nodes_per_layer_df, all_node_ids, all_node_labels):
    all_node_ids_df = pd.DataFrame(index=all_node_ids)
    nodes_per_layer_filtered_df = nodes_per_layer_df.join(all_node_ids_df, how='right')
    nodes_per_layer_filtered_df = nodes_per_layer_filtered_df.fillna(0)
    nodes_per_layer_filtered_df = nodes_per_layer_filtered_df.groupby(nodes_per_layer_filtered_df.index).min()
    mapping_dict = dict(zip(all_node_ids, all_node_labels))
    nodes_per_layer_filtered_df.index = nodes_per_layer_filtered_df.index.map(lambda x: mapping_dict[x])
    mapping_dict = {y: x for x, y in all_node_labels.to_dict().iteritems()}
    nodes_per_layer_filtered_df.index = nodes_per_layer_filtered_df.index.map(lambda x: mapping_dict[x])
    return nodes_per_layer_filtered_df


# features_weights = pd.read_csv(join(module_path,'./extracted/gradient_importance_0.csv'), index_col =[0,1])
# features_weights = features_weights.reset_index()
# features_weights.columns= ['target', 'source', 'value']
# features_weights['layer'] = 0
# features_weights.head()

def get_links_with_first_layer():
    '''
        :return: all_links_df: dataframe with all the connections in the model (with first layer)
        '''
    links = []

    link = pd.read_csv(join(module_path, './extracted/gradient_importance_0.csv'), index_col=[0, 1])
    link = link.reset_index()
    link.columns = ['target', 'source', 'value']
    link['layer'] = 0
    links.append(link)
    for l in range(1, 7):
        link = pd.read_csv(join(module_path, './extracted/link_weights_{}.csv'.format(l)), index_col=0)
        link.index.name = 'source'
        link = link.reset_index()
        link_unpivoted = pd.melt(link, id_vars=['source'], var_name='target', value_name='value')
        link_unpivoted['layer'] = l
        link_unpivoted = link_unpivoted[link_unpivoted.value != 0.]
        link_unpivoted = link_unpivoted.drop_duplicates(subset=['source', 'target'])
        links.append(link_unpivoted)
    all_links_df = pd.concat(links, axis=0, sort=True)

    return all_links_df


def get_links():
    '''
    :return: all_links_df: dataframe with all the connections in the model (except first layer)
    '''
    links = []
    for l in range(1, 7):
        link = pd.read_csv(join(module_path, './extracted/link_weights_{}.csv'.format(l)), index_col=0)
        link.index.name = 'source'
        link = link.reset_index()
        link_unpivoted = pd.melt(link, id_vars=['source'], var_name='target', value_name='value')
        link_unpivoted['layer'] = l
        link_unpivoted = link_unpivoted[link_unpivoted.value != 0.]
        link_unpivoted = link_unpivoted.drop_duplicates(subset=['source', 'target'])
        links.append(link_unpivoted)
    all_links_df = pd.concat(links, axis=0)
    return all_links_df


def get_high_nodes(node_importance, nlargest, column):
    '''
    get n largest nodes in each layer
    :param:  node_importance: dataframe with coef_combined  and layer columns
    :return: list of high node names
    '''
    layers = np.sort(node_importance.layer.unique())
    high_nodes = []
    for i, l in enumerate(layers):
        if type(nlargest) == list:
            n = nlargest[i]
        else:
            n = nlargest
        high = list(node_importance[node_importance.layer == l].nlargest(n, columns=column).index)
        high_nodes.extend(high)
    return high_nodes


def filter_nodes(node_importance, high_nodes, add_others=True):
    high_nodes_df = node_importance[node_importance.index.isin(high_nodes)].copy()
    # add others:

    if add_others:
        layers = list(node_importance.layer.unique())
        names = ['others{}'.format(l) for l in layers]
        names = names + ['root']
        layers.append(np.max(layers) + 1)
        print(layers)
        print(names)
        data = {'index': names, 'layer': layers}
        df = pd.DataFrame.from_dict(data)
        df = df.set_index('index')
        high_nodes_df = high_nodes_df.append(df)
    return high_nodes_df


def filter_connections(df, high_nodes, add_unk=False):
    """
    :param df: dataframe [ source, target, layer]
    :param high_nodes: list of high node ids
    :param add_unk: boolean , add other connections or not
    :return: ret, dataframe [ source, target, layer] with filtered connection based on the high_nodes
    """
    high_nodes.append('root')

    def apply_others(row):
        # print row
        if not row['source'] in high_nodes_list:
            row['source'] = 'others' + str(row['layer'])

        if not row['target'] in high_nodes_list:
            row['target'] = 'others' + str(row['layer'] + 1)
        return row

    layers_id = np.sort(df.layer.unique())
    high_nodes_list = high_nodes
    layer_dfs = []
    for i, l in enumerate(layers_id):
        layer_df = df[df.layer == l].copy()
        ind1 = layer_df.source.isin(high_nodes_list)
        ind2 = layer_df.target.isin(high_nodes_list)
        if add_unk:
            layer_df = layer_df[ind1 | ind2]
            layer_df = layer_df.apply(apply_others, axis=1)

        else:
            layer_df = layer_df[ind1 & ind2]

        layer_df = layer_df.groupby(['source', 'target']).agg({'value': 'sum', 'layer': 'min'})

        layer_dfs.append(layer_df)
    ret = pd.concat(layer_dfs)
    return ret


def get_x_y(df_encoded, layers_nodes):
    '''
    :param df_encoded: datafrme with columns (source  target  value) representing the network
    :param layers_nodes: data frame with index (nodes ) and one columns (layer) representing the layer of the node
    :return: x, y positions onf each node
    '''

    # node_id = range(len(layers_nodes))
    # node_weights = pd.DataFrame([node_id, layers_nodes], columns=['node_id', 'node_name'])
    # print node_weights
    # node weight is the max(sum of input edges, sum of output edges)

    def rescale(val, in_min, in_max, out_min, out_max):
        print(val, in_min, in_max, out_min, out_max)
        if in_min == in_max:
            return val
        return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))

    source_weights = df_encoded.groupby(by='source')['value'].sum()
    target_weights = df_encoded.groupby(by='target')['value'].sum()

    node_weights = pd.concat([source_weights, target_weights])
    node_weights = node_weights.to_frame()
    node_weights = node_weights.groupby(node_weights.index).max()

    node_weights = node_weights.join(layers_nodes)
    # node_weights['value'] = node_weights.groupby('layer')['value'].apply(lambda x: rescale(x, min(x), max(x), 0., 1.))

    ind = node_weights.index.str.contains('others')

    others_value = node_weights.loc[ind, 'value']
    print('others_value', others_value)
    node_weights.loc[ind, 'value'] = 0.
    node_weights.sort_values(by=['layer', 'value'], ascending=False, inplace=True)
    print('others_value', others_value)
    node_weights.loc[others_value.index, 'value'] = others_value
    n_layers = len(layers_nodes['layer'].unique())

    node_weights['x'] = (node_weights['layer'] - 2) * 0.1 + 0.16
    # node_weights['x'] = (node_weights['layer']-2) *0.15 + 0.16
    ind = node_weights.layer == 0
    node_weights.loc[ind, 'x'] = 0.01
    ind = node_weights.layer == 1
    node_weights.loc[ind, 'x'] = 0.08
    ind = node_weights.layer == 2
    node_weights.loc[ind, 'x'] = 0.14

    xs = np.linspace(0.14, 1, 6, endpoint=False)
    for i, x in enumerate(xs[1:]):
        print(i, x)
        ind = node_weights.layer == i + 3
        node_weights.loc[ind, 'x'] = x

    # node_weights.loc[ind,'x' ] = 0.3
    # ind = node_weights.layer==4
    # node_weights.loc[ind,'x' ] = 0.4
    # ind = node_weights.layer==5
    # node_weights.loc[ind,'x' ] = 0.5
    # ind = node_weights.layer==6
    # node_weights.loc[ind,'x' ] = 0.6
    # ind = node_weights.layer==7
    # node_weights.loc[ind,'x' ] = 0.7

    print('node_weights', node_weights)
    # node_weights.to_csv('node_weights.csv')
    dd = node_weights.groupby('layer')['value'].transform(pd.Series.sum)
    node_weights['layer_weight'] = dd
    node_weights['y'] = node_weights.groupby('layer')['value'].transform(pd.Series.cumsum)
    node_weights['y'] = (node_weights['y'] - .5 * node_weights['value']) / (1.5 * node_weights['layer_weight'])
    # node_weights['y'] = (node_weights['y'] -  .5 * node_weights['value']) /(node_weights['layer_weight'])

    # root node
    # ind = node_weights.layer==7
    # node_weights.loc[ind,'x' ] = 0.67
    node_weights.loc[ind, 'y'] = 0.33

    print('node_weights', node_weights['x'], node_weights['y'])
    node_weights.sort_index(inplace=True)
    node_weights.to_csv('node_weights_original.csv')
    # node_weights.to_csv('xy.csv')
    return node_weights['x'], node_weights['y']


# def get_data_trace(links, all_node_labels, node_pos, layers, node_colors=None):
def get_data_trace(links, node_df, height, width, fontsize=6):
    '''
    linkes: dataframe [source, target, value]
    all_node_labels: list of real node names
    node_pos: tuples of (x, y) for neach node
    layers: dataframe with index is node name and columns is layer
    node_colors, list of colors for each node (Same order as all_node_labels )
    '''

    # df = pd.DataFrame(node_colors, index=all_node_labels,columns=['color'] )
    # df.to_csv('all_node_labels.csv')

    all_node_labels = node_df.short_name.values
    node_colors = node_df.color.values
    x = node_df.x.values
    y = node_df.y.values

    def rescale(val, in_min, in_max, out_min, out_max):
        print(val, in_min, in_max, out_min, out_max)
        if in_min == in_max:
            return val
        return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))

    x = rescale(x, min(x), max(x), 0.01, .98)
    # y = rescale(y, min(y), max(y), 0.01, .96)

    data_trace = dict(
        type='sankey',
        arrangement='snap',
        domain=dict(
            x=[0, 1.],
            y=[0, 1.]
        ),
        orientation="h",
        valueformat=".0f",
        node=dict(
            pad=2,
            thickness=10,
            line=dict(
                color="white",
                width=.5
            ),
            label=all_node_labels,
            x=x,
            y=y,
            color=node_colors,

        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color']
        )
    )

    layout = dict(
        height=height,
        width=width,
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0.1,  # bottom margin
            t=8,  # top margin
        ),
        font=dict(
            size=fontsize, family='Arial',
        )
    )
    return data_trace, layout


def get_node_colors(all_node_labels, remove_others=True):
    def color_to_hex(color):
        r, g, b, a = [255 * c for c in color]
        c = '#%02X%02X%02X' % (r, g, b)
        return c

    color_idx = np.linspace(1, 0, len(all_node_labels))
    cmp = plt.cm.Reds
    node_colors = {}
    for i, node in zip(color_idx, all_node_labels):
        if 'other' in node:
            if remove_others:
                c = (255, 255, 255, 0.0)
            else:
                c = (232, 232, 232, 0.5)
        else:
            colors = list(cmp(i))
            colors = [int(255 * c) for c in colors]
            colors[-1] = 0.7  # set alpha
            c = colors
        node_colors[node] = c

    return node_colors


def get_edge_colors(df, node_colors_dict, remove_others=True):
    colors = []
    for i, row in df.iterrows():
        if ('others' in row['source']) or ('others' in row['target']):
            if remove_others:
                colors.append('rgba(255, 255, 255, 0.0)')
            else:
                colors.append('rgba(192, 192, 192, 0.2)')
        else:
            #             colors.append('rgba(252, 186, 3, .4)')
            base_color = [l for l in node_colors_dict[row['source']]]
            base_color[-1] = 0.2
            base_color = 'rgba{}'.format(tuple(base_color))
            # colors.append('rgba(192,192,192,0.5)')
            colors.append(base_color)
    return colors


def get_node_colors_ordered(high_nodes_df, col_name, remove_others=True):
    '''
    high_nodes_df: data frame with index (node names) andcolumns [layer, coef_combined]
    '''
    node_colors = {}
    layers = high_nodes_df.layer.unique()
    for l in layers:
        nodes_ordered = high_nodes_df[high_nodes_df.layer == l].sort_values(col_name, ascending=False).index
        node_colors.update(get_node_colors(nodes_ordered, remove_others))

    return node_colors


def get_first_layer(node_weights, number_of_best_nodes, col_name='coef', include_others=True):
    gene_weights = node_weights[1].copy()
    feature_weights = node_weights[0].copy()

    gene_weights = gene_weights[[col_name]]
    feature_weights = feature_weights[[col_name]]

    if number_of_best_nodes == 'auto':
        S = gene_weights[col_name].sort_values()
        n = get_nlargeest_ind(S)
        top_genes = list(gene_weights.nlargest(n, col_name).index)
    else:
        top_genes = list(gene_weights.nlargest(number_of_best_nodes, col_name).index)

    # gene normalization
    print('top_genes', top_genes)
    genes = gene_weights.loc[top_genes]
    genes[col_name] = np.log(1. + genes[col_name].abs())

    print(genes.shape)
    print(genes.head())

    df = feature_weights

    if include_others:
        df = df.reset_index()

        df.columns = ['target', 'source', 'value']
        df['target'] = df['target'].map(lambda x: x if x in top_genes else 'others1')
        df = df.groupby(['source', 'target']).sum()

        df = df.reset_index()
        df.columns = ['source', 'target', 'value']

        print('df groupby')
        print(df)

    else:
        df = feature_weights.loc[top_genes]
        df = df.reset_index()
        df.columns = ['target', 'source', 'value']

    df['direction'] = df['value'] >= 0.
    df['value'] = abs(df['value'])

    df['source'] = df['source'].replace('mut_important', 'mutation')
    df['source'] = df['source'].replace('cnv', 'copy number')
    df['source'] = df['source'].replace('cnv_amp', 'amplification')
    df['source'] = df['source'].replace('cnv_del', 'deletion')
    df['source'] = df['source'].replace('gene_expression', 'methylation')
    df['layer'] = 0

    # normalize features by gene
    # groups = df.groupby('target')

    # sum1 = groups['value'].transform(np.sum)
    df['value'] = df['value'] / df.groupby('target')['value'].transform(np.sum)
    df = df[df.value > 0.0]

    # multiply by the gene importance
    # genes
    df = pd.merge(df, genes, left_on='target', right_index=True, how='left')
    print(df.shape)
    df.coef.fillna(10.0, inplace=True)
    df.value = df.value * df.coef * 150.

    print(df.shape)

    return df


def get_fromated_network(links, high_nodes_df, col_name, remove_others):
    # get node colors
    node_colors_dict = get_node_colors_ordered(high_nodes_df, col_name, remove_others)

    print('node_colors_dict', node_colors_dict)
    # exception for first layer
    node_colors_dict['amplification'] = (224, 123, 57, 0.7)  # amps
    node_colors_dict['deletion'] = (1, 55, 148, 0.7)  # deletion
    node_colors_dict['mutation'] = (105, 189, 210, 0.7)  # mutation
    node_colors_dict['methylation'] = (100, 55, 200, 0.7)  # mutation
    node_colors_dict['expression_tpm'] = (60, 55, 50, 0.7)  # mutation

    # get colors
    links['color'] = get_edge_colors(links, node_colors_dict, remove_others)

    # get node colors
    for key, value in node_colors_dict.items():
        node_colors_dict[key] = 'rgba{}'.format(tuple(value))

    # remove links with no values
    links = links.dropna(subset=['value'], axis=0)

    # enocde nodes (convert node names into numbers)
    linkes_filtred_encoded_df, all_node_labels, node_code = encode_nodes(links)

    # get node per layers
    node_layers_df = high_nodes_df[['layer']]

    # remove self connection
    ind = linkes_filtred_encoded_df.source == linkes_filtred_encoded_df.target
    linkes_filtred_encoded_df = linkes_filtred_encoded_df[~ind]

    # make sure we positive values for all edges
    linkes_filtred_encoded_df.value = linkes_filtred_encoded_df.value.abs()

    # linkes_filtred_encoded_df = linkes_filtred_encoded_df.fillna(0.001)

    # linkes_filtred_encoded_df = linkes_filtred_encoded_df[~linkes_filtred_encoded_df['value'].isna()]

    x, y = get_x_y(links, node_layers_df)

    # shorten names

    def get_short_names(all_node_labels):
        PATHWAY_PATH = "/v//projects/methyldeeplearn-pallag/paurel/pnet_aging/_database/pathways/"
        df = pd.read_excel(join(PATHWAY_PATH, 'pathways_short_names.xlsx'), index_col=0)
        mapping_dict = {}
        for k, v in zip(df['Full name'].values, df['Short name (Eli)'].values):
            mapping_dict[k] = str(v)

        all_node_labels_short = []
        for l in all_node_labels:
            short_name = l
            if l in mapping_dict.keys() and not mapping_dict[l] == 'nan':
                short_name = mapping_dict[l]

            if 'others' in short_name:
                short_name = 'residual'
            if 'root' in short_name:
                short_name = 'outcome'

            all_node_labels_short.append(short_name)

        # all_node_labels_short = []
        # for n in all_node_labels:
        #     to_be_added = str(n)
        #     if len(to_be_added) >=30:
        #         to_be_added = to_be_added[:25] + ' ...'
        #     if 'others' in to_be_added :
        #         to_be_added ='residual'
        #     if 'root' in to_be_added :
        #         to_be_added ='outcome'
        #         # to_be_added ='root'
        #     # to_be_added = '{}({},{})'.format(to_be_added,str(round(x[n],2)),str(round(y[n],2)) )
        #     all_node_labels_short.append(to_be_added)
        return all_node_labels_short

    # node_colors_list = []
    # for l in all_node_labels:
    #     node_colors_list.append(node_colors_dict[l])
    # xy_pos = (x, y)
    # node_colors_df = pd.DataFrame(node_colors_dict)

    node_colors_list = []
    for l in all_node_labels:
        node_colors_list.append(node_colors_dict[l])

    all_node_labels_short = get_short_names(all_node_labels)

    data = np.column_stack((node_code, node_colors_list, all_node_labels_short))
    nodes_df = pd.DataFrame(data, columns=['code', 'color', 'short_name'], index=all_node_labels)
    nodes_df = nodes_df.join(x, how='left')
    nodes_df = nodes_df.join(y, how='left')

    print(nodes_df.head())
    print(nodes_df.shape)
    # return linkes_filtred_encoded_df, all_node_labels_short, xy_pos, node_layers_df, node_colors_list
    return linkes_filtred_encoded_df, nodes_df


# #remove self connections


def get_MDM4_nodes(links_df):
    import networkx as nx
    net = nx.from_pandas_edgelist(links_df, 'target', 'source', create_using=nx.DiGraph())
    net.name = 'reactome'
    # add root node
    roots = [n for n, d in net.in_degree() if d == 0]
    root_node = 'root'
    edges = [(root_node, n) for n in roots]
    net.add_edges_from(edges)
    # convert to tree
    tree = nx.bfs_tree(net, 'root')
    traces = list(nx.all_simple_paths(tree, 'root', 'HMGA1_cg01745499'))
    #traces = list(nx.all_simple_paths(tree, 'root', 'MDM4'))
    print(len(traces))
    all_odes = []
    for t in traces:
        print ('-->'.join(t))
        all_odes.extend(t)

    nodes = np.unique(all_odes)
    return nodes
    # MDM4_subnet = tree.subgraph(nodes)


