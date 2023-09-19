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