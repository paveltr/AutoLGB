import warnings

warnings.filterwarnings('ignore')
import pickle

from tqdm import tqdm
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.model_selection import check_cv
from category_encoders import CatBoostEncoder
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
import numpy as np
import pandas as pd
import os
import gc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, average_precision_score, roc_auc_score, precision_score, \
    mean_absolute_error, mean_squared_error, r2_score
import datetime
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

np.random.seed(0)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6)))


def sd(col, max_loss_limit=0.001, avg_loss_limit=0.001, na_loss_limit=0, n_uniq_loss_limit=0, fillna=0):
    """
    max_loss_limit - don't allow any float to lose precision more than this value. Any values are ok for GBT algorithms as long as you don't unique values.
                     See https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]
    avg_loss_limit - same but calculates avg throughout the series.
    na_loss_limit - not really useful.
    n_uniq_loss_limit - very important parameter. If you have a float field with very high cardinality you can set this value to something like n_records * 0.01 in order to allow some field relaxing.
    """
    is_float = str(col.dtypes)[:5] == 'float'
    na_count = col.isna().sum()
    n_uniq = col.nunique(dropna=False)
    try_types = ['float16', 'float32']

    if na_count <= na_loss_limit:
        try_types = ['int8', 'int16', 'float16', 'int32', 'float32']

    for type in try_types:
        col_tmp = col

        # float to int conversion => try to round to minimize casting error
        if is_float and (str(type)[:3] == 'int'):
            col_tmp = col_tmp.copy().fillna(fillna).round()

        col_tmp = col_tmp.astype(type)
        max_loss = (col_tmp - col).abs().max()
        avg_loss = (col_tmp - col).abs().mean()
        na_loss = np.abs(na_count - col_tmp.isna().sum())
        n_uniq_loss = np.abs(n_uniq - col_tmp.nunique(dropna=False))

        if max_loss <= max_loss_limit and avg_loss <= avg_loss_limit and na_loss <= na_loss_limit and n_uniq_loss <= n_uniq_loss_limit:
            return col_tmp

    # field can't be converted
    return col


def reduce_mem_usage_sd(df, deep=True, verbose=False, obj_to_cat=False):
    numerics = ['int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes

        # collect stats
        na_count = df[col].isna().sum()
        n_uniq = df[col].nunique(dropna=False)
        
        # numerics
        if col_type in numerics:
            df[col] = sd(df[col])

        # strings
        if (col_type == 'object') and obj_to_cat:
            df[col] = df[col].astype('category')
        
        if verbose:
            print(f'Column {col}: {col_type} -> {df[col].dtypes}, na_count={na_count}, n_uniq={n_uniq}')
        new_na_count = df[col].isna().sum()
        if (na_count != new_na_count):
            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost na values. Before: {na_count}, after: {new_na_count}')
        new_n_uniq = df[col].nunique(dropna=False)
        if (n_uniq != new_n_uniq):
            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost unique values. Before: {n_uniq}, after: {new_n_uniq}')

    end_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))
    return df


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2 # just added 
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))
    return df

class TargetEncoderCV(BaseEstimator, TransformerMixin):
    '''
    Fold-based target encoder robust to overfitting
    '''

    def __init__(self, cv, **cbe_params):
        self.cv = cv
        self.cbe_params = cbe_params

    @property
    def _n_splits(self):
        return check_cv(self.cv).n_splits

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        self.cbe_ = []
        cv = check_cv(self.cv)
        cbe = CatBoostEncoder(
            cols=X.columns.tolist(),
            return_df=False,
            **self.cbe_params)

        X_transformed = np.zeros_like(X, dtype=np.float64)
        for train_idx, valid_idx in cv.split(X, y):
            self.cbe_.append(clone(cbe).fit(X.loc[train_idx], y[train_idx]))
            X_transformed[valid_idx] = self.cbe_[-1].transform(X.loc[valid_idx])
        return pd.DataFrame(X_transformed, columns=X.columns)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = np.zeros_like(X, dtype=np.float64)
        for cbe in self.cbe_:
            X_transformed += cbe.transform(X) / self._n_splits
        return pd.DataFrame(X_transformed, columns=X.columns)


def uncorrelate_df(df):
    fixed_columns = []
    pearson = df.sample(n=df.shape[1] * 10, random_state=0).corr(method='pearson') if df.shape[0] > 10000 else df.corr(method='pearson')
    pca_encoder = {}
    for i, column in tqdm(enumerate(pearson.columns)):
        if column not in fixed_columns:
            d = pearson[column]
            d = d[np.abs(d) > 0.7].index
            if d.shape[0] > 1:
                try:
                    pca = PCA(n_components=1)
                    df[str(column) + '_PCA'] = pca.fit_transform(df[d.tolist()].fillna(0))
                    pca_encoder[str(column)] = {'pca': pca, 'enc_cols': d.tolist()}
                    fixed_columns.extend(d.tolist())
                    df.drop(d.tolist(), axis=1, inplace=True)
                    print('Column %s was processed for correlation' % column)
                except KeyError:
                    pass
    return df, pca_encoder


def pca_encode(X, pca_enc):
    if type(X) != pd.DataFrame:
        X = pd.DataFrame(X, columns=[str(c) for c in range(X.shape[1])])
    for column in X.columns:
        try:
            X[column + '_PCA'] = pca_enc[str(column)]['pca'] \
                .transform(X[pca_enc[str(column)]['enc_cols']].fillna(0))
            X.drop(pca_enc[str(column)]['enc_cols'], axis=1, inplace=True)
        except KeyError as e:
            pass
    return X


def train_model(X, y, clf_params, folder, group_id=None, task_type='binary_classification',
               id_list=[]):
    '''
    Train LGB model
    @id_list => list of specific ids to exclude from prediction because model was trained on them
    '''
    
    X = X.fillna(0)
    
    cols = [c for c in X.columns if X[c].dtype != 'object']
    if not np.isfinite(X[cols]).all().all():
        for c in cols:
            if not np.isfinite(X[c]).all():
                max_ = X.loc[X[c] < np.inf][c].max()
                min_ = X.loc[X[c] > -np.inf][c].min()
                X.loc[X[c] == np.inf] = max_
                X.loc[X[c] == -np.inf] = min_
            
    assert np.isfinite(X[cols]).all().all() == True
    
    models = []
    id_list = np.array(id_list)

    oof = 0 * y.copy()
    encoded_data = []
    if group_id is None:
        folds = folder.split(X, y)
    else:
        folds = folder.split(X, y, group_id)
        
    for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds)):
        print('Fold:', fold_)

        tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]

        print('Number of train samples: {}, test samples: {}'.format(len(tr_x), len(vl_x)))

        cat_cols = []
        for column in tr_x.columns:
            if tr_x[column].dtype == 'object':
                cat_cols.append(column)

        te_cv = None
        if len(cat_cols) > 0:
            print('Categorical columns for target encoding: {}'.format(len(cat_cols)))
            print(cat_cols)
            te_cv = TargetEncoderCV(KFold(n_splits=3))
            tr_x[cat_cols] = te_cv.fit_transform(tr_x[cat_cols], tr_y)
            vl_x[cat_cols] = te_cv.transform(vl_x[cat_cols])

        if task_type == 'binary_classification':
            clf = LGBMClassifier(objective='binary',
                                 boosting='gbdt',
                                 metric='auc',
                                 n_jobs=20,
                                 num_iterations=10 ** 6,
                                 learning_rate=0.02,
                                 random_state=0,
                                 class_weight={0: np.mean(tr_y), 1: 1 - np.mean(tr_y)},
                                 **clf_params).fit(tr_x, tr_y, eval_set=[(tr_x, tr_y), (vl_x, vl_y)],
                                                   early_stopping_rounds=50, verbose=100)
            oof[val_idx] = clf.predict_proba(vl_x)[:, 1]
        elif task_type == 'regression':
            clf = LGBMRegressor(objective='regression',
                                boosting='gbdt',
                                metric='rmse',
                                n_jobs=20,
                                num_iterations=10 ** 6,
                                learning_rate=0.02,
                                random_state=0,
                                **clf_params).fit(tr_x, tr_y, eval_set=[(tr_x, tr_y), (vl_x, vl_y)],
                                                  early_stopping_rounds=50, verbose=100)
            oof[val_idx] = clf.predict(vl_x)
        encoded_data.append(vl_x)
        models.append({'clf': clf, 'teach_cols': X.columns, 'encoder': te_cv, 'cat_cols': cat_cols,
                       'train_ids': id_list[trn_idx] if len(id_list) > 0 else []})
    return models, oof, pd.DataFrame(np.vstack(encoded_data), columns=X.columns)


def gen_name():
    return ''.join(r for r in re.findall(r'[0-9]+', str(datetime.datetime.today())[:19]))


def lift_score(y_target, y_prediction, top=0.01):
    df = pd.DataFrame({'y': y_target.tolist(), 'y_pred': y_prediction.tolist()})
    top_df = df.nlargest(int(df.shape[0] * top), 'y_pred')
    return top_df['y'].mean() / df['y'].mean()


def precision_at_k(y_target, y_prediction, k=0.1):
    df = pd.DataFrame({'y': y_target.tolist(), 'y_pred': y_prediction.tolist()})
    top_df = df.nlargest(int(df.shape[0] * k), 'y_pred')
    return precision_score(top_df['y'], np.ones(top_df.shape[0]))


class AutoML():
    '''
    AutoML Class to train model
    '''

    def __init__(self,
                 task_type='binary_classification',
                 save_models=True,
                 save_metrics=True,
                 model_name = 'Model'
                 ):
        '''
        Initialize AutoML class

        @task_type => one of ['binary_classification', 'regression']
        @save_models => default True, save models after training in special folder
        @save_metrics => default True, save metrics and charts
        '''
        if task_type not in ['binary_classification', 'regression']:
            raise Exception("task_type needs to be one of ['binary_classification', 'regression']")
        self.task_type = task_type
        self.save_models = save_models
        self.save_metrics = save_metrics
        self.model = None
        self.oof = None
        self.model_name = model_name
        self.train_metrics = 'automl/metrics'
        self.train_features = 'automl/train/features'
        self.test_metrics = 'automl/test/metrics'
        self.test_features = 'automl/test/features'
        self.model_path = 'automl/model'

        if not os.path.exists(self.train_metrics):
            os.makedirs(self.train_metrics)
        if not os.path.exists(self.train_features):
            os.makedirs(self.train_features)

        if not os.path.exists(self.test_metrics):
            os.makedirs(self.test_metrics)
        if not os.path.exists(self.test_features):
            os.makedirs(self.test_features)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def save(self):
        '''
        Saved model to disk
        '''
        with open(self.model_path + '/' + gen_name() + '_' + self.model_name + '_model.pkl', 'wb') as pfile:
            pickle.dump(self, pfile, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path_to_model):
        '''
        Loads model from disk
        @path_to_model => path to the saved model
        '''
        with open(path_to_model, 'rb') as fid:
            clf = pickle.load(fid)
        return clf

    def fit(self, X, y, tune_params=True, fold_strategy='KFold', group_id=None, tune_mode='easy',
            id_list=[], manual_params = None):
        '''
        Returns trained model and out of fold predictions

        @tune_params => whether we need to find optimal model paramets
        @fold_strategy => can be KFold (by default), StratifiedKFold, GroupKFold
                          For GroupKFold you need to specify unique id list, e.g. customers' id list
        @tune_mode => can be 'easy' with 100 rounds for parameters search or 'hard' for 1000 rounds
        @id_list => list of specific ids to exclude from prediction because model was trained on them
        '''
        data_shape = X.shape
        print('Data shape: ', data_shape)
        
        if len(id_list) > 0:
            assert len(id_list) == data_shape

        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X, columns=[str(c) for c in range(X.shape[1])])
        if type(y) != pd.Series:
            y = pd.Series(y)
        y = y.reset_index(drop=True)
        # reduct memory comsumption
        X = reduce_mem_usage_sd(X)
        # merge correlated features into one single feature
        X_pca, self.pca_enc = uncorrelate_df(X[[c for c in X.columns if X[c].dtype != 'object']])
        X = pd.concat([X_pca,
                       X[[c for c in X.columns if X[c].dtype == 'object']]], axis=1)
        del X_pca
        gc.collect()
        assert X.shape[0] == data_shape[0]
        
        
        # train model
        if fold_strategy not in ['KFold', 'StratifiedKFold', 'GroupKFold']:
            raise Exception("fold_strategy parameter needs to be in ['KFold', 'StratifiedKFold', 'GroupKFold']")
        if fold_strategy == 'KFold':
            fold = KFold(random_state=0, shuffle=False, n_splits=5)
        elif fold_strategy == 'StratifiedKFold':
            fold = StratifiedKFold(random_state=0, shuffle=False, n_splits=5)
        elif fold_strategy == 'GroupKFold':
            fold = GroupKFold(n_splits=5)
        
        if not manual_params:
            cat_cols = []
            X_check = X.reset_index(drop=True).copy()
            for column in tqdm(X.columns):
                if X_check[column].dtype == 'object':
                    cat_cols.append(column)
            print('Categorical columns: ', cat_cols)
            X_check[cat_cols] = TargetEncoderCV(KFold(n_splits=3)).fit_transform(X_check[cat_cols], y)

            # set up model parameters
            if self.task_type == 'binary_classification':
                clf = LGBMClassifier(max_depth=-1, random_state=0, silent=True,
                                     metric='None', n_jobs=10, n_estimators=5000)
            elif self.task_type == 'regression':
                clf = LGBMRegressor(max_depth=-1, random_state=0, silent=True,
                                    metric='None', n_jobs=10, n_estimators=5000)

            param_test = {'num_leaves': [30, 50, 80, 100],
                          'min_child_samples': [100, 200, 500],
                          'min_child_weight': [1, 5, 10],
                          'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
                          'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
                          'reg_alpha': [1, 2, 5, 10],
                          'reg_lambda': [1, 2, 5, 10],
                          'max_depth': [3, 4, 5, 8]}
            gs = RandomizedSearchCV(
                estimator=clf,
                param_distributions=param_test,
                n_iter=100 if tune_mode == 'easy' else 1000,
                scoring='roc_auc' if self.task_type == 'binary_classification' else make_scorer(mean_squared_error),
                cv=3,
                refit=True,
                random_state=0,
                verbose=False)
            if X.shape[0] >= 20000:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_check.fillna(0).sample(n=X.shape[1] * 10, random_state=0).copy(),
                    y.sample(n=X.shape[1] * 10, random_state=0).copy(),
                    test_size=0.33, random_state=0)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_check.fillna(0).copy(),
                                                                    y.copy(),
                                                                    test_size=0.33, random_state=0)

            fit_params = {"early_stopping_rounds": 5,
                          "eval_metric": 'auc' if self.task_type == 'binary_classification' else 'rmse',
                          "eval_set": [(X_test, y_test)],
                          'eval_names': ['valid'],
                          'verbose': 0,
                          'categorical_feature': 'auto'}

            assert np.isfinite(X_train).all().all()
            assert ~np.isnan(X_train).all().all()

            assert np.isfinite(X_test).all().all()
            assert ~np.isnan(X_test).all().all()

            gs.fit(X_train, y_train, **fit_params)
            print('Parameters optimization best score reached: {} with params: {} ' \
                  .format(gs.best_score_, gs.best_params_))
        
            self.model, self.oof, self.train_data = train_model(X.reset_index(drop=True), y.reset_index(drop=True),
                                                                gs.best_params_, fold, group_id=group_id,
                                                                id_list=id_list)
        else:
            self.model, self.oof, self.train_data = train_model(X.reset_index(drop=True), y.reset_index(drop=True),
                                                                manual_params, fold, group_id=group_id,
                                                                id_list=id_list)

        print('Model trained! Success...')

        print('...Model evaluation....')
        print(self.evaluate(X, y, data_type='train'))
        self.plot_PR_curve(X, y, data_type='train')
        return self

    def explain_model(self, save=True, n_features=30):
        '''
        Explains model features
        @save => True by default, if we need to save the feature importance plot
        '''
        X = self.train_data.copy()
        n_limit = 25000 if X.shape[0] > 25000 else X.shape[0]
        n_features = n_features if len(self.model[0]['teach_cols']) > n_features else len(self.model[0]['teach_cols'])

        shaps = []
        data = []
        for model in self.model:
            shaps.append(shap.TreeExplainer(model['clf']).shap_values(X.sample(n=n_limit, random_state=0))[1])
            data.append(X.sample(n=n_limit, random_state=0))

        if save:
            fig = shap.summary_plot(
                np.vstack(shaps),
                pd.DataFrame(np.vstack(data), columns=self.model[0]['teach_cols']),
                max_display=n_features,
                title='Feature Importance Plot',
                show=not save
            )
            plt.savefig(self.train_features + '/' + gen_name() + '_' + self.model_name + '_feature_importance.png')
        else:
            shap.summary_plot(
                np.vstack(shaps),
                pd.DataFrame(np.vstack(data), columns=self.model[0]['teach_cols']),
                max_display=n_features,
                title='Feature Importance Plot',
                show=save
            )

        shap_importance = []
        shap_values = np.vstack(shaps)
        for i in range(len(self.model[0]['teach_cols'])):
            shap_importance.append([self.model[0]['teach_cols'][i], np.sum(np.absolute(shap_values[:, i]))])

        shap_importance = pd.DataFrame(shap_importance, columns=['feature', 'weight'])
        shap_importance = shap_importance.groupby('feature')['weight'].sum().reset_index()
        shap_importance = shap_importance.sort_values(by=['weight'], ascending=False).reset_index(drop=True)
        shap_importance.to_csv(self.train_features + '/' + gen_name() + '_' + self.model_name + '_feature_importance.csv', index=False,
                               sep=';', decimal=',')

    def explain_data(self, df, save=False, n_features=30):
        '''
        Explains model features for specific dataframe
        @df => pandas dataframe with features
        @save => False by default, if we need to save the feature importance plot
        '''
        n_limit = 25000 if df.shape[0] > 25000 else df.shape[0]
        n_features = n_features if len(self.model[0]['teach_cols']) > n_features else len(self.model[0]['teach_cols'])
        X = self.process_data(df)

        shaps = []
        data = []
        n_limit = 25000 if X.shape[0] > 25000 else X.shape[0]
        n_features = 30 if len(self.model[0]['teach_cols']) > 30 else len(self.model[0]['teach_cols'])

        for model in self.model:
            shaps.append(shap.TreeExplainer(model['clf']).shap_values(X.sample(n=n_limit, random_state=0))[1])
            data.append(X.sample(n=n_limit, random_state=0))

        if save:
            fig = shap.summary_plot(
                np.vstack(shaps),
                pd.DataFrame(np.vstack(data), columns=self.model[0]['teach_cols']),
                max_display=n_features,
                title='Feature Importance Plot',
                show=not save
            )
            plt.savefig(self.test_features + '/' + gen_name() + '_' + self.model_name + '_feature_importance.png')
        else:
            shap.summary_plot(
                np.vstack(shaps),
                pd.DataFrame(np.vstack(data), columns=self.model[0]['teach_cols']),
                max_display=n_features,
                title='Feature Importance Plot',
                show=save
            )

        shap_importance = []
        shap_values = np.vstack(shaps)
        for i in range(len(self.model[0]['teach_cols'])):
            shap_importance.append([self.model[0]['teach_cols'][i], np.sum(np.absolute(shap_values[:, i]))])

        shap_importance = pd.DataFrame(shap_importance, columns=['feature', 'weight'])
        shap_importance = shap_importance.groupby('feature')['weight'].sum().reset_index()
        shap_importance = shap_importance.sort_values(by=['weight'], ascending=False).reset_index(drop=True)
        shap_importance.to_csv(self.test_features + '/' + gen_name() + '_' + self.model_name + '_feature_importance.csv', index=False,
                               sep=';', decimal=',')

    def process_data(self, X):
        '''
        Converts data from raw to model data format
        @X => dataframe with features
        '''
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(pca_encode(X, self.pca_enc), columns=self.model[0]['teach_cols'])
        else:
            X = pca_encode(X, self.pca_enc)

        if self.model[0]['encoder'] is not None:
            X_cat = X[self.model[0]['cat_cols']].copy()
            X[self.model[0]['cat_cols']] = 0
            for m in tqdm(self.model):
                encoder = m['encoder']
                if encoder is not None:
                    X[m['cat_cols']] += encoder.transform(X_cat[m['cat_cols']].copy()) / len(self.model)
        return X[self.model[0]['teach_cols']].reset_index(drop=True)

    def plot_PR_curve(self, X_test, y_test,
                      data_type='test',
                      title='All',
                      y_limit=1.0):
        '''Plot average precision chart'''
        if self.task_type != 'binary_classification':
            raise Exception('Average precision curve can be built only for binary classification')
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import matplotlib.pyplot as plt
        from funcsigs import signature

        y_score = self.predict(X_test) if data_type != 'train' else self.oof.tolist()
        average_precision = average_precision_score(y_test, y_score)
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        step_kwargs = ({'step': 'post'}

                       if 'step' in signature(plt.fill_between).parameters

                       else {})
        plt.figure(figsize=(10, 10))
        plt.step(recall, precision, color='r', alpha=0.4, where='post')
        plt.fill_between(recall, precision, alpha=0.4, color='r', **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, y_limit])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}. Target: {1}'.format(average_precision, title))
        plt.show()
        if data_type == 'train':
            plt.savefig(self.train_metrics + '/' + gen_name() + '_PR_curve.png')
        elif data_type == 'test':
            plt.savefig(self.test_metrics + '/' + gen_name() + '_PR_curve.png')
        else:
            raise Exception("data_type should be one of values ['train', 'test']")

    def evaluate(self, X, y, save=True, data_type='test'):
        '''
        Evaluates model
        @X => dataframe with features
        @y => target
        @save => True by default, saves metrics to file
        @data_type => can be train or test
        '''
        preds = self.predict(X.copy()) if data_type != 'train' else self.oof
        metric_names = []
        metric_values = []
        comment = ''

        if self.task_type == 'binary_classification':
            for metric_name, metric_value in zip(['average_precision',
                                                  'roc_auc_score',
                                                  'precision@10%',
                                                  'lift_top_1%',
                                                  'lift_top_5%',
                                                  'lift_top_10%'],
                                                 [average_precision_score(y, preds),
                                                  roc_auc_score(y, preds),
                                                  precision_at_k(y, preds),
                                                  lift_score(y, preds),
                                                  lift_score(y, preds, top=0.05),
                                                  lift_score(y, preds, top=0.1)]):
                metric_names.append(metric_name)
                metric_values.append(metric_value)
            if roc_auc_score(y, preds) < 0.65 and lift_score(y, preds) < 2:
                comment = "\nWARNING !!!! THIS MODEL IS VERY WEAK. PERFORMANCE IS LOW!!!\n\n"
        elif self.task_type == 'regression':
            for metric_name, metric_value in zip(['mean_absolute_error',
                                                  'mean_squared_error',
                                                  'mean_absolute_percentage_error',
                                                  'R_Squared'],
                                                 [mean_absolute_error(y, preds),
                                                  mean_squared_error(y, preds),
                                                  mean_absolute_percentage_error(y, preds),
                                                  r2_score(y, preds)]):
                metric_names.append(metric_name)
                metric_values.append(metric_value)
            if r2_score(y, preds) < 0.5 or mean_absolute_percentage_error(y, preds) > 1:
                comment = "WARNING !!!! THIS MODEL IS VERY WEAK. PERFORMANCE IS LOW!!!"

        metrics_df = pd.DataFrame({'metric_name': metric_names, 'metric_value': metric_values})

        if save:
            if data_type == 'train':
                metrics_df.to_csv(self.train_metrics + '/' + gen_name() + '_' + self.model_name + '_metrics.csv', index=False,
                                  sep=';', decimal=',')
            elif data_type == 'test':
                metrics_df.to_csv(self.test_metrics + '/' + gen_name() + '_' + self.model_name + '_metrics.csv', index=False,
                                  sep=';', decimal=',')
            else:
                raise Exception("data_type should be one of values ['train', 'test']")
        print(comment)
        return metrics_df

    def predict(self, X, id_list=[]):
        '''
        Makes predictions
        @X => dataframe with features
        @id_list => list of specific ids to exclude from prediction because model was trained on them
        '''
        data = self.process_data(X.copy()).fillna(0)
        preds = pd.Series(np.zeros(X.shape[0])) # predictions
        counters = pd.Series(np.zeros(X.shape[0])) # count of number of predictions
        id_list=pd.Series(id_list)
        for m in tqdm(self.model):
            except_list = id_list[id_list.isin(m['train_ids'])].index.tolist()
            if self.task_type == 'binary_classification':
                preds[~preds.index.isin(except_list)] += \
                        m['clf'].predict_proba(data\
                                               .loc[~data.index.isin(except_list)][m['teach_cols']])[:, 1]
            elif self.task_type == 'regression':
                preds[~preds.index.isin(except_list)] += \
                        m['clf'].predict(data\
                                         .loc[~data.index.isin(except_list)][m['teach_cols']])
            counters[~counters.index.isin(except_list)] += 1
        return preds / counters

    def check_test_data(self, X):
        '''
        Checks if test data is very different from train
        @X => dataframe with features from test data
        '''
        X_train = self.train_data.copy()
        X_test = self.process_data(X.copy())[X_train.columns]
        X_train['y'] = np.ones(X_train.shape[0])
        X_test['y'] = np.zeros(X_test.shape[0])
        data = pd.concat([X_train, X_test]).reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(data.fillna(0).drop('y', axis=1).reset_index(drop=True),
                                                            data['y'],
                                                            test_size=0.33, random_state=0)

        clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
        clf.fit(X_train, y_train)
        predictions = clf.predict_proba(X_test)[:, 1]

        if roc_auc_score(y_test, predictions) > 0.8:
            print('WARNING!!!! Test data is very different from train, pay attention for future predictions')
        else:
            print('Data seems to be OK')

    def explain_one_sample(self, X):
        '''
        Draws decision plot for one sample
        @X => one sample dataframe
        '''
        if X.shape[0] > 1:
            raise Exception(
                'You need to pass only one sample of data for this function.\nIt means sample size (1, n_features)')
        explainer = shap.TreeExplainer(self.model[0]['clf'])
        shap_values = explainer.shap_values(self.process_data(X))[1]
        try:
            shap.decision_plot(explainer.expected_value[1], shap_values,
                               ignore_warnings=False, feature_names=self.model[0]['teach_cols'].tolist())
        except IndexError:
            shap.decision_plot(explainer.expected_value, shap_values,
                               ignore_warnings=False, feature_names=self.model[0]['teach_cols'].tolist())
