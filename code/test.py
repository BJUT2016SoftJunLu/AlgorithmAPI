import pandas as pd
import gc
import xgboost as xgb

from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score


def do_count(df,groups,target,new_name,target_dtype):
    gp = df[groups + [target]].groupby(groups)[target].count().reset_index().rename(columns={target:new_name})
    df = df.merge(gp,on=groups,how='left')
    df[new_name] = df[new_name].astype(target_dtype)
    del gp
    gc.collect()
    return


def do_var(df,groups,target,new_name,target_dtype):
    gp = df[groups + [target]].groupby(groups)[target].var().reset_index().rename(columns={target:new_name})
    df = df.merge(gp,on=groups,how='left')
    df[new_name] = df[new_name].astype(target_dtype)
    del gp
    gc.collect()
    return df


def do_mean(df,groups,target,new_name,target_dtype):
    gp = df[groups + [target]].groupby(groups)[target].mean(),reset_index().rename(column={target:new_name})
    df = df.merge(gp,on=groups,how='left')
    df[new_name] = df[new_name].astype(target_dtype)
    del gp
    gc.collect()
    return df


def do_nuique(df,groups,target,new_name,target_dtype):
    gp = df[groups + [target]].groupby(groups)[target].nunique().reset_index().rename(column={target:new_name})
    df = df.merge(gp,on=groups,how='left')
    df[new_name] = df[new_name].astype(target_dtype)
    del gp
    gc.collect()
    return df


def do_cumcount(df,groups,target,new_name,target_dtype):
    gp = df[groups + [target]].groupby(groups)[target].cumcount().reset_index().rename(column={target:new_name})
    df = df.merge(gp,on=groups,how='left')
    df[new_name] = df[new_name].astype(target_dtype)
    del gp
    gc.collect()
    return df


def do_nextClick(df,groups,new_name,target_dtype):
    df[new_name] = (df[groups + ['click_time']].groupby(groups).click_time.shift(-1) - df['click_time']).dt.second.astype(target_dtype)
    return df


def do_preClick(df,groups,new_name,target_dtype):
    df[new_name] = (df['click_time'] - (df[groups + ['click_time']].groupby(groups).click_time.shift(+1))).dt.second.astype(target_dtype)
    return df


def xgb_data(X_train,Y_train,X_valid,Y_valid):
    xgb_train = xgb.DMatrix(X_train,label=Y_train,nthread=8)
    xgb_valid = xgb.DMatrix(X_valid,label=Y_valid,nthread=8)
    return xgb_train,xgb_valid


def model_save(model,path):
    model.save(path)
    return

def model_load(path):
    model = xgb.Booster(path)
    return model


def model_predict(model,xgb_test):
    result = model.predict(xgb_test)
    return result


def xgb_cv(xgb_params,xgb_train):
    cv_out = xgb.cv(xgb_params,
                    xgb_train,
                    num_boost_round=1000,
                    nfold=10,
                    early_stopping_rounds=30,
                    verbose_eval=10,
                    show_stdv=True)
    return cv_out


def xgb_train(xgb_params,xgb_train,xgb_valid,num_boost_round,early_stopping_rounds,verbose_eval):
    model = xgb.train(xgb_params,
                      xgb_train,
                      evals=[(xgb_train,'train'),(xgb_valid,'valid')],
                      num_boost_round=num_boost_round,
                      early_stopping_rounds=early_stopping_rounds,
                      verbose_eval=verbose_eval,
                      xgb_model=None,   # 是否加载已有模型(路径)
                      feval=None,       # 是否自定义评价指标
                      obj=None)         # 是否自定义目标函数

    return model


def show_importance(model):

    feature_importance = model.get_fscore()
    features = pd.DataFrame()
    features['features'] = feature_importance.keys()
    features['importance'] = feature_importance.values()
    features.sort_values(by=['importance'],ascending=False,inplace=True)
    features.plot(kind='bar', figsize=(20, 7))
    return


def xgb_tune(x_trian,y_train):
    param = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }
    estimator = XGBClassifier(learning_rate =0.1,
                              n_estimators=140,
                              max_depth=5,
                              min_child_weight=1,
                              gamma=0,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              objective= 'binary:logistic',
                              nthread=4,
                              scale_pos_weight=1,
                              seed=27)

    gsearch = GridSearchCV(estimator=estimator,
                           param_grid=param,
                           scoring='roc_auc',
                           iid=False,
                           cv=5)

    gsearch.fit(x_trian, y_train)

    print(gsearch.cv_results_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)

    return


def main():

    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint8',
        'os': 'uint16',
        'channel': 'uint16',
        'click_time': 'str',
        'is_attributed': 'uint8',
        'click_id': 'uint32'
    }

    val_size = 10000

    train_data = pd.read_csv("../data/train.csv",
                             dtype=dtypes,
                             nrows=1000000,
                             index_col=False,
                             parse_dates=['click_time'],
                             usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed','click_id'])

    test_data = pd.read_csv("../data/test.csv",
                            dtype=dtypes,
                            nrows=10000,
                            index_col=False,
                            parse_dates=['click_time'],
                            usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])

    data = train_data.append(test_data)
    del train_data
    gc.collect()
    del test_data
    gc.collect()

    data['day'] = data['click_time'].dt.day.astype('uint8')
    data['hour'] = data['click_time'].dt.hour.astype('uint8')
    data['minute'] = data['click_time'].dt.minute.astype('uint8')
    data['second'] = data['click_time'].dt.second.astpye('uint8')

    xgb_params = {

        "booster": 'gbtree',                # 基础模型(默认:gbtree,选项包括(gbliner,gbtree))
        "objective": "binary:logistic",     # 任务类型(默认:(reg:linear),选项包括(reg:logistic,reg:logistic,binary:logitraw,gpu:reg:linear等))
        "tree_method": 'hist',              # 树的生成方法(默认:(auto)) 选项包括('auto','exact’,'approx','hist’,'gpu_exact','gpu_hist')
        "eval_metric": "auc",               # 评价标准(默认:根据objective选定)
        "eta": 0.2,                         # 学习率(默认:0.3)
        "gamma": 0,                         # 进行分裂的loss减少的最小值(默认是:0,增大防止过拟合,太大会导致欠拟合)[0.5~1]
        "min_child_weight": 5,              # 如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束(默认是:1,取值范围是0到正无穷)range(1,6,2)
        "max_depth": 4,                     # 每棵树的最大深度(默认:6) 通常取值：3-10
        "subsample": 0.9,                   # 采样的比例(默认:1,减小这个参数的值防止过拟合,太小会导致欠拟合)
        "colsample_bytree": 0.9,            # 每次列采样比例(默认:1,减小这个参数的值防止过拟合,太小会导致欠拟合)
        "lambda": 1,                        # L1正则化系数,防止过拟合(默认:1) 通常不使用，但可以用来降低过拟合
        "alpha": 4,                         # L2正则化系数，加快算法速度(默认:1)
        "scale_pos_weight": 1,              # 正样本的权重(默认:1，处理数据分布不平衡)
        "nthread": 8,                       # 并行的线程数
        "silent": 1
    }

    next_clicks = [

        ['ip', 'app', 'device', 'os', 'channel'],
        ['ip', 'os', 'device'],
        ['ip', 'os', 'device', 'app'],
        ['ip', 'os', 'device', 'channel'],
        ['ip', 'os', 'device', 'app', 'hour'],
        ['ip', 'os', 'device', 'channel', 'hour'],
        ['ip', 'os', 'device', 'app', 'hour'],
        ['ip', 'os', 'device', 'channel', 'hour'],
        ['device'],
        ['device', 'channel'],
        ['app', 'device', 'channel'],
        ['device', 'hour']
    ]

    pre_clicks = [
        ['ip', 'channel']
    ]

    return




if __name__ == "__main__":
    main()

# XGBoost参数调优:
# 1、选择较高的学习速率(0.1), 然后通过CV确定树数量
# 2、初始化max_depth, min_child_weight, gamma, subsample, colsample_bytree参数
# max_depth = 4
# min_child_weight = 1
# gamma = 0
# subsample = 0.8
# colsample_bytree = 0.8
# 3、max_depth 和 min_child_weight 参数调优
# 步长2
# 'max_depth': range(3, 10, 2) = 4
# 'min_child_weight': range(1, 6, 2) = 6
# 步长1
# 'max_depth': [4, 5, 6],
# 'min_child_weight': [4, 5, 6]
#
# min_child_weight = 6 到了阈值
# 'min_child_weight': [6, 8, 10, 12]
# 4、gamma参数调优
# 'gamma': [i / 10.0 for i in range(0, 5)]
#
# 5、调整subsample 和 colsample_bytree 参数
# 'subsample': [i / 10.0 for i in range(6, 10)],
# 'colsample_bytree': [i / 10.0 for i in range(6, 10)]
#
# 6、正则化参数调优
# 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100] = 1
# 'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05] = 0.005
#
# 'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100] = 1
# 'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05] = 0.005
#
# 7、降低学习速率以及使用更多的决策树，可以利用CV进行






