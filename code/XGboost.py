import xgboost as xgb
import pandas as pd
import datetime
import gc

NOW_TIME = datetime.datetime.now().strftime("%y%m%d%H%M")

MODEL_PATH = '../model/xgboost_' + NOW_TIME
RESULT_PATH = '../result/xgboost_' + NOW_TIME + '.csv'

TRAIN_PATH = '../data/train.csv'
VALID_PATH = '../data/train.csv'
TEST_PATH = '../data/test.csv'


def model_data(train_data, Ytrian_data, Xvalid_data, Yvalid_data):
    dtrain = xgb.DMatrix(train_data,
                         label=Ytrian_data,
                         nthread=8)

    dvalid = xgb.DMatrix(Xvalid_data,
                         label=Yvalid_data,
                         nthread=8)
    return dtrain, dvalid


def model_train(dtrian, dvalid, num_boost_round, early_stopping_rounds, verbose_eval, xgb_model):
    params_dict = {
        "booster": 'gbtree',  # 基础模型(默认:gbtree,选项包括(gbliner,gbtree))
        "objective": "binary:logistic",
        # 任务类型(默认:(reg:linear),选项包括(reg:logistic,reg:logistic,binary:logitraw,gpu:reg:linear等))
        "tree_method": 'hist',  # 树的生成方法(默认:(auto)) 选项包括('auto','exact’,'approx','hist’,'gpu_exact','gpu_hist')
        "eval_metric": "auc",  # 评价标准(默认:根据objective选定)
        "eta": 0.2,  # 学习率(默认:0.3)
        "gamma": 0,  # 进行分裂的loss减少的最小值(默认是:0,增大防止过拟合,太大会导致欠拟合)[0.5~1]
        "min_child_weight": 5,  # 如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束(默认是:1,取值范围是0到正无穷)range(1,6,2)
        "max_depth": 4,  # 每棵树的最大深度(默认:6)
        "subsample": 0.9,  # 采样的比例(默认:1,减小这个参数的值防止过拟合,太小会导致欠拟合)
        "colsample_bytree": 0.9,  # 每次列采样比例(默认:1,减小这个参数的值防止过拟合,太小会导致欠拟合)
        "lambda": 1,  # L1正则化系数,防止过拟合(默认:1)
        "alpha": 4,  # L2正则化系数，加快算法速度(默认:1)
        "scale_pos_weight": 200,  # 正样本的权重(默认:1，处理数据分布不平衡)
        "nthread": 8,  # 并行的线程数
        "silent": 1
    }

    evals_result = {}

    booster_model = xgb.train(params_dict,
                              dtrain=dtrian,
                              evals=[(dtrian, 'train'), (dvalid, 'valid')],
                              num_boost_round=num_boost_round,
                              early_stopping_rounds=early_stopping_rounds,
                              verbose_eval=verbose_eval,  # # 每个多少次迭代打印训练信息
                              evals_result=evals_result,  # 将训练信息存储到字典
                              xgb_model=xgb_model,  # 是否加载已有模型(路径)
                              feval=None,  # 是否自定义评价指标
                              obj=None, )  # 是否自定义目标函数

    return booster_model


def model_save(model, model_path):
    # 保存已经训练完成的模型
    model.save_model(model_path)
    return


def model_load(model_path):
    # 加载已经训练完成的模型(model_path:文件路径)
    booster_model = xgb.Booster(model_path)
    return booster_model


def model_predict(model, test_data):
    # 预测数据
    result = model.predict(test_data)
    return result


def main():
    predictors = []
    predictors.extend(['app', 'device', 'os', 'channel', 'hour', 'day',
                       'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                       'ip_app_os_count', 'ip_app_os_var',
                       'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 'nextClick', 'nextClick_shift'])

    for i in range(0, 9):
        predictors.append("X%d" % (i))

    target = 'is_attributed'

    dtypes = {
        'app': 'uint16',
        'channel': 'uint16',
        'device': 'uint16',
        'is_attributed': 'uint8',
        'os': 'uint16',
        'day': 'uint8',
        'hour': 'uint8',
        'X0': 'uint16',
        'X1': 'uint16',
        'X2': 'uint16',
        'X3': 'uint16',
        'X4': 'uint16',
        'X5': 'uint16',
        'X6': 'uint16',
        'X7': 'uint16',
        'X8': 'uint16',
        'nextClick': 'int64',
        'nextClick_shift': 'float32',
        'ip_tcount': 'uint16',
        'ip_app_count': 'uint16',
        'ip_app_os_count': 'uint16',
        'ip_tchan_count': 'float32',
        'ip_app_os_var': 'float32',
        'ip_app_channel_var_day': 'float32',
        'ip_app_channel_mean_hour': 'float32'
    }

    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

    print("加载训练数据")
    train_data = pd.read_csv(TRAIN_PATH, usecols=predictors + [target], dtype=dtypes, index_col=False)

    # 随机抽取30%
    print("随机抽取")


    sub_train_data = train_data.sample(frac=0.3, axis=0)

    del train_data
    gc.collect()
    print("加载验证数据")
    valid_data = pd.read_csv(VALID_PATH, usecols=predictors + [target], dtype=dtypes, index_col=False)

    print("生成模型数据")
    dtrain, dvalid = model_data(sub_train_data[predictors], sub_train_data[target], valid_data[predictors],
                                valid_data[target])
    del sub_train_data
    del valid_data
    gc.collect()

    print("模型训练")
    booster = model_train(dtrain,
                          dvalid,
                          num_boost_round=50,
                          early_stopping_rounds=10,
                          verbose_eval=10,
                          xgb_model=None)

    del dtrain
    del dvalid
    gc.collect()

    test_data = pd.read_csv(TEST_PATH, index_col=False)
    dtest = xgb.DMatrix(test_data[predictors],nthread=8)
    print("模型预测")
    output = model_predict(booster, dtest)
    result = pd.DataFrame()
    result = test_data[['click_id']]
    result['is_attributed'] = output
    result.to_csv(RESULT_PATH, index=False)
    model_save(booster, MODEL_PATH)

if __name__ == "__main__":
    main()