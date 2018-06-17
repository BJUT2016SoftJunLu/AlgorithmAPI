import lightgbm as lgb
import pandas as pd
import datetime

NOW_TIME = datetime.datetime.now().strftime("%y%m%d%H%M")

MODEL_PATH = '../model/lightgbm_' + NOW_TIME
RESULT_PATH = '../result/lightgbm_' + NOW_TIME + '.csv'

TRAIN_PATH = '../data/train.csv'
VALID_PATH = '../data/train.csv'
TEST_PATH = '../data/test.csv'

def model_dataset(Xtrain_data,Ytrian_data,Xvalid_data,Yvalid_data,X_names,categorical_names):

    dtrian = lgb.Dataset(Xtrain_data.values,
                         label=Ytrian_data.values,
                         # 配合categorical_feature使用
                         feature_name=X_names,
                         # 指定种类特征的名字,否则使用Pandas中的种类变量
                         categorical_feature=categorical_names)

    dvalid = lgb.Dataset(Xvalid_data.values,
                         label=Yvalid_data.values,
                         feature_name=X_names,
                         categorical_feature=categorical_names)

    return dtrian,dvalid




def model_train(train_data,early_stopping_rounds,num_boost_round,valid_data,verbose_eval,init_model):

    params_dict = {

        "boosting":'gbdt',      # default=gbdt  options=gbdt, rf, dart, goss
        "application":'binary', # 任务类型:默认regression,选项包括(regression, regression_l1, huber, fair, poisson, quantile, mape, gammma, tweedie, binary, multiclass, multiclassova, xentropy, xentlambda, lambdarank)
        'metric':'auc',         # 评价标准:默认空,选项包括[l1(absolute loss),l2(square loss),l2_root(root square los),auc,binary_logloss(),multi_logloss]
        "learning_rate":0.05,   # 学习率(默认:0.1)
        "num_leaves":15,         # 每棵树的叶子节点个数(默认:127,加大能够提升准确率,减小防止过拟合)
        "max_depth":4,          # 每棵树的最大深度(默认:-1,加大能够提升准确率,减小防止过拟合)
        "max_bin":200,          # 每个特征的最大分箱数(加大能够提升准确率,减小防止过拟合)
        "bin_construct_sample_cnt":200000,      # 分箱时使用的样本数(默认:200000,加大能够提升准确率)
        "min_data_in_leaf":100,                 # 每个叶子节点最小的样本数(默认:20,加大防止过拟合,避免生成一个过深的树,过小可能导致欠拟合)
        "bagging_fraction":0.9,                 # 采样的比例(默认:1,加大防止过拟合)
        "bagging_freq":1,                       # 进行采样的迭代次数(默认:0)
        "feature_fraction":0.9,                 # 每次列采样比例(减小加快训练，加大防止过拟合)
        "lambda_l1":0,                          # L1正则化系数(默认:0)
        "lambda_l2":0,                          # L2正则化系数(默认:0)
        "scale_pos_weight":200,                 # 正样本的权重(默认:1，处理数据分布不平衡)
        "min_sum_hessian_in_leaf":5,            # 如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束(默认是:1,取值范围是0到正无穷)range(1,6,2)
        "num_threads":8,                        # 并行的线程数
        "verbosity":1                           # 日志打印方式
    }

    evals_result = {}

    boost_model = lgb.train(params_dict,
                          train_set=train_data,
                          valid_sets=[train_data,valid_data],
                          valid_names=['train','valid'],     # 训练数据标签名词
                          early_stopping_rounds=early_stopping_rounds,
                          num_boost_round=num_boost_round,
                          verbose_eval = verbose_eval, # 每个多少次迭代打印训练信息
                          evals_result=evals_result,   # 将训练信息存储到字典
                          init_model=init_model,       # 是否已存在模型
                          fobj=None,                   # 是否自定义目标函数
                          feval=None)                  # 是否自定义评价指标

    # print("the total iteration number is %d" %(boost_model.current_iteration()))
    # print("the featrue number is %d" % (boost_model.num_feature()))
    # print("the featrue number name is", boost_model.feature_name())
    # print("the time featrue is used is", boost_model.feature_importance())

    return boost_model


def model_save(boost_model,model_path):
    # 保存已经训练完成的模型(model_path:文件路径,num_iteration:保存第几次迭代的模型)
    boost_model.save_model(model_path,num_iteration=-1)
    return


def model_load(model_path):
    # 加载已经训练完成的模型(model_path:文件路径)
    boost_model = lgb.Booster(model_file=model_path)
    return boost_model


def model_predict(boost_model,test_data):
    output = boost_model.predict(test_data)
    return output

def main():
    predictors = []
    predictors.extend(['app', 'device', 'os', 'channel', 'hour', 'day',
                       'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                       'ip_app_os_count', 'ip_app_os_var',
                       'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 'nextClick', 'nextClick_shift',
                       'ip_app_count', 'ip_day_hour_count', 'ip_app_os_count', 'ip_day_hour_nunique',
                       'ip_app_os_hour_var',
                       'ip_os_device_next_click_time', 'ip_app_device_os_channel_next_click_time',
                       'ip_os_device_app_next_click_time', 'ip_channel_previous_click_time',
                       'ip_os_previous_click_time'])

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
        'ip_app_channel_mean_hour': 'float32',
        # *************************************
        'ip_app_count': 'uint16',
        'ip_day_hour_count': 'uint16',
        'ip_app_os_count': 'uint16',
        'ip_day_hour_nunique': 'uint16',
        'ip_app_os_hour_var': 'float32',
        'ip_os_device_next_click_time': 'float32',
        'ip_app_device_os_channel_next_click_time': 'float32',
        'ip_os_device_app_next_click_time': 'float32',
        'ip_channel_previous_click_time': 'float32',
        'ip_os_previous_click_time': 'float32'
    }

    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

    train_data = pd.read_csv(TRAIN_PATH,usecols=predictors + ['is_attributed'],dtype=dtypes,index_col=False,compression='bz2')
    valid_data = pd.read_csv(VALID_PATH, usecols=predictors + ['is_attributed'], dtype=dtypes, index_col=False,compression='bz2')




    dtrain, dvalid = model_dataset(train_data[predictors],train_data[target],valid_data[predictors],valid_data[target],predictors,categorical)
    booster = model_train(train_data=dtrain,valid_data=dvalid,early_stopping_rounds=30,num_boost_round=50,verbose_eval=10,init_model=None)

    test_data = pd.read_csv(TEST_PATH,index_col=False)


    output = model_predict(booster,test_data[predictors])
    result = pd.DataFrame()
    result = test_data[['click_id']]
    result['is_attributed'] = output
    result.to_csv(RESULT_PATH,index=False)
    model_save(booster,MODEL_PATH)



if __name__ == "__main__":
    main()