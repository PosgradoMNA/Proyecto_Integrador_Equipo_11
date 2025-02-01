####################################################################################
########################   Juan A. Nolazco Flores   ################################
####################################################################################
start_time = time.time()

path = "/content/drive/My Drive/JnOTE/graficos/"
name = 'oth0.18_ith0.7_PCA_Em0_KF_SF_CDF_SDF_Split3_nf100_bwf1_fi0.5_ov0_bdfb15_Stress_Expression'
file = path + "Emo3class/3Segments/" + name + ".csv"
container = []
# results_file = open(path + 'Results/' + 'result_' + file +"_15per_prop_01"+ '.txt', 'a')

# file= path + "Data/TotalMatrix_S2320_C720_norm.csv"
# PaHaw_data = pd.read_csv(path+file+'.csv')
data = pd.read_csv(file)
data = data.dropna(axis=1)  # Dropping the columns having NaN/NaT values
data = data.iloc[np.random.permutation(data.index)].reset_index(
    drop=True)  # Randomly Shuffle DataFrame Rows to avoid one of the folds only contains one category.

fbl = 0

# data=data._get_numeric_data()  # keep only columns which all values are numeric.
h2o_df = h2o.H2OFrame(data)  # convert panda dataframe to use the h2o machine learning plataform.
h2o.no_progress()
##h2o.show_progress()

nf = data.shape[1] - 1  # número de features (componentes)
nc = nf  # número de componetes (features) a utilizar
col = list(range(nc))

per_testing_set = 15  # Percentage of the testing rate.
prop_testing_set = per_testing_set / 100
num_testing_samples = int(data.shape[0] * (prop_testing_set))  # Number of obervation used for testing.
print('num_testing_samples')

k = 1
acc_max = 0
for w in range(1, 200):
    for j in range(w + 1, w + 10):
        print('&&&&&&&& w=', w, 'j=', j, '&&&&&&&&&&&&&&&&&&&&&&&&&&&& Execution time: ', time.time() - start_time)
        for prop in [0.1]:
            if fbl >= 1: print('prop=', prop)
            for Eps in [0.1, 0.2, 0.3, 0.4, 0.5]:  # std deviation of the Gaussian Noise.
                if (fbl == 1): print('Eps=', Eps)
                random.seed(w)
                rng = np.random.RandomState(j)
                acc_acu = 0
                test_ctl_pd = data  # initialize test_ctl_pd with all the users... in each iteration test_ctl_pd file will be deletting the testing files.
                i = 0
                N = int(data.shape[
                            0] / num_testing_samples)  # The number of iterations for co-evaluation depend on the num_obs (number of observations used for testing)
                print('N=', N, 'data.shape[0]', data.shape[0], 'num_testing_samples=', num_testing_samples)
                for n in range(0, N):
                    if fbl >= 1: print(i, '.', end='')
                    i += 1
                    start_time2 = time.time()
                    train, test, test_ctl_pd = train_test_LOOnd(data, test_ctl_pd, n, num_testing_samples)

                    if fbl >= 1: print('train.shape[0]', train.shape[0])

                    train = make_dataframes_classes_same_length(train, Eps, prop)
                    e0 = train[(train['Stress_Expression'] == 0)]  # data no presenting the emotion.
                    e1 = train[(train['Stress_Expression'] == 1)]  # data presenting the emotion.
                    e2 = train[(train['Stress_Expression'] == 2)]  # data presenting the emotion.
                    l_e0 = len(e0)
                    l_e1 = len(e1)
                    l_e2 = len(e2)
                    if fbl >= 1: print('train.shape[0]', train.shape[0], 'no=', l_e0, 'moderate=', l_e1, 'high=',
                                       l_e2, )
                    train = agument_Data(train, Eps, prop)
                    e0 = train[(train['Stress_Expression'] == 0)]  # data no presenting the emotion.
                    e1 = train[(train['Stress_Expression'] == 1)]  # data presenting the emotion.
                    e2 = train[(train['Stress_Expression'] == 2)]  # data presenting the emotion.
                    l_e0 = len(e0)
                    l_e1 = len(e1)
                    l_e2 = len(e2)
                    if fbl >= 1: print('train.shape[0]', train.shape[0], 'no=', l_e0, 'moderate=', l_e1, 'high=',
                                       l_e2, )

                    test.loc[(test.Stress_Expression) == 0, 'Stress_Expression'] = 'no'
                    test.loc[(
                                 test.Stress_Expression) == 1, 'Stress_Expression'] = 'moderate'  # this assignment was needed, when the dataframe is convert to h2o dataframes.
                    test.loc[(
                                 test.Stress_Expression) == 2, 'Stress_Expression'] = 'high'  # this assignment was needed, when the dataframe is convert to h2o dataframes.
                    train.loc[(train.Stress_Expression) == 0, 'Stress_Expression'] = 'no'
                    train.loc[(
                                  train.Stress_Expression) == 1, 'Stress_Expression'] = 'moderate'  # this assignment was needed, when the dataframe is convert to h2o dataframes.
                    train.loc[(
                                  train.Stress_Expression) == 2, 'Stress_Expression'] = 'high'  # this assignment was needed, when the dataframe is convert to h2o dataframes.
                    train = train.reset_index(drop=True)

                    train_shuffled = train.iloc[np.random.permutation(train.index)].reset_index(
                        drop=True)  # Randomly Shuffle DataFrame Rows to avoid one of the folds only contains one category.

                    test_h2o = h2o.H2OFrame(test)  # convert panda dataframe to h20 dataframe
                    train_h2o = h2o.H2OFrame(train_shuffled)  # convert panda dataframe to h20 dataframe

                    print('time.time()-start_time=', time.time() - start_time2)

                    # Identify predictors and response
                    x = train_h2o.columns
                    y = "Stress_Expression"
                    x.remove(y)

                    # model definition
                    aml = H2OAutoML(max_runtime_secs=200,
                                    max_models=15,
                                    exclude_algos=['GBM'],
                                    seed=1,
                                    nfolds=2,
                                    balance_classes=True
                                    # to minize the categories balance problem . ¿balance_classes=True, class_sampling_factors=[1.0,1.0,2.5]?
                                    # sort_metric='mean_per_class_error',
                                    # stopping_metric="logloss", # logloss (this is the devalut for Classification), MSE, MAE; RMSLE, AUC, AUCPR, lift_top_group, misClassification, maean_per_Class_error
                                    # sort_metric='auc',    # mse, MAE; RMSLE, auc, aucpr, lift_top_group, misclasification, maean_per_Class_error
                                    # project_name='Completed'
                                    )

                    # model trining
                    # %time aml.train(x=x, y=y, training_frame=train)
                    aml.train(x=x, y=y, training_frame=train_h2o)

                    lb = aml.leaderboard
                    if fbl >= 1: print('lb.head', lb.head(rows=lb.nrows))  # Print all rows instead of default (10 rows)

                    # print('i=',i,'test=',test)
                    model = h2o.get_model(lb[0, 0])  # Select the leader model to test.

                    cma = model.model_performance(test_h2o).confusion_matrix()
                    # print('cma=',cma)
                    true_positives = (cma['high'][0] + cma['moderate'][1] + cma['no'][2])
                    acc = true_positives / test.shape[0]
                    acc_acu += acc
                    acc_m = acc_acu / i
                    # print('****** w=',w,'j=',j,'n=',n,'prop=',prop,'Eps=',Eps,'***********', 'acc',acc,'acc_m',acc_m, 'lb[0,0]=',lb[0,0], 'Execution time: ',time.time()-start_time)
                acc_m = acc_acu / (n + 1)
                container = container + [acc_m]
                print("indicator", max(container))
                print(' ')
                # print('******###################### w=',w,'j=',j,'prop=',prop,'Eps=',Eps,'***********','acc_m=',acc_m,'Execution time: ', time.time()-start_time)
                string_to_write = '****** w=' + str(w) + 'j=' + str(j) + 'prop=' + str(prop) + 'Eps=' + str(
                    Eps) + "average accuracy=" + str(acc_m) + '**********' 'Execution time: ' + str(
                    time.time() - start_time) + '\n'
                # results_file.write(string_to_write)

                # View the AutoML Leaderboard
                # print('lb.head',lb.head(rows=lb.nrows))  # Print all rows instead of default (10 rows)
                # Get the "All Models" Stacked Ensemble model
                # se = aml.leader
                # Get the Stacked Ensemble metalearner model
                # metalearner = h2o.get_model(se.metalearner()['name']) # the goal of the meta learner is to ignore what is not important.
results_file.close()

import pandas as pd
import matplotlib.pyplot as plt

var_imp = model.varimp(use_pandas=True).head(15)
var_imp.plot(kind='bar', x='variable', y='relative_importance')
plt.ylabel('Relative Importance')
plt.show

path = "/content/drive/My Drive/Computadora Virtual/Data Analytics/Investigacion/Parkinson/"
file = path + 'Results/6.9. Feature Relative Importance.svg'  # https://stackoverflow.com/questions/7608066/in-matplotlib-is-there-a-way-to-know-the-list-of-available-output-format
plt.savefig(file)