from TransformPipeler import *
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, f1_score, accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

MAX_LEN = 10**4


def parallel_return_get_data(path):
    try:
        return np.load(path)
    except Exception as e:
        ut.mess("error", path, e)


def get_data(df):
    pics = df['X'].to_list()
    # return [parallel_return_task(path) for path in pics]
    x_data = Parallel(n_jobs=8)(
        delayed(parallel_return_get_data)(path) for path in pics)

    pics = df['Y'].to_list()
    # return [parallel_return_task(path) for path in pics]
    y_data = Parallel(n_jobs=8)(
        delayed(parallel_return_get_data)(path) for path in pics)

    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = x_data.reshape(len(x_data), -1)
    y_data = y_data.reshape(len(y_data))
    return x_data, y_data

# def get_data(df):
#     x_data, y_data = [], []
#     for i in range(min(len(df), MAX_LEN)):
#         x_path = df['X'].iloc[i]
#         y_path = df['Y'].iloc[i]
#         try:
#             x = np.load(x_path)
#             y = np.load(y_path)
#             x = np.array([x])
#             y = np.array([y])
#             # ut.over(x, "x")
#             # ut.over(y, "y")
#             if (isinstance(x_data, list)):
#                 x_data, y_data = x.copy(), y.copy()
#             else:
#                 x_data, y_data = np.concatenate(
#                     (x_data, x), axis=0), np.concatenate((y_data, y), axis=0)
#         except Exception as e:
#             ut.mess("x_path: ", x_path)
#             ut.mess("y_path: ", y_path)
#             ut.mess("error", e)

#     x_data = x_data.reshape(len(x_data), -1)
#     y_data = y_data.reshape(len(y_data))
#     return x_data, y_data


def show_heatmap(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    # Create a heatmap using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    # Add labels and title
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def show_distribution(x_train, y_train, x_test, y_test, x_val=None, y_val=None, when="before"):
    ut.note_verbose(True, "distribution of data... {} transform".format(when))
    for label in range(0, np.max(y_train)+1):
        ut.mess("label {}: ".format(LABEL[label]), sum(
            y_train == label)/len(y_train), sum(y_test == label)/len(y_test))
        x = x_train[y_train == label]
        ut.over(x, "x_train {}".format(LABEL[label]))
        x = x_test[y_test == label]
        ut.over(x, "x_test {}".format(LABEL[label]))


def evaluate_cross_valid(hypo_model, x, y, scorer, len_test):
    CV = int(len(x)/len_test)
    results = cross_val_score(hypo_model, x, y, cv=CV,
                              scoring=scorer)  # results of evaluating
    trained_model = skl.base.clone(hypo_model)  # copy and fit model
    trained_model.fit(x, y)
    return results, trained_model


def get_hypo_models():
    models = [('linearSVM', LinearSVC(C=10, dual=True, random_state=42,
                                      max_iter=1000, verbose=0))]
    models += [('KNN', KNeighborsClassifier(n_neighbors=200, weights='distance'))]
    models += [('DT', DecisionTreeClassifier(max_depth=10, random_state=42))]
    models += [('RF', RandomForestClassifier(n_estimators=100,
                                             random_state=42, verbose=0))]
    return models


def get_transform_pipeline():
    return []
    step_list = [get_ins_scaler(), PCA(
        n_components=0.95), get_ins_scaler()]
    return step_list


def show_evluation(trained_model, x_test, y_test, name):
    y_pred = trained_model.predict(x_test)
    print(name + str(" :"), scorer(trained_model, x_test, y_test),
          f1_score(y_true=y_test, y_pred=y_pred, average='weighted'))


if (__name__ == "__main__"):
    with open('D:\cd_data_C\Desktop\git_upload\MRI\_log\log.txt', 'w') as f:
        f.write('relog')
        f.close()

    df = pd.read_csv(os.path.join(os.getcwd(), 'X_Y_train.csv'))
    df_test = pd.read_csv(os.path.join(os.getcwd(), 'X_Y_test.csv'))
    ut.mess(df, df_test)

    # Get data
    x, y = get_data(df)
    x_test, y_test = get_data(df_test)
    show_distribution(x, y, x_test, y_test)

    # Transform
    step_list = get_transform_pipeline()
    step_list, x, y = get_transform_pipeline_data(
        step_list=step_list, x=x, y=y, use_fit=True)
    step_list, x_test, y_test = get_transform_pipeline_data(
        step_list=step_list, x=x_test, y=y_test, use_fit=False)
    # Show distribution
    show_distribution(x, y, x_test, y_test, when="after")

    # Hypothesis
    models = get_hypo_models()

    # Hypo Testing
    for name, hypo in models:
        if (name not in ['linearSVM', 'KNN']):
            continue

        ut.note_verbose(True, "training... of {}".format(name))
        scorer = make_scorer(accuracy_score)
        results, trained_model = evaluate_cross_valid(
            hypo, x, y, scorer, len(x_test))
        try:
            # Access the coefficients/weights of the model
            weights = trained_model.coef_
            ut.over(weights, "weights of {}".format(name))
        except:
            ut.mess("No weights")

        # Overal Seen data
        ut.note_verbose(True, "evaluating... of {}".format(name))
        print("results: ", results)
        print("mean of: ", np.array(results).mean())
        print("std2 of: ", np.array(results).std()**2)

        # Not seen data
        show_evluation(trained_model, x, y, name="train result")
        show_evluation(trained_model, x_test, y_test, name="test result")
        show_heatmap(trained_model, x_test, y_test)
        show_distribution(x, y, x_test, y_test, when="after")
