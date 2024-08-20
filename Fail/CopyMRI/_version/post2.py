from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.model_selection import train_test_split
from TransformPipeler import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# arr = np.load("D:\cd_data_C\Desktop\git_upload\sub_folder_0\e_image.npy")
# print(arr)
CHOOSE = 'hog_fd.npy'
TRANSFORM = True


def select_df():
    fdf = FileDataFrame()
    current_folder = os.getcwd()
    df_folder = fdf.call(current_folder)
    df_folder = df_folder.loc[df_folder['file_name'] ==
                              CHOOSE].copy().reset_index(drop=True)
    return df_folder


LIM = 100


def get_data(df):
    x_data, y_data = [], []
    for i in range(min(len(df), LIM)):
        x_path = df['file_path'].iloc[i]
        y_path = df['file_path'].iloc[i].replace(CHOOSE, 'y.npy')
        try:
            x = np.load(x_path)
            y = np.load(y_path)
            if (isinstance(x_data, list)):
                x_data, y_data = x.copy(), y.copy()
            else:
                x_data, y_data = np.concatenate(
                    (x_data, x), axis=0), np.concatenate((y_data, y), axis=0)
        except Exception as e:
            ut.mess("x_path: ", x_path)
            ut.mess("y_path: ", y_path)
            ut.mess("error", e)

    # ut.over(x_data, "x_data")
    # ut.mess(sys.getsizeof(x_data), "Bytes")
    # ut.over(y_data, "y_data")
    x_data = x_data.reshape(len(x_data), -1)
    return x_data, y_data


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


def show_distribution(x_train, y_train, x_val, y_val, x_test, y_test):
    ut.over(x_train, "x_train")
    ut.over(y_train, "y_train")
    ut.over(x_val, "x_val")
    ut.over(y_val, "y_val")
    ut.over(x_test, "x_test")
    ut.over(y_test, "y_test")


def select_df_train_test(df):
    test_folders = ['sub_folder_32', 'sub_folder_29',
                    'sub_folder_30', 'sub_folder_31']
    mask = map(lambda x: x not in test_folders, df['folder'])
    mask = [x for x in mask]
    df_train = df.loc[mask].copy().reset_index(drop=True)
    mask = [not x for x in mask]
    df_test = df.loc[mask].copy().reset_index(drop=True)
    return df_train, df_test


def get_cross_valid(hypo_model, x, y, scorer, len_test) -> KNeighborsClassifier:
    CV = int(len(x)/len_test)
    results = cross_val_score(hypo_model, x, y, cv=CV, scoring=scorer)
    trained_model = skl.base.clone(hypo_model)
    trained_model.fit(x, y)
    return results, trained_model


# def get_custom_score(y_true, y_pred):
#     return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='weighted')


if (__name__ == "__main__"):
    # Select df folder
    df_folder = select_df()
    df_train, df_test = select_df_train_test(df_folder)
    print(df_train)
    print(df_test)

    # Get data from df folder
    x_train, y_train = get_data(df_train)
    x_test, y_test = get_data(df_test)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)

    show_distribution(x_train, y_train, x_val, y_val, x_test, y_test)

    # Transform (if need)
    if (TRANSFORM):
        step_list = [get_ins_scaler(), PCA(
            n_components=0.95), get_ins_scaler()]
        step_list, x_train, y_train = get_transform_pipeline_data(
            step_list, x_train, y_train, use_fit=True)
        step_list, x_val, y_val = get_transform_pipeline_data(
            step_list, x_val, y_val
        )
        step_list, x_test, y_test = get_transform_pipeline_data(
            step_list, x_test, y_test)

        # Show distribution
        show_distribution(x_train, y_train, x_val, y_val, x_test, y_test)

    # Hypothesis
    short_dim = True
    models = [('linearSVM', LinearSVC(random_state=42, C=10, dual=True,
                                      max_iter=1000, verbose=2))]
    models += [('KNN', KNeighborsClassifier(n_neighbors=500, weights='distance'))]
    models += [('DT', DecisionTreeClassifier(random_state=42, max_depth=10))]
    models += [('RF', RandomForestClassifier(n_estimators=100,
                                             random_state=42, verbose=0))]
    models += [('Boost', GradientBoostingClassifier(random_state=42, verbose=0))]

    # Hypo Testing
    for name, hypo in models:
        if (name != 'linearSVM'):
            continue

        ut.mess("training... of {}".format(name))
        x, y = np.concatenate((x_train, x_val), axis=0), np.concatenate(
            (y_train, y_val), axis=0)
        scorer = make_scorer(accuracy_score)
        results, trained_model = get_cross_valid(
            hypo, x, y, scorer, len(x_test))
        try:
            ut.over(trained_model.coef_)
        except:
            pass

        # Overal Seen data
        ut.mess("evaluating... of {}".format(name))
        print("results: ", results)
        import math
        print("mean of: ", np.array(results).mean())
        print("std2 of: ", np.array(results).std()**2)

        # Notseen data
        y_pred = trained_model.predict(x_test)
        print("test resutl: ", scorer(trained_model, x_test, y_test),
              f1_score(y_true=y_test, y_pred=y_pred, average='weighted'))
        show_heatmap(trained_model, x_test, y_test)
