from TransformPipeler import *


def show_sub_dfs(sub_dfs, sub_test_dfs):
    for idx, df in enumerate(sub_dfs):
        show_info_df(df, name="df {}".format(idx))

    for idx, df_test in enumerate(sub_test_dfs):
        show_info_df(df_test, name="df_test {}".format(idx))    # df_test


LOCK = True


def perform_transform_pipeline(sub_dfs, sub_test_dfs, step_list):
    if (LOCK):
        return

    for idx, df in enumerate(sub_dfs):
        step_list, x, y = get_transform_pipeline_data(
            step_list=step_list, x=df, use_fit=True)

    for idx, df_test in enumerate(sub_test_dfs):
        step_list, x_test, y_test = get_transform_pipeline_data(
            step_list=step_list, x=df_test, use_fit=True)


if (__name__ == "__main__"):
    if (LOCK):
        assert (1 == 0)
    current_folder = os.getcwd()
    ut.note_verbose(True, f"current_folder: {current_folder}")

    # get dataframe
    df = get_dataframe(
        os.path.join(current_folder, 'Dataset', 'Training'), 0.999)
    df_test = get_dataframe(
        os.path.join(current_folder, 'Dataset', 'Testing'), 0.999)

    # split the DataFrame into sub-DataFrames of K rows each
    K = 100
    sub_dfs = [df.iloc[i:i+K] for i in range(0, len(df), K)]
    sub_test_dfs = [df_test.iloc[i:i+K] for i in range(0, len(df_test), K)]

    # try show
    SHOW = False
    if (SHOW):
        show_sub_dfs(sub_dfs=sub_dfs, sub_test_dfs=sub_test_dfs)

    print("df: \n", df)
    print("df_test: \n", df_test)
    print("len of (df, df_test): ", len(sub_dfs), len(sub_test_dfs))
    df.to_csv('df.csv')
    df_test.to_csv('df_test.csv')


def store_y_data(sub_dfs):
    for idx, df in enumerate(sub_dfs):
        # get numpy data
        y_data = []
        for i in range(len(df)):
            folder = df['folder'].iloc[i]
            y = LABEL.index(folder)
            y_data.append(y)
        y_data = np.array(y_data)

        # save numpy data
        current_folder = os.getcwd()
        Const.CNT_EXTRACT += 1
        sub_folder = os.path.join(
            current_folder, "sub_folder_{}".format(Const.CNT_EXTRACT))
        np_file_path = os.path.join(sub_folder, "y.npy")
        with open(np_file_path, "wb") as f:
            np.save(f, y_data)
            ut.mess("saved y data to: {}".format(Const.CNT_EXTRACT))
            ut.over(y_data)
            ut.mess(np.sum(y_data == 0), np.sum(y_data == 1),
                    np.sum(y_data == 2), np.sum(y_data == 3))


if (__name__ == "__main__"):
    if (LOCK):
        assert (1 == 0)
    # store y data
    Const.CNT_EXTRACT = -1
    store_y_data(sub_dfs=sub_dfs)
    store_y_data(sub_dfs=sub_test_dfs)
