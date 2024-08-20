from FileDataFrame import *


def save_np(path, arr):
    with open(path, 'wb') as f:
        np.save(f, arr=arr)


putall = Putall()
current_folder = os.getcwd()


def split_get_dataframe(folder, rate):
    df = get_dataframe(
        os.path.join(current_folder, folder, 'Training'), rate=rate)
    df_test = get_dataframe(
        os.path.join(current_folder, folder, 'Testing'), rate=rate)

    # Split the DataFrame into sub-DataFrames of K rows each
    K = 100
    sub_dfs = [df.iloc[i:i+K] for i in range(0, len(df), K)]
    sub_test_dfs = [df_test.iloc[i:i+K] for i in range(0, len(df_test), K)]
    return sub_dfs, sub_test_dfs


def store_X_Y_disk(sub_dfs, folderX, folderY):
    x_list = []
    y_list = []
    for df in sub_dfs:
        x_train, y_train = putall.call(df)
        for idx in range(len(df)):  # loop through each row in the DataFrame
            x = x_train[idx]
            y = y_train[idx]

            name = df['folder'].iloc[idx] + '_' + df['file_name'].iloc[idx]
            name = name.replace(' ', '')
            name = name.replace('jpg', 'npy')

            save_np(os.path.join(current_folder, folderX, name), x)
            save_np(os.path.join(current_folder, folderY, name), y)

            x_list.append(os.path.join(current_folder, folderX, name))
            y_list.append(os.path.join(current_folder, folderY, name))
    return x_list, y_list


def store_X_Y_csv(sub_dfs, folderX, folderY, name_file):
    x_list, y_list = store_X_Y_disk(sub_dfs, folderX, folderY)
    df = pd.DataFrame({'X': x_list, 'Y': y_list})
    df.to_csv(os.path.join(current_folder, name_file), index=True)


if (__name__ == "__main__"):
    sub_dfs, sub_test_dfs = split_get_dataframe(
        folder=Const.FOLDER_DATASET, rate=0.999)

    store_X_Y_csv(sub_dfs, 'X', 'Y', 'X_Y_train.csv')
    store_X_Y_csv(sub_test_dfs, 'X_test', 'Y_test', 'X_Y_test.csv')
