from sklearn.model_selection import train_test_split
from Extractor import *


class FileDataFrame(ExtractLayer):
    def __init__(self): super().__init__()

    def list_files_from_start_folder(self, start_folder_path):
        files, info, filenames = [], [], []
        for root, _, file_name_list in os.walk(start_folder_path):
            for filename in file_name_list:
                file_path = os.path.join(root, filename)
                parent_folder_name = os.path.basename(
                    os.path.dirname(file_path))
                files.append(file_path)
                info.append(parent_folder_name)
                filenames.append(filename)
        df = pd.DataFrame(
            {'file_path': files, 'folder': info, 'file_name': filenames})
        return df

    def call(self, start_folder_path):
        return self.list_files_from_start_folder(start_folder_path)


def get_dataframe(root_folder, rate):
    fdf = FileDataFrame()
    df = fdf.call(root_folder)
    if (rate == None):
        return df
    small_df, _, _, _ = train_test_split(
        df, [x for x in range(len(df))], test_size=1-rate, random_state=42)
    return small_df


def show_info_df(df, name=""):
    ut.note_verbose(True, "show df: " + name)
    ut.prt_section()
    ut.mess(df.shape)
    for i in range(len(LABEL)):
        ut.mess(LABEL[i], sum(df['folder'] == LABEL[i]))
    ut.prt_section()


# examples
if __name__ == "__main__":
    root_folder = 'D:\cd_data_C\Desktop\git_upload\Dataset\Training'
    df = get_dataframe(root_folder, 0.2)
    print(df)
    print(df.shape)
    for i in range(len(LABEL)):
        print(LABEL[i], sum(df['folder'] == LABEL[i]))
