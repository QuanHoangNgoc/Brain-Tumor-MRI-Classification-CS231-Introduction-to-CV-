from Transformer import *


def get_transform_pipeline_data(step_list, x, y=None, use_fit=False):
    # x, y = df.copy(), None
    for i in range(len(step_list)):
        ut.over(x, "X in loop {}".format(i))
        if (use_fit):
            step_list[i].fit(x)
        z = step_list[i].transform(x)
        if (isinstance(z, tuple)):
            x, y = z
        else:
            x = z
    return step_list, x, y


# _dt = DataTrainer()
# _scaler1 = get_ins_scaler()
# _scaler2 = get_ins_scaler()
# _pca = PCACustomizer()
# _step_list = [_dt, _scaler1, _pca, _scaler2]
