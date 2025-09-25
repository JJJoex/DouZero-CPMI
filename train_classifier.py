import pickle
from douzero.cpmi.utils import estimate_cpmi, train_classifer, construct_batch
import torch

if __name__ == '__main__':
    with open('cmi_joint_data_10000_10.pkl', 'rb') as f:
        cmi_joint_data = pickle.load(f)

    batch_train, target_train, batch_test, target_test = construct_batch(cmi_joint_data, set_size=50000, neighbor_size=3)
    with open('cpmi_data_10_7.pkl', 'wb') as f:
        pickle.dump((batch_train, target_train, batch_test, target_test), f)
 
    with open('cpmi_data_10_7.pkl', 'rb') as f:
        batch_train, target_train, batch_test, target_test = pickle.load(f)
    model = train_classifer(batch_train, target_train, batch_test, target_test)
    torch.save(model.state_dict(), 'cpmi_10_7.pth')
    # pass