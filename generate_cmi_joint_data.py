# pretrain CMI Classifier
import os 
import argparse
import multiprocessing as mp
import pickle
from douzero.evaluation.simulation import data_allocation_per_worker, load_card_play_models
from douzero.env.game import GameEnv
import time

def simulate(card_play_data_list, card_play_model_path_dict, q):
    players = load_card_play_models(card_play_model_path_dict)
    env = GameEnv(players)
    env.cpmi_joint_data = []
    for _, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        env.reset()
        # cpmi joint data (state, last_action, current_action)
        print('len(env.cmi_joint_data):', len(env.cpmi_joint_data))
    q.put(env.cpmi_joint_data)

def data_collect(landlord, landlord_up, landlord_down, eval_data, num_workers):
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down
    }
    
    cmi_data = []

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for card_paly_data in card_play_data_list_each_worker:
        p = ctx.Process(
                target=simulate,
                args=(card_paly_data, card_play_model_path_dict, q))
        processes.append(p)
        p.start()

    for _ in range(num_workers):
        result = q.get()
        cmi_data.extend(result)
    
    for p in processes:
        p.join()

    return cmi_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'generate CMI data')
    parser.add_argument('--landlord', type=str,
            default='douzero_ADP/landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='douzero_ADP/landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='douzero_ADP/landlord_down.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='eval_data_10000.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

   
    start_time = time.time()
    
    cmi_joint_data = data_collect(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.eval_data,
             args.num_workers)
    
    print('length:', len(cmi_joint_data))
    end_time = time.time()
    print('time:', end_time - start_time)

    with open('cmi_joint_data_10000_5.pkl','wb') as g:
        pickle.dump(cmi_joint_data, g, pickle.HIGHEST_PROTOCOL)


