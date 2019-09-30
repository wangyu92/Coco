import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import tensorflow as tf
import a3c as a3c
import env

S_INFO = 5
S_LEN = 30
A_DIM = 30
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001

RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './test_results/log_sim_rl'
TEST_TRACES = '../dataset/cooked_test/'

# NN_MODEL = sys.argv[1]

def main(NN_MODEL):
    np.random.seed(RANDOM_SEED)

    net_env = env.Environment(num_of_cluster_high=S_LEN, trace_dir=TEST_TRACES)
    
    log_path = LOG_FILE + '_' + net_env.selected_trace_files[0].split('/')[-1]
    log_file = open(log_path, 'wb')

    with tf.compat.v1.Session() as sess:

        actor = a3c.ActorNetwork(sess, state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess, state_dim=[S_INFO, S_LEN], learning_rate=CRITIC_LR_RATE)

        sess.run(tf.compat.v1.global_variables_initializer())
        
        # 여기서 문제가 발생함.
        saver = tf.compat.v1.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        time_stamp = 0

        s_batch = []
        a_batch = []
        r_batch = []
        entropy_record = []

        video_count = 0
        
        t.write('before while\n')
        t.flush()

        while True:
            rembs = net_env.get_remb_of_cluster_head()
            
            if len(s_batch) == 0:
                state = np.zeros((S_INFO, S_LEN))
            else:
                state = np.array(s_batch[-1], copy=True)
            
            cpu_usage = 0
            bandwidth = 10000000
            source_bitrate = 10000
            
            remain_size = A_DIM - len(rembs)
            n_clients = net_env.num_of_client_in_each_cluster
            state[0, :] = np.pad(rembs, (0, remain_size), 'constant')
            state[1, :] = np.pad(n_clients, (0, remain_size), 'constant')
            state[2, -1] = cpu_usage # cpu
            state[3, -1] = bandwidth # bandwidth
            state[4, -1] = source_bitrate # source video
            
            action = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action.clip(min=0)
            listofbitrates = action.flatten().tolist()
            listofbitrates = listofbitrates[:len(rembs)]
                
            reward, end_of_video = net_env.set_bitrate_of_streams(listofbitrates)
            time_stamp += 1
            
            # store the state and action into batches
            s_batch.append(state)

            action_vec = action
            a_batch.append(action_vec)
            r_batch.append(reward)
            
            entropy_record.append(a3c.compute_entropy(action[0]))
            
            
            msg = str(time_stamp) + '\t' + str(rembs) + '\t' + str(listofbitrates) + '\t' + str(cpu_usage) + '\t' + str(bandwidth) + '\t' + str(source_bitrate) + '\t' + str(reward) + '\n'
            log_file.write(msg.encode())
            log_file.flush()
            
            if end_of_video:
                log_file.write('\n'.encode())
                log_file.close()
                
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                
                video_count += 1
                
                if video_count >= len(net_env.trace_files_all):
                    break

                log_path = LOG_FILE + '_' + net_env.selected_traces
                log_file = open(log_path, 'wb')

if __name__ == '__main__':
    main()