"""
main.py - Reinforcement Learning Algorithm for Exosuit

Versions:
    v1 - junmin modified, first implementation of the offline RL algorithm
    v2 - emiliano modified to simulate the real-time implementation
    v3 - emiliano modified, incorporate mocap data features in the training (from gait_data_extraction_v5.py)  
    v4 - junmin modified, tuned training parameters
    v5 - emiliano modified. implemented the realtime online RL
    v6 - junmin modified (6/19), tuned the training parameters, improved the code and cost function.  emiliano modified (6/20) for debugging and added sending the RL action to the exosuit/plant
    v7 - used for the online RL experiments of June 22nd

"""

import numpy as np
import csv
from numpy import genfromtxt
from DHDP import DHDP
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import utils
import torch
#ZMQ Communication Libraries:
import zmq


plt.close('all')  #close all plots

#Remove All Variables in Spyder Console:
from sys import exit #for debug: adds exit() function to spyder
try:
    from IPython import get_ipython
    get_ipython().magic('reset -f') #clear memory/variables
    get_ipython().magic('clear')  #clear screen
except:
    pass



def setup_communication(ip, port, rcv_or_send):
    #note: rcv_or_send is a char that specifies if the port is to send data or receive data
    
    #Setup for sending data
    if (rcv_or_send == "receive"):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.CONFLATE,True) #read latest rcvd msg only 
        socket.connect(f"tcp://{ip}:{port}")  # Replace with the appropriate address and port
        return socket
    
    #Setup for receiving data
    elif (rcv_or_send == "send"):
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.CONFLATE,True) #keep latest sent msg only
        socket.bind(f"tcp://{ip}:{port}")
        return socket

#Setup communication to receive data from gait_timing.py
ip_0 = "localhost"  #local IP
port_0 = "3333"
socket0 = setup_communication(ip_0, port_0, 'receive')

#Socket to export/send the RL action (timing), to the pressure_commands.py
ip_export = "*"  #localhost IP
port_export = "4444" 
socket_export = setup_communication(ip_export, port_export, "send")



""" Select Options """
Deploy_RL = 0
Train_RL = 1
Realtime_Run = True  #(T/F). (True)= Run Realtime/online training. (False)= use offline data



if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--MAX_EPISODES", default=50, type=int)    # Max EP to run environment
    parser.add_argument("--expl_noise", default=0.05)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=2, type=int)        # Batch size for both actor and critic
    parser.add_argument("--max_mem_size", default=50, type=int)     # max buffer size
    parser.add_argument("--discount", default=0.95)                 # Discount factor
    parser.add_argument("--tau", default=0.5)                      # Target network update rate
    parser.add_argument("--policy_noise", default=0.00)             # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.1)                # Range to clip target policy noise
    parser.add_argument("--save_model", default = False)             # Save model and optimizer parameters
    parser.add_argument("--load_model", default="dhdp_mocap")          # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    

    # #Load Offline Data
    # # data = genfromtxt(r"C:\Users\jz5zh\eclipse-workspace\soft_suit\gait_data_2control.csv", dtype=float,delimiter=',')
    # data = genfromtxt(r"gait_data_1.csv", dtype=float,delimiter=',')
    #   # 0:Total gait time
    #   # 1: period time stf to ste
    #   # 2: control 1
    #   # 5: normalized control [-1 1]
    # print(data.shape[0],'total data length') #row number
    
    #Setup folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
        
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    
    # set RL env
    state_dim = 5
    action_dim = 4
    max_action = 1
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # set RL method (continuous)
    policy = DHDP(**kwargs)
    

    """ ----------- TRAINING RL Loop (Simulates Real-Time Implementation) ----------- """
    if Train_RL:
        
        # #Load a trained policy
        policy_file = "dhdp_6_22_percent"
        policy.load(f"./models/{policy_file}")
        print("Model Loaded: ", f"./models/{policy_file}")
        
        #Load Offline Data
        if not Realtime_Run:
            data_csv = genfromtxt(r"gait_data/gait_data_v6.csv", dtype=float,delimiter=',')
            data_mocap_csv = genfromtxt(r"gait_data/gait_data_v5_mocap.csv", dtype=float,delimiter=',')
            print(data_csv.shape[0],'total data length') #row number
            print(data_csv.shape[0],'total mocap data length') #row number
        
        
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim,args.max_mem_size)
        training_data = []
        counter_samples = 0
        run_exp = True
        print("---------------------------------------")
        print(f" Start RL Training")
        print("---------------------------------------")
        
        #Get initial Gait Data, this we can get from warm up session by assign initial data to mocap average
        if Realtime_Run:
            #Receive Gait Data Array
            data = socket0.recv_pyobj()
        else:
            #Read Gait Data Array from data structure
            data = data_csv[0][4:8]
            data_mocap = data_mocap_csv[0][4:8]
            
        
        # state = (data[0],data[2]) #first gait state data
        state = (data[0],data[1]/2,data[2],data[3],data[4]) #first gait state data
        
        #Run training loop for the number of max episodes. One episode is one run of the whole data set.
        # for j in range(int(args.MAX_EPISODES)): #no need for
        while run_exp:
        # for j in range(1):
            try:
                #Compute RL action
                action = (
                        policy.select_action(np.array(state))  #policy from state
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)  #add some noise
                            ).clip(-max_action, max_action)
                
                """ Send action data to pressure controller """
                #Convert the action from normalized values to seconds (junmin)
                #Junmin add here the denormalization (a + 1)*(max - min)/2 + min
                gaie_duration = data[8]
                t1 = (((action[0] + 1)*(0.2-0)/2) + 0)*gaie_duration
                u1 = (((action[1] + 1)*(0.25)/2) + 0)*gaie_duration
                t2 = (((action[2] + 1)*(0.74-0.5)/2) + 0.5)*gaie_duration
                u2 = (((action[3] + 1)*(0.28-0)/2) + 0)*gaie_duration 
                
                #Send the RL timings (action) to the pressure controller (emiliano)
                timing_export_array = np.array([t1, u1, t2, u2])
                socket_export.send_pyobj(timing_export_array) #send data to pressure_commands.py
                print("action timings (sec) =", timing_export_array)
                
                
                """ RL """
                ###double check this read data will wait 5 steps average?
                ###apply action to plant
                if Realtime_Run:
                    #Receive Gait Data Array
                    data = socket0.recv_pyobj()
                else:
                    #Read Gait Data Array from data structure
                    data = data_csv[i][4:8]
                    data_mocap = data_mocap_csv[i][4:8]
                
                # next_state = (data[0],data[2]) #first gait state data
                emg_state = data[5]+data[6]+data[7]
                next_state = (data[0],data[1]/2,data[2],data[3],data[4])
                
                ####in online training we should compare real human data with target. NOT ACTION anymore.
                #JUNMIN 0.232 0.523 1.118 0.289 1.095
                #juseph 0.231 0.684 1.326 0.311 0.698
                t1_error = data[0] - 0.124 ####t1 12.4%
                d1_error = data[1]/2 - 0.375/2 ####d1 
                t2_error = data[2] - 0.71
                d2_error = data[3] - 0.168
                kinematic_error = data[4]- 0.698
                emg_VL = data[5]
                emg_RF = data[6]
                emg_BF = data[7]
                ####state action alignment
                action1_error =  (2*(data[0])/(0.2) - 1) - action[0]
                action2_error =  (2*(data[1]/2)/(0.25) - 1) - action[1]
                action3_error =  (2*(data[2]-0.5)/(0.24) - 1) - action[2]
                action4_error =  (2*(data[3])/(0.28) - 1) - action[3]
                
                coeff_t1 = 20
                coeff_d1 = 20
                coeff_t2 = 10
                coeff_d2 = 20
                coeff_a1 = 2
                coeff_a2 = 2
                coeff_a3 = 1
                coeff_a4 = 2
                coeff_kine = 0.05
                coeff_emg = 0.2
                
                error_state = coeff_t1*(t1_error)**2 + coeff_d1*(d1_error)**2 + coeff_t2*(t2_error)**2 + coeff_d2*(d2_error)**2
                error_action = coeff_a1*(action1_error)**2 + coeff_a2*(action2_error)**2 + coeff_a3*(action3_error)**2 + coeff_a4*(action4_error)**2
                
                cost = 0.5*(error_state + error_action
                           + coeff_kine*(kinematic_error)**2 
                           + coeff_emg*(emg_state)**2) #cost for timing and duration
                
                done = 0
                counter_samples += 1  #add to counter
                replay_buffer.add(state, action, next_state, cost, done)
            
                if counter_samples >= 2:  #collect at least 10 samples of data before training
                    
                    policy.train(replay_buffer, args.batch_size) #training
                
                training_data.append((counter_samples,cost,t1_error,d1_error,t2_error,d2_error,kinematic_error,state,action,next_state,emg_VL,emg_RF,emg_BF))  
                state = next_state
                
                #Print results for current sample training
                print('Sample=', counter_samples, 
                       '\t Cost=', cost)
                
                print('\t t1=', t1/gaie_duration,
                      '\t d1=', u1/gaie_duration,
                      '\t t2=', t2/gaie_duration,
                      '\t d2=', u2/gaie_duration)
                
                print('\t t1 error=', t1_error,
                        '\t d1 error=', d1_error,
                        '\t t2 error=', t2_error,
                        '\t d2 error=', d2_error)
                
                print('\t a1 error=', action1_error,
                        '\t a2 error=', action2_error,
                        '\t a3 error=', action3_error,
                        '\t a4 error=', action4_error)
                
                print('emg = ', emg_state)
                
                if counter_samples > 45:
                    run_exp = False
                    
            except KeyboardInterrupt:
                if Realtime_Run: 
                    socket0.close()
                    socket_export.close()
                    print("---------------------------------------")
                    print('Connection to Socket Closed.')
                    run_exp = False
                    break
                # exit()
                break    


            
            
            
        #Export Training Data
        filename = "results/dhdp_online_7_12_trial1.csv"
        fields = ['steps','training cost','t1 error','d1 error','t2 error','d2 error','kinematic error','state','action','next state','emg_vl_ext1','emg_rf_ext2','emg_bf_flex']
        with open(filename, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile,lineterminator = '\n') 
            # writing the fields 
            csvwriter.writerow(fields) 
            # writing the data rows 
            csvwriter.writerows(training_data)   
            print("Training data saved to: ", filename)     

        #Export Policy Model
        export_filename = "dhdp_online_7_12_trial1"
        policy.save(f"./models/{export_filename}")
        print('Policy files saved to:', f"./models/{export_filename}")
        
            
        print("RL TRAINING Fininshed.")
        if Realtime_Run: 
            socket0.close()
            socket_export.close()
        exit()
            
        
        

