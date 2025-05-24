"""
main.py

Versions:
    v1 - junming modified, first implementation of the offline RL algorithm
    v2 - emiliano modified to simulate the real-time implementation
    v3 - emiliano modified, incorporate mocap data features in the training (from gait_data_extraction_v5.py)   

"""

import numpy as np
import csv
from numpy import genfromtxt
from DHDP import DHDP
import numpy as np
import argparse
import os
import utils
import torch



""" Select Options """
Deploy_RL = 0
Train_RL = 1 if Deploy_RL == 0 else 0



if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--MAX_EPISODES", default=20, type=int)    # Max EP to run environment
    parser.add_argument("--expl_noise", default=0.05)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)        # Batch size for both actor and critic
    parser.add_argument("--max_mem_size", default=200, type=int)     # max buffer size
    parser.add_argument("--discount", default=0.95)                 # Discount factor
    parser.add_argument("--tau", default=0.5)                      # Target network update rate
    parser.add_argument("--policy_noise", default=0.0)             # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.1)                # Range to clip target policy noise
    parser.add_argument("--save_model", default = True)             # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")          # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    

    
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
    
    
    
    
    """ ----------- DEPLOYMENT RL Loop (Simulates Real-Time Implementation) ----------- """
    if Deploy_RL:
         
        #Load trained Policy
        policy_file = "dhdp_6_22_percent"
        policy.load(f"./models/{policy_file}")
        print("Model Loaded: ", f"./models/{policy_file}")
         
        
        data_csv = genfromtxt(r"gait_data_6_22_percent.csv", dtype=float,delimiter=',')
        data_mocap_csv = genfromtxt(r"gait_data_6_22_percent_mocap.csv", dtype=float,delimiter=',')
        print(data_csv.shape[0],'total data length') #row number
        print(data_mocap_csv.shape[0],'total mocap data length') #row number
         
         
        valid_data = []
        ep_c = 0  #cost sum
         
        print("---------------------------------------")
        print(f"start RL deployment/validation")
        print("---------------------------------------")
         
         
        # while True:
        for i in range(data_mocap_csv.shape[0] - 2):  #alternative loop statement. 214 = number of gait cycles in the offline data
 
            
            data = data_csv[i][0:5]
            data_mocap = data_mocap_csv[i][5:10]
 
             
            # state = (data[0],data[1],data[3])
            state = (data[0],data[1],data[2],data[3],data[4])
             
            action = policy.select_action(np.array(state))
             
            # cost = 0.5*((action[0] - data[4])**2 + (action[1]- data[5])**2) #cost between two duration
            cost = 0.5*((action[0] - data_mocap[0])**2 + (action[1]- data_mocap[1])**2 + 2*(action[2]- data_mocap[2])**2 + (action[3]- data_mocap[3])**2) #cost for timing and duration
             
            valid_data.append((cost,action[0],action[1],action[2],action[3]))
            done = 0
            ep_c += cost
             
            print(state[0:4])
            print('T1',((action[0] + 1)*(0.2-0)/2) + 0 ,'D1',((action[1] + 1)*(0.5-0.25)/2) + 0.25, 'T2',((action[2] + 1)*(0.74-0.5)/2) + 0.5 , 'D2',((action[3] + 1)*(0.45-0)/2) + 0 )
            
                 
                 
        #Export Validation (deployment) Results
        filename = "results/valid_mocap.csv"
        with open(filename, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile,lineterminator = '\n') 
         
            # writing the data rows 
            csvwriter.writerows(valid_data)   
            print("Validation/Deployment data saved to: ", filename)
         
         
        print("RL Deployment Fininshed.")

         
           
     


    """ ----------- TRAINING RL Loop (Simulates Real-Time Implementation) ----------- """
    if Train_RL:
        
        
        data_csv = genfromtxt(r"gait_data_6_22_percent.csv", dtype=float,delimiter=',')
        data_mocap_csv = genfromtxt(r"gait_data_6_22_percent_mocap.csv", dtype=float,delimiter=',')
        print(data_csv.shape[0],'total data length') #row number
        print(data_mocap_csv.shape[0],'total mocap data length') #row number
        
        
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim,args.max_mem_size)
        training_data = []
        all_epc = []

        
        print("---------------------------------------")
        print(f" Start RL Training")
        print("---------------------------------------")
        
        #Run training loop for the number of max episodes. One episode is one run of the whole data set.
        for j in range(int(args.MAX_EPISODES)):
        # for j in range(1):
            ep_c = 0
            counter_samples = 0
            

            data = data_csv[0][0:5]
                
            
            # state = (data[0],data[2]) #first gait state data
            state = (data[0],data[1],data[2],data[3],data[4]) #first gait state data
            
            # while True:
            num_data = data_mocap_csv.shape[0]
            for i in range(num_data - 2):  #debug, alternative loop statement. 214 = number of gait cycles in the offline data
                
                #Read Gait Data Array from data structure
                data = data_csv[i+1][0:5]
                data_mocap = data_mocap_csv[i][5:10]
                
                # next_state = (data[0],data[2]) #first gait state data
                next_state = (data[0],data[1],data[2],data[3], data[4])
                
                action = (
                        policy.select_action(np.array(state))  #policy from state
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)  #add some noise
                            ).clip(-max_action, max_action)
                            
                action_raw = policy.select_action(np.array(state))
                

                coeff_t1 = 2
                coeff_d1 = 2
                coeff_t2 = 2
                coeff_d2 = 2
                cost = 0.5*(coeff_t1*(action_raw[0] - data_mocap[0])**2 + coeff_d1*(action_raw[1]- data_mocap[1])**2 
                            + coeff_t2*(action_raw[2]- data_mocap[2])**2 + coeff_d2*(action_raw[3]- data_mocap[3])**2) #cost for timing and duration
                
                done = 0
                ep_c += cost   
                counter_samples += 1  #add to counter
                
                replay_buffer.add(state, action, next_state, cost, done)
                if j > 190:
                    print('T1',((action[0] + 1)*(0.2-0)/2) + 0, 'state T1', state[0] ,'D1',((action[1] + 1)*(0.25-0)/2) + 0, 'state D1', state[1]  )
                    print('T2',((action[2] + 1)*(0.74-0.5)/2) + 0.5,'state T2', state[2] , 'D2',((action[3] + 1)*(0.45-0)/2) + 0, 'state D2', state[3]  )
                    
            
                if counter_samples >= args.batch_size:  #collect at least 10 samples of data before training
                    
                    policy.train(replay_buffer, args.batch_size) #training
                    
                state = next_state
                
                    
                        
            #Print Results for complete episode (1 pass of training data)
            all_epc += [ep_c]
            training_data.append((j,ep_c/num_data))
            print('Ep: %i  | Average step cost: %.1f | ' % (j, ep_c/num_data ))
            
            
        #Export Training Data
        filename = "results/training_6_22_rebuttal.csv"
        fields = ['Episode number','training Episode cost']
        with open(filename, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile,lineterminator = '\n') 
            # writing the fields 
            csvwriter.writerow(fields) 
            # writing the data rows 
            csvwriter.writerows(training_data)   
            print("Training data saved to: ", filename)     

        #Export Policy Model
        export_filename = "dhdp_6_22_percent"
        policy.save(f"./models/{export_filename}")
        print('Policy files saved to:', f"./models/{export_filename}")
        
            
        print("RL TRAINING Fininshed.")
            
        
        

