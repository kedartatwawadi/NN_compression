#!/usr/bin/python

import subprocess
import numpy as np
import os


def main():
    generation_script="generate_lagged_fibonacci.py"
    train_script="char_rnn.py"
    data_dir = "../data/sequence_data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    #output_dir="outputs"
    # generate and run 0entropy models with default parameters
    max_k=100
    num_samples=10000000
    validate_samples=10000
    max_epochs=1
    num_iter=1
    num_layers=2
    p1=0.5
    list_params = [(15,1),(20,3),(25,3),(25,7),(31,13),(35,2),(39,8),(41,20),(49,12),(52,21),(58,19),(63,31),(73,25),(81,16),(95,11)]
    #output_file = os.path.join(output_dir,"output_0entropy_9_popeye.txt")
    for markovity,param in list_params:
        for iter in range(num_iter):
            print "Params: ", markovity,",", param
            file_name = "input_0entropy_" + str(iter) + "_iter_" + str(markovity) + "_markovity_" + str(param) + "_param.txt"
            file_name = os.path.join(data_dir,file_name)
            info_file = "info_0entropy_" + str(iter) + "_iter_" + str(markovity) + "_markovity_" + str(param) + "_param.txt" 
            info_file = os.path.join(data_dir,info_file)
            val_name = "validate_0entropy_" + str(iter) + "_iter_" + str(markovity) + "_markovity_" + str(param) + "_param.txt"
            val_name = os.path.join(data_dir,val_name)
            
            ### Generate validation data first
            arg_string  = "  --num_samples " + str(validate_samples)
            arg_string += "  --data_type "   + "0entropy"
            arg_string += "  --markovity "   + str(markovity)
            arg_string += "  --param "       + str(param)
            arg_string += "  --file_name "   + val_name
            arg_string += "  --info_file "   + info_file    
            arg_string += "  --p1 "          + str(p1)

            generation_command = "python " + generation_script + arg_string
            subprocess.call([generation_command] , shell=True)

            ### Generate the data files
            arg_string  = "  --num_samples " + str(num_samples)
            arg_string += "  --data_type "   + "0entropy"
            arg_string += "  --markovity "   + str(markovity)
            arg_string += "  --param "       + str(param)
            arg_string += "  --file_name "   + file_name
            arg_string += "  --info_file "   + info_file    
            arg_string += "  --p1 "          + str(p1)

            # Generate the data
            generation_command = "python " + generation_script + arg_string
            subprocess.call([generation_command] , shell=True)
            
            assert os.path.isfile(file_name),"The data did not get generated"
            assert os.path.isfile(val_name),"The data did not get generated"
            assert os.path.isfile(info_file),"The info file did not get created"
            print "Data generated .. "


            #### Prepare for training
            for _size in [128]:
                summary_dir = ".summary_azure/big_model/PRNG/"
                summary_dir = os.path.join(summary_dir, "size_" + str(_size))
                summary_dir = os.path.join(summary_dir, "num_layers_" + str(num_layers))
                summary_dir = os.path.join(summary_dir, "markovity_" + str(markovity) + "_" + str(param))
                summary_dir = os.path.join(summary_dir, "run_" + str(iter))
                arg_string  = " --data_path "   + file_name
                arg_string += " --info_path "   + info_file
                arg_string += " --validate_path "    + val_name
                arg_string += " --batch_size " + str(100)
                arg_string += " --num_epochs "  + str(max_epochs)
                arg_string += " --num_layers "  + str(num_layers)
                arg_string += " --hidden_size "    + str(_size)
                arg_string += " --summary_path " + str(summary_dir)
                # Run the training
                train_command = "python " + train_script + arg_string
                subprocess.call([train_command], shell=True) 

if __name__ == '__main__':
    main()

