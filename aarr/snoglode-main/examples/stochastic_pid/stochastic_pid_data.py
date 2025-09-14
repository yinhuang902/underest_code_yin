import os
import numpy as np
np.random.seed(42)

# determines how many random realizations to generate/save
num_scenarios = 100

def get_file(file_path):

    # check if file exists/is empty before accidentally overwriting
    if os.path.exists(file_path):
        with open(file_path, 'r') as file_obj:
            first_char = file_obj.read(1)
            if first_char: 
                input("The file is not empty. Press Enter to continue and overwrite...")

    return open(file_path, "w")


if __name__=="__main__":
    data_fname = os.getcwd() + "/stochastic_pid/data.csv"
    file = get_file(data_fname)
    num_disturbances = 21

    header = "tau_xs,tau_us,tau_ds,"
    for i in range(num_disturbances):
        header += f"disturbance_{i},"
    header += "setpoint_change\n"
    file.write(header)

    for _ in range(num_scenarios):
        
        # input data (values taken from Github Julia code in SNGO/examples/PID/pidnonlinear.jl)
        K = float(np.random.uniform(low=1.0, high=5.0, size=1)[0])          # needed to calculate tau_us 
        Kd = float(np.random.uniform(low=0.2, high=0.8, size=1)[0])         # needed to calculate tau_ds
        tau_xs = float(np.random.uniform(low=0.2, high=0.8, size=1)[0])     # model uncertainty - tau_xs
        tau_us = tau_xs*K                                                   # model uncertainty - tau_us
        tau_ds = tau_xs*Kd                                                  # model uncertainty - tau_ds
        disturbance = np.random.uniform(low=-2.5, high=2.5, size=20+1)
        setpoint_change = float(np.random.uniform(low=-2.3, high=2.3, size=1)[0])

        data = str(tau_xs) + "," + str(tau_us) + "," + str(tau_ds) + ","
        for d in disturbance:
            data += str(d) + ","
        data += str(setpoint_change) + "\n"
        file.write(data)
    print("Done.")