import os
import subprocess

def get_periodicities(path, number_of_periods=5, max_periods=60*60*24*7):
    cmd = "./fremen " + str(path) + " " + str(number_of_periods) + " " + str(max_periods)
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    output = output.read()


    periods = []
    for line in output.splitlines():
        periods.append(line)

    return periods
