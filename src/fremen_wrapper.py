import subprocess

def get_periodicities(path, number_of_periods=5, max_periods=60*60*24*7):
    """
    :param path: Path for the training data with 2 columns in the following order;   "timestamp occurrence(1/0)"
    :param number_of_periods:   number of periods you want in the return array
    :param max_period: maximum period
    :return: array of most fundamental periodicities
    """
    cmd = "./fremen " + str(path) + " " + str(number_of_periods) + " " + str(max_periods)
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    output = output.read()


    periods = []
    for line in output.splitlines():
        periods.append(float(line))
    return periods
