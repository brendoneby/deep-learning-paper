import pandas
import matplotlib.pyplot as plt
from sessions import run_sessions

def get_test_file(type, traderOne, traderMany):
    return 'total_balance_{0}_{1}_{2}.csv'.format(type, traderOne, traderMany)

def parse_test_file(file):
    data = pandas.read_csv(file).to_numpy()
    perf = {}
    for row in data:
        for col in range(4,len(row),4):
            item = row[col:col+4]
            if isinstance(item[0],str):
                trader = item[0].strip()
                if trader not in perf: perf[trader] = []
                perf[trader].append(item[3])
    return perf

def generate_box_plots(file, title, filename):
    data = parse_test_file(file)
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title(title)
    ax.boxplot(data.values(), whis=1.5)
    ax.set_xticklabels(data.keys())
    # plt.show()
    plt.savefig(filename)

def one_in_many_test(traderOne, traderMany):
    test_type = "OMT"
    traders_spec = [(traderOne,1),(traderMany,39)]
    total_file = get_test_file(test_type, traderOne, traderMany)

    n_trials = 100
    print("Starting One-vs-many test for "+traderOne+" vs "+traderMany)
    run_sessions(traders_spec, traders_spec, n_trials, total_file=total_file)
    generate_omt_plot(traderOne, traderMany)

def generate_omt_plot(traderOne, traderMany):
    file = get_test_file("OMT", traderOne, traderMany)
    title = "One-in-many: "+traderOne + " vs " + traderMany
    generate_box_plots(file, title, f'Test Results/{traderOne}/{traderOne} vs {traderMany} OMT.png')

def balanced_group_test(traderOne, traderTwo):
    test_type = "BGT"
    traders_spec = [(traderOne,20),(traderTwo,20)]
    total_file = get_test_file(test_type, traderOne, traderTwo)
    n_trials = 100
    print("Starting Balanced Group test for "+traderOne+" vs "+traderTwo)
    run_sessions(traders_spec, traders_spec, n_trials, total_file=total_file)
    generate_bgt_plot(traderOne, traderTwo)

def generate_bgt_plot(traderOne, traderTwo):
    file = get_test_file("BGT", traderOne, traderTwo)
    title = "Balanced-group-test: "+traderOne + " vs " + traderTwo
    generate_box_plots(file, title, f'Test Results/{traderOne}/{traderOne} vs {traderTwo} BGT.png')
