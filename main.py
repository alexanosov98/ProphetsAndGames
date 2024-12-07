import pickle
import numpy as np
import matplotlib.pyplot as plt

#Typedef
NumpyArray = np.ndarray

#Macros
TRAIN_SET_SIZE = 10000
TEST_SET_SIZE = 1000
NUM_OF_EXPERIMENTS = 100
ONE_PERCENT = 0.01


with open('data.pkl', 'rb') as file:
    data = pickle.load(file)
training_labels = np.array(data['train_set'])
testing_labels = np.array(data['test_set'])


def ERM(training_sets: NumpyArray, num_of_games: int) -> int:
    """
    Receives the predictions of each prophet, number of games the prophets are being evaluated on, number of times
    to repeat the experiment and returns the best prophet after evaluating via ERM for the given number of repetitions.
    """
    correct_predictions = np.zeros(len(training_sets)) #TODO changed from range(num_of_prophets)

    for j in range(num_of_games):
        random_game = np.random.randint(0, TRAIN_SET_SIZE)
        for prophet_idx in range(len(training_sets)): #TODO changed from range(num_of_prophets), depends what len(training_sets) is
            if training_sets[prophet_idx][random_game] == training_labels[random_game]:
                correct_predictions[prophet_idx] += 1


    best_prophets = np.where(correct_predictions == np.max(correct_predictions))[0]
    if len(best_prophets) > 1:
        random_idx = np.floor(np.random.uniform(0, len(best_prophets)))
        return best_prophets[int(random_idx)]
    return int(best_prophets[0])


def evaluate_and_calculate_average_error(testing_set: NumpyArray) -> float:
    """
    Returns the average error of the testing set provided when compared to the testing labels by summing all the
    corresponding samples in the arrays that are different from each other and dividing by the number of the samples.
    """

    return ((1 / TEST_SET_SIZE) * (np.sum(testing_set ^ testing_labels == True)))


#TODO correct the parameter types
def plot_table(data: NumpyArray, M, K) -> None:
    """
    Plots an m*k sized table presenting the data provided.
    """

    #Giving some context to the data to display it more informatively
    stringed_data = np.empty(data.shape, dtype=object)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            values = data[i, j]
            stringed_data[i, j] = f"Mean average error: {values[0]}% \n Mean approximation error: {values[1]}% \n Mean estimation error: {values[2]}%"

    #Start plotting
    fig, ax = plt.subplots(figsize = (12,8))
    row_labels = [f"m = {M[i]}" for i in range(len(M))]
    column_labels = [f"k = {K[i]}" for i in range(len(K))]
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText = stringed_data, rowLabels = row_labels, colLabels = column_labels, loc = 'center',
                         cellLoc = 'center')
    #Adjusting cell sizes
    cell_size = 0.2
    table.auto_set_column_width([i for i in range(len(K))])
    for (i,j), cell in table.get_celld().items():
        if i == 0:
            cell.set_width(cell_size/3)
            cell.set_height(cell_size/3)
        else:
            cell.set_width(cell_size)
            cell.set_height(cell_size)

    plt.savefig("table_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def run_experiments(training_sets: NumpyArray, num_of_games: int, test_sets: NumpyArray, true_risks: NumpyArray, scenario_6: bool) -> NumpyArray:
    """
    Runs the experiments NUM_OF_EXPERIMENTS times, every time choosing the best prophet via ERM and calculating the
    different errors.
    return: scenario 1-5, returns the mean errors computed, for scenario 6 returns just the estimated errors through out
    the different experiments.
    """

    chose_the_best_prophet = 0
    total_estimation_error = 0
    total_average_error = 0
    not_1_percent_worse = 0
    approximation_error = np.min(true_risks)
    if scenario_6:
        estimated_errors = np.ndarray(NUM_OF_EXPERIMENTS)

    for i in range(NUM_OF_EXPERIMENTS):
        best_prophet = ERM(training_sets, num_of_games)
        best_prophet_test_set = test_sets[best_prophet]
        average_error = evaluate_and_calculate_average_error(best_prophet_test_set)
        total_average_error += average_error
        estimation_error = true_risks[best_prophet] - approximation_error
        if scenario_6:
            estimated_errors[i] = estimation_error
        total_estimation_error += estimation_error
        if estimation_error < ONE_PERCENT :
            not_1_percent_worse += 1
        if not estimation_error:
            chose_the_best_prophet += 1
        #Uncomment to print the average error, approximation error and estimation error of the current experiment
        # print("Experiment number " + str(i+1) + ":")
        # print("The selected prophets error is: " + str(average_error * 100) + "%")
        # print("The approximation error is: " + str(approximation_error * 100) + "%")
        # print("The estimation error is: " + str(estimation_error * 100) + "%\n")
        if i == NUM_OF_EXPERIMENTS - 1:
            mean_average_error = total_average_error*100/NUM_OF_EXPERIMENTS
            mean_approximation_error = approximation_error * 100
            mean_estimation_error = total_estimation_error*100/NUM_OF_EXPERIMENTS
            #Uncomment to print the mean average error/ approximation error/ estimation error
            print("Experiments done, the best prophet was chosen " + str(chose_the_best_prophet) + " times out of " +
                  str(NUM_OF_EXPERIMENTS) + " experiments.")
            print("The mean average error is: " + str(mean_average_error) + "%")
            print("The mean approximation error is: " + str(mean_approximation_error) + "%")
            print("The mean estimation error is: " + str(mean_estimation_error) + "%")
            print("We chose a prophet that was not 1% worse than the best prophet " + str(not_1_percent_worse) +
                  " times out of " + str(NUM_OF_EXPERIMENTS) + " experiments.")
            if scenario_6:
                return estimated_errors
            return np.array([mean_average_error, mean_approximation_error, mean_estimation_error])


def plot_histogram(hypothesis1_estimated_errors, hypothesis2_estimated_errors) -> None:
    """
    Plots a histogram presenting the different estimated errors and their occurrences for every hypothesis class.
    """
    unique_values1, occurrences1 = np.unique(hypothesis1_estimated_errors, return_counts=True)
    unique_values2, occurrences2 = np.unique(hypothesis2_estimated_errors, return_counts=True)

    width = 0.4

    all_unique_values = np.union1d(unique_values1, unique_values2)
    x1_indices = np.arange(len(all_unique_values))
    x2_indices = x1_indices + width

    counts1 = [occurrences1[np.where(unique_values1 == val)[0][0]] if val in unique_values1 else 0 for val in
               all_unique_values]
    counts2 = [occurrences2[np.where(unique_values2 == val)[0][0]] if val in unique_values2 else 0 for val in
               all_unique_values]

    plt.bar(x1_indices, counts1, width=width, color='skyblue', label='Hypothesis 1', align='center')
    plt.bar(x2_indices, counts2, width=width, color='orange', label='Hypothesis 2', align='center')

    plt.xlabel("Estimation error of chosen prophet")
    plt.ylabel("Number of occurrences")
    plt.title("Comparison of Hypotheses Estimation Errors")
    plt.legend()

    plt.savefig("histogram_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def Scenario_1():
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    num_of_games = 1
    with open('scenario_one_and_two_prophets.pkl', 'rb') as file:
        scenario = pickle.load(file)
    true_risks = np.array(scenario['true_risk'])
    test_sets = np.array(scenario['test_set'])
    training_sets = np.array(scenario['train_set'])

    run_experiments(training_sets, num_of_games, test_sets, true_risks, False)


def Scenario_2():
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    num_of_games = 10
    with open('scenario_one_and_two_prophets.pkl', 'rb') as file:
        scenario = pickle.load(file)
    true_risks = np.array(scenario['true_risk'])
    test_sets = np.array(scenario['test_set'])
    training_sets = np.array(scenario['train_set'])

    run_experiments(training_sets, num_of_games, test_sets, true_risks, False)


def Scenario_3():
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    num_of_games = 10
    with open('scenario_three_and_four_prophets.pkl', 'rb') as file:
        scenario = pickle.load(file)
    true_risks = np.array(scenario['true_risk'])
    test_sets = np.array(scenario['test_set'])
    training_sets = np.array(scenario['train_set'])

    run_experiments(training_sets, num_of_games, test_sets, true_risks, False)

def Scenario_4():
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    num_of_games = 1000
    with open('scenario_three_and_four_prophets.pkl', 'rb') as file:
        scenario = pickle.load(file)
    true_risks = np.array(scenario['true_risk'])
    test_sets = np.array(scenario['test_set'])
    training_sets = np.array(scenario['train_set'])

    run_experiments(training_sets, num_of_games, test_sets, true_risks, False)


def Scenario_5():
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    M = [1, 10, 50, 1000]
    K = [2, 5, 10, 50]

    with open('scenario_five_prophets.pkl', 'rb') as file:
        scenario = pickle.load(file)
    true_risks = np.array(scenario['true_risk'])
    test_sets = np.array(scenario['test_set'])
    training_sets = np.array(scenario['train_set'])
    #Table initialization
    table = np.empty((len(K), len(M)), dtype = object)
    #Filling the table
    i = 0
    for m in M:
        j=0
        for k in K:
            #Randomly choose k prophets
            rand_prophets_idxs = np.random.randint(0, len(true_risks), size = k)

            table[i][j] = run_experiments(training_sets[rand_prophets_idxs], m, test_sets[rand_prophets_idxs],
                                          true_risks[rand_prophets_idxs], False)

            print("k = " + str(k) + " and m = " + str(m) + ":")
            print("The mean average error is: " + str(table[i][j][0]))
            print("The mean approximation error is: " + str(table[i][j][1]))
            print("The mean estimation error is: " + str(table[i][j][2]) + "\n")
            j += 1

        i += 1

    plot_table(table, M, K)


def Scenario_6():
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    num_of_games = 1000
    with open('scenario_six_prophets.pkl', 'rb') as file:
        scenario = pickle.load(file)
    hypothesis1 = scenario['hypothesis1']
    hypothesis2 = scenario['hypothesis2']
    hypothesis1_training_sets = hypothesis1['train_set']
    hypothesis1_testing_sets = hypothesis1['test_set']
    hypothesis2_training_sets = hypothesis2['train_set']
    hypothesis2_testing_sets = hypothesis2['test_set']
    hypothesis1_true_risks = hypothesis1['true_risk']
    hypothesis2_true_risks = hypothesis2['true_risk']

    print('Hypothesis 1:')
    hypothesis1_estimated_errors = run_experiments(hypothesis1_training_sets, num_of_games, hypothesis1_testing_sets,
                                          hypothesis1_true_risks, True)
    print('Hypothesis 2:')
    hypothesis2_estimated_errors = run_experiments(hypothesis2_training_sets, num_of_games, hypothesis2_testing_sets,
                                          hypothesis2_true_risks, True)


    plot_histogram(hypothesis1_estimated_errors, hypothesis2_estimated_errors)

if __name__ == '__main__':
    

    print(f'Scenario 1 Results:')
    Scenario_1()

    print(f'Scenario 2 Results:')
    Scenario_2()

    print(f'Scenario 3 Results:')
    Scenario_3()

    print(f'Scenario 4 Results:')
    Scenario_4()

    print(f'Scenario 5 Results:')
    Scenario_5()

    print(f'Scenario 6 Results:')
    Scenario_6()


