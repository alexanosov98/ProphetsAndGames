import pickle
import numpy as np
import matplotlib.pyplot as plt

####TODO NOTES TO TAKE CARE OF LATER
# See where you can optimize by using numpy functions
# define parameter types
# documentation



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


#todo remove redundant paramter later
def ERM(training_sets: NumpyArray, num_of_games: int, num_of_prophets: int) -> int:
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
    return int(best_prophets[0])  # todo why the cast?



def evaluate_and_calculate_average_error(testing_set: NumpyArray) -> float:
    """
        Returns the average error of the testing set provided when compared to the testing labels by summing all the
        corresponding samples in the arrays that are different from each other and dividing by the number of the samples.
        """

    return ((1 / TEST_SET_SIZE) * (np.sum(testing_set ^ testing_labels == True)))  ##TODO CHANGED!!!!!!


#TODO correct the parameter types
def create_table(data: NumpyArray, M, K) -> None:
    """

    :return:
    """

    #Giving some context to the data to display it more informatively
    stringed_data = np.empty(data.shape, dtype=object)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            values = data[i, j]
            stringed_data[i, j] = f"Mean average error: {values[0]}% \n Mean approximation error: {values[1]}% \n Mean estimated error: {values[2]}%"

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


#Todo remove redundant paramter later
def run_experiments(training_sets: NumpyArray, num_of_games: int, num_of_prophets: int, num_of_repetitions: int, test_sets : NumpyArray, true_risks: NumpyArray) -> NumpyArray:
    """"""
    chose_the_best_prophet = 0
    total_estimation_error = 0
    total_average_error = 0
    not_1_percent_worse = 0
    approximation_error = np.min(true_risks)

    for i in range(NUM_OF_EXPERIMENTS):
        best_prophet = ERM(training_sets, num_of_games, num_of_prophets)
        best_prophet_test_set = test_sets[best_prophet]
        average_error = evaluate_and_calculate_average_error(best_prophet_test_set)
        total_average_error += average_error
        estimation_error = true_risks[best_prophet] - approximation_error
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
            # print("Experiments done, the best prophet was chosen " + str(chose_the_best_prophet) + " times out of " +
            #       str(num_of_repetitions) + " experiments.")
            # print("The mean average error is: " + str(mean_average_error) + "%")
            # print("The mean approximation error is: " + str(mean_approximation_error) + "%")
            # print("The mean estimation error is: " + str(mean_estimation_error) + "%")
            # print("We chose a prophet that was not 1% worse than the best prophet " + str(not_1_percent_worse) +
            #       " times out of " + str(num_of_repetitions) + " experiments.")

            return np.array([mean_average_error, mean_approximation_error, mean_estimation_error])




def Scenario_1():
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    num_of_prophets = 2
    num_of_games = 1
    with open('scenario_one_and_two_prophets.pkl', 'rb') as file:
        scenario = pickle.load(file)
    true_risks = np.array(scenario['true_risk'])
    test_sets = np.array(scenario['test_set'])
    training_sets = np.array(scenario['train_set'])

    run_experiments(training_sets, num_of_games, num_of_prophets, NUM_OF_EXPERIMENTS, test_sets, true_risks)


def Scenario_2():
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    num_of_prophets = 2
    num_of_games = 10
    with open('scenario_one_and_two_prophets.pkl', 'rb') as file:
        scenario = pickle.load(file)
    true_risks = np.array(scenario['true_risk'])
    test_sets = np.array(scenario['test_set'])
    training_sets = np.array(scenario['train_set'])

    run_experiments(training_sets, num_of_games, num_of_prophets, NUM_OF_EXPERIMENTS, test_sets, true_risks)


def Scenario_3():
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    num_of_prophets = 500
    num_of_games = 10
    with open('scenario_three_and_four_prophets.pkl', 'rb') as file:
        scenario = pickle.load(file)
    true_risks = np.array(scenario['true_risk'])
    test_sets = np.array(scenario['test_set'])
    training_sets = np.array(scenario['train_set'])

    run_experiments(training_sets, num_of_games, num_of_prophets, NUM_OF_EXPERIMENTS, test_sets, true_risks)

def Scenario_4():
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    num_of_prophets = 500
    num_of_games = 1000
    with open('scenario_three_and_four_prophets.pkl', 'rb') as file:
        scenario = pickle.load(file)
    true_risks = np.array(scenario['true_risk'])
    test_sets = np.array(scenario['test_set'])
    training_sets = np.array(scenario['train_set'])

    run_experiments(training_sets, num_of_games, num_of_prophets, NUM_OF_EXPERIMENTS, test_sets, true_risks)


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

            table[i][j] = run_experiments(training_sets[rand_prophets_idxs], m, k, NUM_OF_EXPERIMENTS,
                                          test_sets[rand_prophets_idxs], true_risks[rand_prophets_idxs])

            print("k = " + str(k) + " and m = " + str(m) + ":")
            print("The mean average error is: " + str(table[i][j][0]))
            print("The mean approximation error is: " + str(table[i][j][1]))
            print("The mean estimation error is: " + str(table[i][j][2]) + "\n")
            j += 1

        i += 1

    create_table(table, M, K)


def Scenario_6():
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    pass


if __name__ == '__main__':
    

    # print(f'Scenario 1 Results:')
    # Scenario_1()

    # print(f'Scenario 2 Results:')
    # Scenario_2()
    #
    # print(f'Scenario 3 Results:')
    # Scenario_3()
    #
    # print(f'Scenario 4 Results:')
    # Scenario_4()
    #
    print(f'Scenario 5 Results:')
    Scenario_5()
    #
    # print(f'Scenario 6 Results:')
    # Scenario_6()

    # M = [1, 10, 50, 1000]
    # K = [2, 5, 10, 50]
    # create_histogram(np.random.rand(4, 4), M, K)

