import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores_df, figure_file, algorithm, labels):
    plt.figure(figsize=(10, 6))
    
    # Iterate over each row in the DataFrame
    for index, scores in scores_df.iterrows():
        scores = scores.dropna()
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])

        x_adjusted = x[:len(running_avg)]
        plt.plot(x_adjusted, running_avg, label=labels[index])
    
    plt.title(f'Running average of previous 100 scores - {algorithm}')
    plt.xlabel("Episode")
    plt.ylabel("Avg Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_file)
    #plt.show()
