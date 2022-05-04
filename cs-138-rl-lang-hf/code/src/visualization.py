import matplotlib.pyplot as plt

def plot_results(rewards, names, colors=['r', 'g', 'b']):
    for i, d in enumerate(rewards):
        xs = d.keys()
        ys = d.values()
        plt.plot(xs, ys, linewidth=3,
                 label=names[i], alpha=0.5, color=colors[i])
    plt.legend()
    plt.xlabel('Training Steps')
    plt.ylabel('Rewards')
    plt.title('Training Steps vs. Rewards on GridWorld Change-Up')
    plt.show()

