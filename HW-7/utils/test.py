import matplotlib.pyplot as plt


def plotGraphs(hidden_units, learning_rate, reg_param, val):
    # Create the subplots

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    # Iterate over the subplots and plot the data
    for i, ax in enumerate(axs.flat):
        if i == 0:
            print(i)
            ax.plot([ep for ep in range(1, 31)], val[0], label="Train Accuracy", color = "#804674")
            ax.scatter([p for p in range(1, 31)], val[0], color = "#804674", marker='o', s = 30, alpha=1)
            ax.set_ylabel('Train Accuracy(%)')
            ax.set_xlabel('Epochs')
            ax.set_title(f"Hidden Units: {hidden_units}, Learning rate: {learning_rate}, lambda: {reg_param}")
            ax.legend(["Accuracy"], loc = 0, frameon = True)
            
        elif i == 1:
            print(i)
            ax.plot([ep for ep in range(1, 31)], val[1], label="Train Loss", color = "#804674")
            ax.scatter([p for p in range(1, 31)], val[1], color = "#804674", marker='o', s = 30, alpha=1)
            ax.set_ylabel('Training Loss')
            ax.set_xlabel('Epochs')
            ax.set_title(f"Hidden Units: {hidden_units}, Learning rate: {learning_rate}, lambda: {reg_param}")
            ax.legend(["Loss"], loc = 0, frameon = True)

        elif i == 2:
            print(i)
            ax.plot([ep for ep in range(1, 31)], val[2], label="Validation Accuracy", color = "#804674")
            ax.scatter([p for p in range(1, 31)], val[2], color = "#804674", marker='o', s = 30, alpha=1)
            ax.set_ylabel('Validation Accuracy(%)')
            ax.set_xlabel('Epochs')
            ax.set_title(f"Hidden Units: {hidden_units}, Learning rate: {learning_rate}, lambda: {reg_param}")
            ax.legend(["Accuracy"], loc = 0, frameon = True)

        else:
            print(i)
            ax.plot([ep for ep in range(1, 31)], val[3], label="Validation Loss", color = "#804674")
            ax.scatter([p for p in range(1, 31)], val[3], color = "#804674", marker='o', s = 30, alpha=1)
            ax.set_ylabel('Validation Loss')
            ax.set_xlabel('Epochs')
            ax.set_title(f"Hidden Units: {hidden_units}, Learning rate: {learning_rate}, lambda: {reg_param}")
            ax.legend(["Loss"], loc = 0, frameon = True)


    # Show the plot
    plt.show()



result = {
    40: {
        0.001: {
            0.0001: [],
            0.001: [],
            0.01 : []
        },

        0.01: {
            0.0001: [],
            0.001: [],
            0.01 : []
        },

        0.1: {
            0.0001: [],
            0.001: [],
            0.01 : []
        }
    },

    80:{
      0.001: {
            0.0001: [],
            0.001: [],
            0.01 : []
        },

        0.01: {
            0.0001: [],
            0.001: [],
            0.01 : []
        },

        0.1: {
            0.0001: [],
            0.001: [],
            0.01 : []
        }

    },

    160: {
        0.001: {
            0.0001: [],
            0.001: [],
            0.01 : []
        },

        0.01: {
            0.0001: [],
            0.001: [],
            0.01 : []
        },

        0.1: {
            0.0001: [],
            0.001: [],
            0.01 : []
        }
    }
}

for hidden_units, vals in result.items():
    for learning_rate, values in vals.items():
        for reg_param, val in values.items():
            print(hidden_units, learning_rate, reg_param, val)
            plotGraphs(hidden_units, learning_rate, reg_param, val)

# plotGraphs(2)
