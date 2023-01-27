import matplotlib.pyplot as plt

def plotGraphAcc(m, accuracy_History, datasetName):
    ax = plt.axes()
    ax.plot(m, accuracy_History, c = "grey")


    for i in range(40):
        if accuracy_History[i] == 100.0:
            ax.scatter(i, accuracy_History[i], c='purple', marker='o', s = 50, alpha=1)
        else:
            ax.scatter(i, accuracy_History[i], c='orange', marker='o', s = 50, alpha=1)

    ax.set_title("Accuracy Vs. # of Vectors " + "(" + datasetName + ")")
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('m')


plt.show()

def plotGraphCER(m, CER_History, datasetName):
    ax = plt.axes()
    ax.plot(m, CER_History, c = "grey")


    for i in range(40):
        if CER_History[i] == 0.0:
            ax.scatter(i, CER_History[i], c='green', marker='o', s = 50, alpha=1)
        else:
            ax.scatter(i, CER_History[i], c='red', marker='o', s = 50, alpha=1)

    ax.set_title("Classification Error Rate Vs. # of Vectors " + "(" + datasetName + ")")
    ax.set_ylabel('Classification Error Rate (CER)')
    ax.set_xlabel('m')


    plt.show()