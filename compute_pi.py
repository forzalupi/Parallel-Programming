import matplotlib.pyplot as plt
import time
def amdahls_plot(measured, theoretical , cores, save=False):
    """
    """
    fig, ax = plt.subplots()
    ax.plot(cores, measured, color="r", marker="o", label = "measured")
    ax.plot(cores, theoretical, color="b", marker="o", label = "theoretical")
    ax.set(title="Measured and Theoretical speedup as a function of used Cores", xlabel="Cores", ylabel="Speedup")
    ax.legend()
    plt.show()

    if save:
        plt.savefig(f"amdahls_{time.time()}.png")
