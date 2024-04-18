from protein_folding import ProteinFolding
from logger import Logger
from plotter import ProteinPlotting

def run_simulation(pf, log, sweep):
    """ Runs the MC simulation """
    for _ in range(sweep):
        pf.perform_mc_step(log)

def task1_5():
    """ Performing X = 1,10,100 sweeps on a N=15 unfolded protein for T = 10 """
    # Initialize proteinFolding, logger and plotting instance
    pf = ProteinFolding(15,10)
    log = Logger()
    pplot = ProteinPlotting(pf,log)
    
    # Generate the unfolded protein
    pf.gen_unfolded_protein()
    # Visualize it, making sure its unfolded
    pplot.plot_monomer(0)
    # Plot the interaction matrix
    pplot.plot_im()
    run_simulation(pf,log,1)
    pplot.plot_data(1)
    pplot.plot_monomer(1)

    run_simulation(pf,log,10)
    pplot.plot_data(10)
    # Check if its updated
    pplot.plot_monomer(10)

    run_simulation(pf,log,100)
    pplot.plot_data(100)
    # Plot the final shape after 100 sweeps
    pplot.plot_monomer(100)

def task1_6():
    """ Weird task, not a lot happens when we have N = 100 because there is such a low chance for a move to be valid """
    pf = ProteinFolding(15,1)
    log = Logger()
    pplot = ProteinPlotting(pf,log)
    pf.gen_unfolded_protein()
    run_simulation(pf,log,1)
    pplot.plot_monomer(1)

    run_simulation(pf,log,10)
    pplot.plot_monomer(10)

    run_simulation(pf,log,100)
    pplot.plot_monomer(100)
    
    run_simulation(pf,log,150)
    pplot.plot_monomer(150)
if __name__ == "__main__":
    # task1_5()
    task1_6()




