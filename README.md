# RPU_Rate_Model
A biologically inspired and fully local and recurrent rectified polynomial unit (RePU) rate model network.

The purpose of this repository is proof of existence of a RePU rate model that uses only local rules (neurons only have access to locally stored variables (trackers) and their synapses to obtain rate information or similar from connected neurons. The network can start in a gigantic basin in parameter space and gradually moves towards a regime where is can stablely learn competitive, drifiting (semi stable) recepitve fields between neurons with local connectivity. Large work in progress and documentation in progress. (Paper draft in progress, as well as more formal presentation of various emergence results are also wip)
The network contains a population of fully recurrent excitatory and inhibitory neuron pools with cross connectivity and high recurrent amplification. The system finds a semi stable equilibrium point.
