# NSE-MOBJ
Codebase for "A Multi-Objective Approach to Mitigate Negative Side Effects", published at IJCAI 2020.
Authors: Sandhya Saisubramanian, Ece Kamar, and Shlomo Zilberstein

Link to paper: https://www.ijcai.org/Proceedings/2020/50

-----------------------------------------------------------------------------------------------------------
Setup:

1. Clone the library
2. cd to the cloned folder in the local folder
3. cd NSE/include
4. Download boost: wget -O boost_1_67_0.tar.gz https://sourceforge.net/projects/boost/files/boost/1.67.0/boost_1_67_0.tar.gz/download
5. Extract boost: tar xzvf boost_1_67_0.tar.gz

Note: The code was tested on Ubuntu 16.04. Some of the packages are not supported by the latest Ubuntu, gcc versions and may result in compilation errors. 


Execution:
To compile: make testNSE

To run experiments: python runNSE.py [domain_name] [sensitivity]

To plot results: python plotLineNSE.py [logfile_name] [output_name].. The plots will be generated as output_name_HA.png and output_name_RL.png

Example command line: 

./testNSE --box=data/boxpushing/grid-3.bp --algorithm=LLAO --numObj=2 --v=100 --gamma=0.95 --slack=5

./testNSE --nav=data/navigation/grid-3.nav --algorithm=LLAO --numObj=2 --v=100 --gamma=0.95 --slack=5

Misc:
Instead of randomly generating problem instances for evaluation, this repository includes instances described by a map. Therefore the plots may be slightly different than those in the paper. 
1. The maps describing the problems are in the folder:data/[domain_name]/. Data files for Boxpushing domains end with .bp and navigation problems end with .nav
2. Test_boxpushing.py and test_navigation.py contain the domain-specific support functions for runNSE.py
