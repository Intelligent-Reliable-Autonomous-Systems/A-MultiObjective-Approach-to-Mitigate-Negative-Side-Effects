# NSE-MOBJ

1. Clone the library
2. cd to the cloned folder in the local folder
3. cd NSE/include

Download boost
4. wget -O boost_1_67_0.tar.gz https://sourceforge.net/projects/boost/files/boost/1.67.0/boost_1_67_0.tar.gz/download
5. tar xzvf boost_1_67_0.tar.gz

6. To compile: make testNSE
7. To run experiments: python runNSE.py [domain_name] [sensitivity]
8. To plot results: python plotLineNSE.py [logfile_name] [output_name].. The plots will be generated as output_name_HA.png and output_name_RL.png

Example command line: ./testNSE --box=data/boxpushing/grid-3.bp --algorithm=LLAO --numObj=2 --v=100 --gamma=0.95 --slack=5
		      ./testNSE --nav=data/navigation/grid-3.nav --algorithm=LLAO --numObj=2 --v=100 --gamma=0.95 --slack=5

The maps describing the problems are in the folder:data/[domain_name]/. Data files for Boxpushing domains end with .bp and navigation problems end with .nav

9. Test_boxpushing.py and test_navigation.py contain the domain-specific support functions for runNSE.py
