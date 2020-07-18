#########################################################################
#                                VARIABLES                              #
#########################################################################


# Compilation flags and variables
CC = g++
CFLAGS = -std=c++11 -g -DATOM_STATES -pthread

# Variables for directories
ID = include
SD = src
TD = test
OD = obj
ID_UTIL = $(ID)/util
SD_UTIL = $(SD)/util
ID_SOLV = $(ID)/solvers
SD_SOLV = $(SD)/solvers
OD_SOLV = $(OD)/solvers

ID_DOMAINS = $(ID)/domains
SD_DOMAINS = $(SD)/domains
OD_DOMAINS = $(OD)/domains
SD_BP = $(SD_DOMAINS)/boxpushing
ID_BP = $(ID_DOMAINS)/boxpushing
SD_NV = $(SD_DOMAINS)/navigation
ID_NV = $(ID_DOMAINS)/navigation

#Variables for include directives
INCLUDE_DOM = -I$(ID_BP) -I$(ID_DOMAINS)
INCLUDE_CORE = -I$(ID_UTIL) -I$(ID)
INCLUDE_SOLVERS = -I$(ID_SOLV) 
INCLUDE = $(INCLUDE_DOM) $(INCLUDE_CORE) $(INCLUDE_SOLVERS)
I_BOOST = -I$(ID)/boost_1_67_0

# Variables for source/header files
I_H = $(ID)/*.h
S_CPP = $(SD)/*.cpp
SOLV_CPP = $(SD_SOLV)/*.cpp
SOLV_H = $(ID_SOLV)/*.h
UTIL_CPP = $(SD_UTIL)/*.cpp
UTIL_H = $(ID_UTIL)/*.h

BP_CPP = $(SD_BP)/*.cpp
BP_H = $(ID_BP)/*.h
NV_CPP = $(SD_NV)/*.cpp
NV_H = $(ID_NV)/*.h

DOM_CPP = $(BP_CPP) $(NV_CPP)  $(SD_DOMAINS)
DOM_H = $(BP_H) $(NV_H)
ALL_H = $(I_H) $(SOLV_H) $(DOM_H) $(UTIL_H)
ALL_CPP = $(DOM_CPP) $(SOLV_CPP) $(UTIL_CPP)

# Libraries
LIBS = lib/libmdp.a lib/libmdp_domains.a -Llib

#########################################################################
#                                 TARGETS                               #
#########################################################################

# Compiles the core MDP-LIB library #
libmdp: lib/libmdp.a
lib/libmdp.a: $(OD)/core.a $(OD)/solvers.a
	make $(OD)/core.a
	make $(OD)/solvers.a
	ar rvs libmdp.a $(OD)/core/*.o $(OD)/solvers/*.o
	mkdir -p lib
	mv libmdp.a lib
	
$(OD)/solvers.a: $(S_CPP) $(UTIL_CPP) $(I_H) $(UTIL_H) $(SOLV_CPP) $(SOLV_H) $(ID_DOMAINS) $(SD_DOMAINS)
	make $(OD)/core.a
	$(CC) $(CFLAGS) $(INCLUDE_CORE) $(ID_DOMAINS) -c $(SOLV_CPP) 
	mkdir -p $(OD_SOLV)
	mv *.o $(OD_SOLV)
	ar rvs $(OD)/solvers.a $(OD_SOLV)/*.o		
	
# Compiles the core classes
$(OD)/core.a: $(S_CPP) $(UTIL_CPP) $(I_H) $(UTIL_H)
	$(CC) $(CFLAGS) $(INCLUDE_CORE) -c $(UTIL_CPP) $(S_CPP) $(UTIL_CPP)
	mkdir -p obj/core
	mv *.o obj/core
	ar rvs $(OD)/core.a $(OD)/core/*.o
	

testNSE: lib/libmdp.a domains
	$(CC) $(CFLAGS) $(I_BOOST) $(INCLUDE) -o testNSE.out $(TD)/testNSE.cpp $(LIBS)


domains: lib/libmdp_domains.a
lib/libmdp_domains.a: lib/libmdp.a $(DOM_H) $(DOM_CPP)
	$(CC) $(CFLAGS) $(I_BOOST) $(INCLUDE) -c $(DOM_CPP)
	mkdir -p $(OD_DOMAINS)
	mv *.o $(OD_DOMAINS)
	ar rvs lib/libmdp_domains.a $(OD_DOMAINS)/*.o	
	
.PHONY: clean
clean:
	rm -f $(TD)/*.o
	rm -f *.o
	rm -f $(OD)/*.a
	rm -f $(OD)/*.o
	rm -f $(OD)/core/*.o
	rm -f $(OD)/domains/*.o
	rm -f $(OD)/domains/*.a
	rm -f $(OD)/solvers/*.o
	rm -f $(OD)/solvers/*.a
	rm -f lib/libmdp*.a