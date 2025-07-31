CXX = g++ -O3

HELPERS = helper.h
SRC_OUR = main.cu
KER_OUR = ker-helpers.cu.h ker-addition.cu.h ker-classic-mul.cu.h ker-fft-mul.cu.h
EXE_OUR = big-num-our

default: compile run

#.cu.o: $(SRC) $(HELPERS)
#	$(CXX) -c $@ $<


template: helper.h demo-template.cpp
	$(CXX) demo-template.cpp -o demo-template
	./demo-template 100000 

clean:
	rm -f demo-template
