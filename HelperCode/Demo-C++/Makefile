CXX = g++ -O3

HELPERS = helper.h
SRC_OUR = main.cu
KER_OUR = ker-helpers.cu.h ker-addition.cu.h ker-classic-mul.cu.h ker-fft-mul.cu.h
EXE_OUR = big-num-our

default: compile run

#.cu.o: $(SRC) $(HELPERS)
#	$(CXX) -c $@ $<


templates: helper.h templates.cpp
	$(CXX) templates.cpp -o templates
	./templates 100000 

clean:
	rm -f templates
