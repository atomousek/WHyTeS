CXX=g++
CPPFLAGS+=-Wall -Werror -g
RM=rm -f

main: main.o load.o grid.o
	$(CXX) $(CPPFLAGS) main.o load.o grid.o -o main

main.o: main.cpp
	$(CXX) $(CPPFLAGS) -c main.cpp

load.o: load.cpp load.h
	$(CXX) $(CPPFLAGS) -c load.cpp

grid.o:	grid.cpp grid.h
	$(CXX) $(CPPFLAGS) -c grid.cpp

clean:
	$(RM) *.o main
