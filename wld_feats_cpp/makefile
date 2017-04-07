OPENCV = /usr
CXX = g++
LINKFLAG = -pthread `pkg-config --cflags --libs opencv`
INCFLAGS = -I"$(OPENCV)/include/opencv"
CXXFLAGS = -O2 --fast-math -msse2

all: race_feat

clean:
	rm *.o ../lib/*.o

race_feat: race_feat.o cvwld.o
	$(CXX) -o $@ $^ $(LINKFLAG)

.c.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cpp.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cc.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)
