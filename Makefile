CC=gcc
CXX=g++
RM=rm -f
# CPPFLAGS=-g $(shell root-config --cflags)
CPPFLAGS=-g -Wall -MMD -std=c++11
LDFLAGS=-g $(shell root-config --ldflags)
LDLIBS=$(shell root-config --libs)

SRCS=Viola_Jones.cpp features.cpp GetEyeList.cpp BMPstream.cpp imgProcessing.cpp Thresholds.cpp Weights.cpp Train.cpp
OBJS=$(subst .exe,.o,$(SRCS))

all: Viola_Jones

tester: $(OBJS)
	$(CXX) $(LDFLAGS) -o Viola_Jones.exe $(OBJS) $(LDLIBS) 

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CPPFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) *~ .depend

include .depend