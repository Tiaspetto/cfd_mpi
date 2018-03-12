CC=gcc
CFLAGS=-O2 -Wall -g -pg 

.c.o:
	$(CC) -c $(CFLAGS) $<

all: bin2ppm diffbin pingpong colcopy karman # karman-par

clean:
	-del bin2ppm diffbin pingpong colcopy karman *.o *.exe

karman: alloc.o boundary.o init.o karman.o simulation.o
	$(CC) $(CFLAGS) -o $@ $^ -lm -lmsmpi

karman-par: alloc.o boundary.o init.o karman-par.o simulation-par.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

bin2ppm: bin2ppm.o alloc.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

diffbin: diffbin.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

pingpong: pingpong.o
	$(CC) $(CFLAGS) -o $@ $^ -lmsmpi

colcopy: colcopy.o alloc.o
	$(CC) $(CFLAGS) -o $@ $^ -lmsmpi

bin2ppm.o        : alloc.h datadef.h
boundary.o       : datadef.h
colcopy.o        : alloc.h
init.o           : datadef.h
karman.o         : alloc.h boundary.h datadef.h init.h simulation.h
karman-par.o     : alloc.h boundary.h datadef.h init.h simulation.h
simulation.o     : datadef.h init.h
simulation-par.o : datadef.h init.h
