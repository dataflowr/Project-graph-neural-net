CC	= mpicc
CFLAGS	= -Ofast -g -Wall
LDFLAGS = -lm
TARGET	= multiprocess
EXECUTABLES_DIR=./

all: $(TARGET)

multiprocess: multiprocess.o
	$(CC) -o "$(EXECUTABLES_DIR)$@" $< $(LDFLAGS)
%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o

all:
	rm -f *.o
