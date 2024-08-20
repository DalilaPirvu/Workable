# Default to GNU compiler without OpenMP, no warnings and no profiling
FC = gfortran
OMP = T
PROF = T
WARN = T
BENCH = T
BENCH_FINE = T
BLAS = T

# Default FFTW Lib location
#FFTW_DIR = /usr/local/
FFTW_DIR = /home/dpirvu/fftw/
#FFTW_DIR = /opt/fftw/3.3.8-gcc-9.1.0-openmpi-4.0.1/
#FFTW_DIR = /usr/local/fftw/fftw-3.3.4-gcc-5.1.0/
#FFTW_DIR = $(HOME)/local/fftw-3.3.4/gcc-4.4.7/
#FFTW_DIR = /share/apps/fftw-3.3.5/
FFTW_INC = -I$(FFTW_DIR)include/
FFTW_LIB = -L$(FFTW_DIR)lib/

# Intel Compiler
ifeq ($(FC),ifort)
	FFLAGS= -r8 -fpp
ifeq ($(OMP),T)
	FFLAGS+= -openmp
endif	
	FOPT= -O3 -ipo -xHost #-fast -no-prec-div -parallel
ifeq ($(WARN),T)
	FFLAGS +=
endif
ifeq ($(PROF),T)
	FFLAGS +=
endif

# GNU Compiler
else ifeq ($(FC),gfortran)
	FFLAGS=-fdefault-real-8 -fdefault-double-8 -cpp -ffree-line-length-none 
ifeq ($(OMP),T)
	FFLAGS += -fopenmp
endif
	FOPT=-funroll-loops -O3 #-fblas  (check this blas one to replace matmul calls)
ifeq ($(WARN),T)
	FFLAGS+= -Wall -fbounds-check -pedantic
endif
ifeq ($(PROF),T)
	FFLAGS+= -g -pg
endif
endif

ifeq ($(OMP),T)
	FFTW = -lfftw3_omp -lfftw3
else
	FFTW = -lfftw3
endif

FLIBS = $(FFTW) 
ifeq ($(BLAS),T)
	FLIBS+=-lblas
endif
FLIBS+= -lm

#ifdef THREADS
#THREAD_LIB = -lpthread
#FFTW = -lfftw3_threads -lfftw3

# Define compiler macros
MACROS=
ifeq ($(OMP),T)
	MACROS+= -DUSEOMP=1
endif
ifeq ($(BLAS),T)
	MACROS+= -DUSEBLAS=1
endif
ifeq ($(BENCH),T)
	MACROS+= -DBENCHMARK=1
endif
ifeq ($(BENCH_FINE),T)
	MACROS+= -DBENCHMARK_FINE_GRAIN=1
endif

OBJS = constants.o fftw_mod_wtype.o random_field.o eom-scalar.o integrator.o

scalar: %: $(OBJS) evolve-scalar.o
	$(FC) $(FFLAGS) $(FOPT) $(FFTW_INC) $(MACROS) -o scalar-1d evolve-scalar.o $(OBJS) $(FFTW_LIB) $(FLIBS) $(THREAD_LIB)

%.o: %.f90
	$(FC) $(FFLAGS) $(FOPT) $(FFTW_INC) $(MACROS) -c $< -o $@ $(FFTW_LIB) $(FLIBS) $(THREAD_LIB)

.PHONY : clean

clean:
	rm -f *.o
	rm -f *.mod
	rm -f scalar-1d

cleadat:
	rm -f sims/fields*.dat sims/fft_fields*.dat
