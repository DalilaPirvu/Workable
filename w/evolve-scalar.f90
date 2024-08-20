#include "macros.h"
#define SMOOTH 1

program Gross_Pitaevskii_1d
  use, intrinsic :: iso_c_binding
  use gaussianRandomField
  use integrator
  use constants
  use eom
  implicit none

  integer :: sim
  real(dl), pointer :: time
  real(dl), dimension(:,:), pointer :: fld
  type(transformPair1D) :: tPairgsq

  fld(1:nLat,1:2) => yvec(1:nVar-1) ! store the field in yvec
  time => yvec(nVar) ! last position stores the time?
  call initialize_rand(93286123, seedfac)
  call setup(nVar)

  ! change this script to vary over specific parameters, as needed
  print*, "Temperature ", temp

  do sim = 0, nSim-1
      call initialise_fields(fld)
      if (sim >= lSim) then
!          if (ANY(simList==sim)) then
         call time_evolve(sim, alph)
         print*, "Simulation ", sim+1, " out of ", nSim , " done!"
!          endif
      endif
  end do
  print*, "All Done!"

contains

  subroutine initialise_fields(fld)
    real(dl), dimension(:,:), intent(inout) :: fld

    fld(:,1)   = fldinit
    fld(:,2)   = mominit
    yvec(nVar) = 0 ! Add a tcur pointer here

    call initialize_linear_fluctuations(fld)
  end subroutine initialise_fields

  subroutine time_evolve(sim, alp) 
    real(dl) :: dt
    real(dl) :: dtout
    real(dl) :: alp
    integer :: j, k
    integer :: sim
    integer :: m
    integer :: tclock
    logical :: bool1
    real(dl), dimension(1:size(fld(:,1)), 2) :: df

    df = fld(:,:)

    tclock = 0

    dt = dx / alp
    if (dt > dx) print*, "Warning, violating Courant condition"
    dtout = dt * alp

    k = 0
    bool1 = .True.

    call output_fields(df, dtout, sim, tclock)

    do while ( bool1 )
       do j = 1, int(alp)
          call gl10(yvec, dt)
       end do

       k = k + 1
       if (MOD(k, 256) == 0) then
           do m = 1, nLat
               if ( abs(fld(m,1)) > 10. ) then
                  bool1 = .False.
                  exit
               endif
           enddo

           if ( MOD(k, 1024) == 0 .AND. bool1 ) then
               call output_fields(fld, dtout, sim, tclock)
           endif
       endif

       if ( k == nTimeMax .OR. .NOT. bool1 ) then
          tclock = k
          bool1 = .False.
          print*, "tclock ", tclock, tclock*dtout, " out of ", nTimeMax, nTimeMax*dtout
          call output_fields(fld, dtout, sim, tclock)
       end if
    end do
  end subroutine time_evolve

  subroutine initialize_linear_fluctuations(fld)
    real(dl), dimension(:,:), intent(inout) :: fld
    real(dl), dimension(1:nLat) :: df
    real(dl), dimension(1:nLat/2+1) :: specfld, w2eff
    integer :: i, nn, j, k

    nn = size(specfld)
    do i = 1, nn
       w2eff(i) = m2eff + (2. / (dx**2.))*(1. - cos(dx * dk * (i-1)))
    enddo

    specfld(:) = norm * sqrt(temp / w2eff(:))

    df(:) = 0.
    call generate_1dGRF(df, specfld(1:kspec))
    fld(:,1) = fld(:,1) + df(:)

    specfld(:) = 0.
    specfld(:) = norm * sqrt(temp)

    df(:) = 0.
    call generate_1dGRF(df, specfld(1:kspec))
    fld(:,2) = fld(:,2) + df(:)

  end subroutine initialize_linear_fluctuations

  subroutine setup(nVar)
    integer, intent(in) :: nVar
    call init_integrator(nVar)
    call initialize_transform_1d(tPair,nLat)
    call initialize_transform_1d(tPairgsq,nLat)
  end subroutine setup

  character(len=20) function str(k)
    integer, intent(in) :: k
    write (str, *) k
    str = adjustl(str)
  end function str

  character(len=20) function real_str(k)
    real(dl), intent(in) :: k
    write (real_str, '(f12.4)') k
    real_str = adjustl(real_str)
  end function real_str

  subroutine output_fields(fld, dtout, sim, tclock)
    real(dl), dimension(1:nLat, 1:2) :: fld
    real(dl) :: dtout
    integer  :: m, sim, tclock
    logical  :: o
    integer, parameter :: oFile = 98

    inquire(file='/gpfs/dpirvu/prefactor_len80/x'//trim(str(nLat))//'_m2eff'//trim(real_str(m2eff))//'_T'//trim(real_str(temp))//'_sim'//trim(str(sim))//'_fields.dat', opened=o)

    if (.not.o) then
       open(unit=oFile,file='/gpfs/dpirvu/prefactor_len80/x'//trim(str(nLat))//'_m2eff'//trim(real_str(m2eff))//'_T'//trim(real_str(temp))//'_sim'//trim(str(sim))//'_fields.dat')
       write(oFile,*) "dx", dx
       write(oFile,*) "nLat", nLat
       write(oFile,*) "lenLat", lenLat
       write(oFile,*) "dk", dk
       write(oFile,*) "dtout", dtout
       write(oFile,*) "m2eff", m2eff
       write(oFile,*) "temperature", temp
       write(oFile,*) "tstart", 0
    endif

    do m = 1, nLat
       write(oFile,*) fld(m,:)
    end do

    if ( tclock > 0 ) then
       write(oFile,*) "tfinal", tclock
    endif

  end subroutine output_fields

end program Gross_Pitaevskii_1d
