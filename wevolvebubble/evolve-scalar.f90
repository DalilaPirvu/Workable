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
  logical :: exists
  real(dl), pointer :: time
  real(dl), dimension(:,:), pointer :: fld
  type(transformPair1D) :: tPairgsq

  fld(1:nLat,1:2) => yvec(1:nVar-1) ! store the field in yvec
  time => yvec(nVar) ! last position stores the time?
  call setup(nVar)

  ! change this script to vary over specific parameters, as needed
  print*, "Temperature ", temp

  do sim = startSim, nSim-1

      exists = .False.
      call check_bubble(exists, sim)
      print*, exists

      if ( exists ) then
         call initialise_fields(fld)
         call time_evolve(fld, sim, alph)
         print*, "Simulation ", sim, " out of ", nSim-1, " done!"
      else
         print*, "Skipping simulation ", sim, " out of ", nSim-1, " done!"
      endif

  end do
  print*, "All Done!"

contains

  subroutine initialise_fields(fld)
    real(dl), dimension(:,:), intent(inout) :: fld

    call upload_bubble_seeds(fld)

    yvec(nVar) = 0 ! Add a tcur pointer here

  end subroutine initialise_fields

  subroutine time_evolve(fld, sim, alp) 
    real(dl), dimension(1:nLat, 1:2) :: fld
    real(dl), dimension(1:size(fld(:,1)), 2) :: df

    real(dl) :: dt
    real(dl) :: dtout
    real(dl) :: alp
    integer :: j, k
    integer :: sim

    dt = dx / alp
    if (dt > dx) print*, "Warning, violating Courant condition"
    dtout = dt * alp
    print*, dtout, dt, alp, dx

    ! initial conditions
    df = fld(:,:)
    call output_fields(df, sim)

    k = 0
    do while ( k < nTimeMax )
       do j = 1, int(alp)
          call gl10(yvec, dt)
       end do

       k = k + 1
       call output_fields(fld, sim)
    end do
  end subroutine time_evolve

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


  subroutine check_bubble(on, sim)
    integer, intent(in) :: sim
    logical, intent(inout) :: on

    print*, '/gpfs/dpirvu/prefactor_len80/x'//trim(str(nLat))//'_m2eff'//trim(real_str(m2eff))//'_T'//trim(real_str(temp))//'_sim'//trim(str(sim))//'_bubble_seeds.txt'

    inquire(file='/gpfs/dpirvu/prefactor_len80/x'//trim(str(nLat))//'_m2eff'//trim(real_str(m2eff))//'_T'//trim(real_str(temp))//'_sim'//trim(str(sim))//'_bubble_seeds.txt', exist=on)
  end subroutine check_bubble


  subroutine upload_bubble_seeds(fld)
    real(dl), dimension(1:nLat, 1:2) :: fld
    logical :: op
    integer, parameter :: seedsFile = 98
    integer :: m

    ! Inquire if the file is already open
    inquire(file='/gpfs/dpirvu/prefactor_len80/x'//trim(str(nLat))//'_m2eff'//trim(real_str(m2eff))//'_T'//trim(real_str(temp))//'_sim'//trim(str(sim))//'_bubble_seeds.txt', opened=op)

    if (.not. op) then
        ! If the file is not open, open it
        open(unit=seedsFile, file='/gpfs/dpirvu/prefactor_len80/x'//trim(str(nLat))//'_m2eff'//trim(real_str(m2eff))//'_T'//trim(real_str(temp))//'_sim'//trim(str(sim))//'_bubble_seeds.txt')
    endif

    do m = 1, nLat
        ! Read two real numbers from each line of the file
        read(seedsFile, *) fld(m,:)
    end do

    ! Close the file
    close(seedsFile)
  end subroutine upload_bubble_seeds


  subroutine output_fields(fld, sim)
    real(dl), dimension(1:nLat, 1:2) :: fld
    integer  :: m, sim
    logical  :: o
    integer, parameter :: oFile = 98

    inquire(file='/gpfs/dpirvu/prefactor_len80/bubble_formation_x'//trim(str(nLat))//'_m2eff'//trim(real_str(m2eff))//'_T'//trim(real_str(temp))//'_sim'//trim(str(sim))//'_fields.dat', opened=o)

    if (.not.o) then
       open(unit=oFile,file='/gpfs/dpirvu/prefactor_len80/bubble_formation_x'//trim(str(nLat))//'_m2eff'//trim(real_str(m2eff))//'_T'//trim(real_str(temp))//'_sim'//trim(str(sim))//'_fields.dat')
    endif

    do m = 1, nLat
       write(oFile,*) fld(m,:)
    end do

  end subroutine output_fields

end program Gross_Pitaevskii_1d
