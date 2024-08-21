!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Integrate an externally given set of equations of motion using a 10th order Gauss-Legendre integrator
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!>@author
!> Jonathan Braden
!>University College London
!>
!> @brief
!> Integrate an externally given set of equations of motion using a 10th order Gauss-Legendre integrator.
!> This implementation follows the "compile" method, where the external equations of motion must be provided
!> and compiled before this module.
!>
!> Integrates a set of equations of motion using a 10th order accurate Gauss-Legendre integrator.
!> These integrators are a special case of implicit Runge-Kutta methods, which solve
!> \f[ \frac{d{\bf y}}{dt} = \f]
!> where the integrator is defined by the so-called Butcher tableau
!> \f[ B \f]
!>
!> As well, timing results that are useful for benchmarking various approaches can be included at compile time.
!> These are included by defining the preprocessor flag BENCHMARK at compile time.
!> Other optimisation flags are
!> USEBLAS - Use BLAS for matrix multiplication and matrix-vector multiplication instead of matmul
!> ALLOCATE_G - Allocate required matrices for storing intermediate values of the fields
!>
!> Various additional parameters may be used to control the efficiency of the integrator.
!> These include
!>
!> @todo
!> @arg Describe this module in more detail.  Then copy it over to the library I'm writing.
!> @arg Clean things up in here a bit more
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#include "optimizations.h"

module integrator
  use constants
  use eom

  implicit none
  integer :: numVar
  integer, parameter :: order=5
#ifdef ALLOCATE_G
  real(dl), allocatable, dimension(:,:) :: g
#ifdef USEBLAS
  real(dl), allocatable, dimension(:,:) :: gtmp
#endif
#else
  real(dl), dimension(nvar,order) :: g
#ifdef USEBLAS
  real(dl), dimension(nvar,order) :: gtmp
#endif
#endif

  integer, parameter :: niter = 8
  real(dl), parameter :: a(order,order) = reshape( (/ &
           0.5923172126404727187856601017997934066D-1, -1.9570364359076037492643214050884060018D-2, &
           1.1254400818642955552716244215090748773D-2, -0.5593793660812184876817721964475928216D-2, &
           1.5881129678659985393652424705934162371D-3,  1.2815100567004528349616684832951382219D-1, &
           1.1965716762484161701032287870890954823D-1, -2.4592114619642200389318251686004016630D-2, &
           1.0318280670683357408953945056355839486D-2, -2.7689943987696030442826307588795957613D-3, &
           1.1377628800422460252874127381536557686D-1,  2.6000465168064151859240589518757397939D-1, &
           1.4222222222222222222222222222222222222D-1, -2.0690316430958284571760137769754882933D-2, &
           4.6871545238699412283907465445931044619D-3,  1.2123243692686414680141465111883827708D-1, &
           2.2899605457899987661169181236146325697D-1,  3.0903655906408664483376269613044846112D-1, &
           1.1965716762484161701032287870890954823D-1, -0.9687563141950739739034827969555140871D-2, &
           1.1687532956022854521776677788936526508D-1,  2.4490812891049541889746347938229502468D-1, &
           2.7319004362580148889172820022935369566D-1,  2.5888469960875927151328897146870315648D-1, &
           0.5923172126404727187856601017997934066D-1 /) , [order,order])
  real(dl), parameter :: b(order) = (/ &
           1.1846344252809454375713202035995868132D-1,  2.3931433524968323402064575741781909646D-1, &
           2.8444444444444444444444444444444444444D-1,  2.3931433524968323402064575741781909646D-1, &
           1.1846344252809454375713202035995868132D-1 /)

contains

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! DESCRIPTION
  !> @brief
  !> Evolve the fields for time step dt with 10th order Gauss-Legendre integrator
  !>
  !> Use a 10th order accurate Gauss-Legendre integrator to evolve a set of variables
  !> forward in time.  The equations of motion must be provided by and external
  !> subroutine called derivs which takes two array variables of length numVar.
  !> The evolution is implemented as
  !> \f{eqnarray}{ \f}
  !>
  !> @param[in,out] y (array(double), numVar) A vector storing the current values of the variables
  !> @param[in] dt (double) The time step to evolve through
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine gl10( y, dt )
    real(dl), intent(inout) :: y(1:nVar)
    real(dl), intent(in) :: dt
!    external :: derivs
!    interface
!       subroutine derivs(yc,yp)
!         real(dl), dimension(1:numVar), intent(in) :: yc
!         real(dl), dimension(1:numVar), intent(out) :: yp
!       end subroutine derivs
!    end interface

    integer :: i,k

    g = 0._dl
    do k = 1,niter
#ifdef USEBLAS
! DGEMM barfs if I try directly storing the result in g
! Unfortunately, the extra copy slows the code down, so I need to fix this by doing some sort of staggering
       call DGEMM('N','N',numVar,order,order,1.,g,numVar,a,order,0.,gtmp,numVar)
#else
       g = matmul(g,a)
#endif
       do i=1,order
#ifdef USEBLAS
          call derivs( y+gtmp(:,i)*dt, g(:,i) )
#else
          call derivs( y+g(:,i)*dt, g(:,i) )
#endif
       enddo
    enddo
#ifdef USEBLAS
    call DGEMV('N',numVar,order,dt,g,numVar,b,1,1._dl,y,1)
#else
    y = y + matmul(g,b)*dt
#endif
  end subroutine gl10

  !> @brief
  !> Allocate the necessary temporary arrays for performing the iterative implicit time evolution.
  !> Should be called before doing time evolution
  !>
  !> @param[in] nv (integer) The number of variables that are being evolved
  subroutine init_integrator(nv)
    integer, intent(in) :: nv
    numVar = nv
#ifdef ALLOCATE_G
    allocate(g(nv,order))
#ifdef USEBLAS
    allocate(gtmp(nv,order))
#endif
#endif
  end subroutine init_integrator

end module integrator
