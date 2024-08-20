!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Module for Generating Gaussian Random Fields in Various Dimensions !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!> @author
!> Jonathan Braden, University College London
!>
! DESCRIPTION
!> @brief
!> A module for generating realisations of Gaussian Random Fields in various dimensions
!>
!> Generate realisations of Gaussian Random Fields in 1,2, or 3 dimension.  Two methods are provided:
!> @arg Sample individual Fourier Modes
!> @arg Convolve a realisation of white noise with a spherically symmetric real-space kernel
!> The amplitudes are normalised so
!> \f$\frac{1}{dk^{d/2}n^d} \f$
!> @todo Finish this
!>
!> @par Generating correlated Gaussian Random Variables
!>
!> Suppose that we want to transform an n-vector \f$ x \f$ of zero mean random variables with \f$\langle xx^T \rangle = \mathbb{I}\f$
!> into a new n-vector \f$ y \f$ with a given covariance \f[\hat{C} \equiv \langle yy^T \rangle\f].
!> For simplicity, let's restrict to linear transformations \f[y = Mx\f].
!> We immediately see that
!> \f[ \hat{C} = \langle Mxx^TM^T \rangle = M\langle xx^T\rangle M^T = MM^T \f]
!> so that \a any such matrix \f$M\f$ will work.
!> Two particularly convenient options are
!> @arg I: Choleski Decomposition \n \f$ \hat{C} = LL^T\f$ with \f$L\f$ lower triangular
!> @arg II: Square-Root Decomposition \n \f$\hat{C} = S^2 \f$ with \f$ S\f$ symmetric and positive definite.  This is easily implemented through matrix diagonalisation.
!>   \f[ \hat{C} = UDU^T = UD^{1/2}U^TUD^{1/2}U^T \f]
!> so that
!>   \f[S = UD^{1/2}U^T\f]
!> @arg III: Continuum of Solutions - \n Given a matrix \f$M\f$ with \f$\hat{C} = MM^T\f$ and an arbitrary \f$U^TU = \mathbb{I} = UU^T\f$ the \f$M' = MU\f$ will also work.
!>
!> In the case of an inhomogeneous latticised field, we must therefore diagonalise the \a full covariance matrix, which may be very numerically expensive.
!> For this case the Cholesky decomposition may be the preferred approach.
!> However, this produces a transformation that is asymmetric between the various random variables and may introduce bias into the sampling?
!> The transformation based on square-rooting the covariance matrix is instead more symmetric in its treatment of each of the original diagonal random variables.
!>
!> For the case of a homogeneous field, we can transform to Fourier space to block diagonalise the
!> covariance matrix and consider the correlations of each individual Fourier mode independently.
!> If we only have a small number of fields to correlate, the required transformation matrices can be computed analytically (or else diagonalisation will not be too numerically expensive).
!> Here we provide a few examples
!> @arg 2-field square root
!> @arg 2-field Cholesky
!> @arg 4-field square root
!> @arg 4-field Cholesky
!> 
!>  
! TO DO
!> @todo
!> @arg Write convolution method in 2D (needs evaluation of Bessels, or else something smarth)
!> @arg Add symmetry based approaches in 2D and 3D
!> @arg Find a better method to pass in the desired spectrum (especially in higher dimensions
!>      where a function is more convenient than having to interpolate from some passed array
!> @arg Find a better way to ensure proper normalisation associated with the discretisation of space
!> @arg Can I implement a non-canonical imaginary part of the two point correlator by being smart here?  This would allow for non-canonical variables.  Or does this let me estimate it?
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module gaussianRandomField
  use, intrinsic :: iso_c_binding
  use constants
  use fftw3
  implicit none
  logical :: seed_init = .false.
contains

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> @brief
  !> Generate a 1D GRF with desired spectrum (passed as an array).
  !>
  !> Generate a realisation of a one-dimensional Gaussian Random Field with the given spectrum.
  !> The amplitude of the field at each wavenumber is passed as an array.
  !> 
  !> An option to generate the field either by convolving with a white noise filter or directly by sampling the Fourier modes is included.
  !> This is controlled by the logical parameter convolve.
  !>
  !> The fields can be chosen to have no symmetry, even symmetry about the origin (ie. cosines) or odd symmetry about the origin (ie. sines)
  !> This is controlled by the integer parameter symmetry.
  !> 
  !> @param[in,out] field (C_DOUBLE, 1D array) Array storing the field values at each point
  !> @param[in] spectrum (double, 1D array) Array storing values of the spectrum at each wavenumber
  !> @param[in] convolve (Boolean) @arg True:  Generate field with convolution based method
  !>                               @arg False: Initialise Fourier amplitudes directly
  !> @param[in] symmetry (Integer) Specify symmetry of the generated field.  Options are:
  !> @arg                          1 - even symmetry about midpoint
  !> @arg                          2 - odd symmetry about midpoint
  !> @arg                          other - no symmetry
  !> @param[in] stride (Integer)   Specify the first stride to take in the random numbers.  This is used to self-consistently generate the same initial conditions with different grid spacings.
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine generate_1dGRF(field, spectrum, sym_in, initStride) !last two args are optional
    real(C_DOUBLE), dimension(:), intent(inout) :: field
    real(dl), dimension(:), intent(in) :: spectrum
    integer, optional :: sym_in, initStride
    integer :: symmetry, nLat, nn, stride, nnk !stride = 'a step or stage in progress toward an aim'
    logical :: cut
    real(dl), allocatable, dimension(:) :: amp, phase
    complex(dl), allocatable, dimension(:) :: deviate
    complex(C_DOUBLE_COMPLEX), allocatable :: Fk(:)
    type(C_PTR) :: fft_plan

    symmetry = 0
    if (present(sym_in)) symmetry = sym_in
    nLat = size(field); nn = nLat/2 + 1; nnk = size(spectrum); cut = .false.
    if (nn > nnk) then
       print*,"Warning spectrum is smaller than the number of required Fourier modes in 1dGRF.  Additional high frequency modes will not be sampled."
       cut = .true.
    endif
    if (.not.present(initStride)) then; stride = nnk; else; stride = initStride; endif
    if ( stride > nnk ) then; print*,"Stride is larger than number of Fourier modes.  Setting to nyquist"; stride = nnk; endif
    if (.not.seed_init) then
       print*,"Error, random number generator not initialized.  Call initialize_rand, using default seed values"
       call initialize_rand(75,13)
    endif
    allocate(Fk(1:nn))
    fft_plan = fftw_plan_dft_c2r_1d(nLat, Fk, field, FFTW_ESTIMATE)
    if (.not.seed_init) then
       print*,"Error, random number generator not initialized.  Calling initialize_rand, using default seed values"
       call initialize_rand(75,13)
    endif

    ! Generate Gaussian random deviates using Box-Muller
    ! Normalization chosen so that < Re(x)^2+Im(x)^2 > = 1
    ! generate random \rho and \phi for \psi = \sqrt(\rho) e^{i \phi}
    allocate( amp(1:nnk),phase(1:nnk),deviate(1:nnk) )
    call random_number(amp(1:stride)) 
    call random_number(phase(1:stride))
    if (stride < nnk) then
      call random_number( amp(stride+1:nnk) )
      call random_number( phase(stride+1:nnk) )
    endif
    select case (symmetry)
     case (1)
       deviate = dreal( sqrt(-2.*log(amp))*exp(iImag*twopi*phase) )
     case (2)
       deviate = iImag*dimag( sqrt(-2.*log(amp))*exp(iImag*twopi*phase) )
     case default
       deviate = sqrt(-log(amp))*exp(iImag*twopi*phase) 
    !construct gaussian random deviates with unit variance \alpha_n in paper
    end select

    ! Now produce the field with the given spectrum
    Fk(1:nnk) = deviate * spectrum
!    Fk(1:nnk) = spectrum
    Fk(1) = 0.
    Fk(nnk+1:nn) = 0.
    call fftw_execute_dft_c2r(fft_plan, Fk, field)

  end subroutine generate_1dGRF

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Warning, this subroutine isn't currently threadsafe
  subroutine initialize_rand(seed, seedfac)
    integer, intent(in) :: seed, seedfac
    integer :: nseed, i
    integer, allocatable, dimension(:) :: seeds

    seed_init=.true.
    call random_seed(size=nseed)
    allocate(seeds(1:nseed))
    seeds = seed + seedfac*(/ (i-1, i=1, nseed) /)
    call random_seed(put=seeds)
    deallocate(seeds)
  end subroutine initialize_rand
end module gaussianRandomField
