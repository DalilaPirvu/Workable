module constants
  integer, parameter :: dl = kind(1.d0)
  real(dl), parameter :: twopi =  6.2831853071795864769252867665590

  integer, parameter :: nSim = 2100
  integer, parameter :: startSim = 2000

  integer, parameter :: nLat     = 2048
  integer, parameter :: nTimeMax = 1280

  real(dl), parameter :: m2    = 1.
  real(dl), parameter :: temp  = 0.13
  real(dl), parameter :: lenLat= 80

  real(dl), parameter :: m2eff = m2 - temp*3./2.

  real(dl), parameter :: dx  = lenLat/nLat
  real(dl), parameter :: dk  = twopi/lenLat
  real(dl), parameter :: alph = 16.   ! courrant number

  integer, parameter :: kspec = nLat/2
  integer, parameter :: nFld  = 1
  integer, parameter :: nVar  = 2*nFld*nLat+1

  real(dl), parameter :: norm = 1. / sqrt(nLat * dx)

end module constants

