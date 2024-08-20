import os, sys
sys.path.append('/home/dpirvu/project/paper_prefactor/bubble_codes/')
from bubble_tools import *
from plotting import *

### Lattice Params
nLat = 2048
lenLat = 100.

phieq  = 0.
dx     = lenLat/nLat
dk     = 2.*np.pi/lenLat
knyq   = nLat//2+1
kspec  = knyq * dk
dtout  = dx
lightc = dx/dtout

# Lattice
lattice = np.arange(nLat)
xlist   = lattice*dx
klist   = np.roll((lattice - nLat//2+1)*dk, nLat//2+1)

#### SPECTRA
w2    = lambda m2: m2 + (2./dx**2.) * (1. - np.cos(klist * dx) )
w2std = lambda m2: m2 + klist**2.
pspec = lambda te,m2: np.array([te / lenLat / w2(m2)[k] if kk!=0. and kk < kspec else 0. for k,kk in enumerate(klist)])
stdev = lambda te,m2: np.sqrt(np.sum(pspec(te,m2)))

def get_general_model(case='free'):
    if case=='free':
        nTimeMAX = 512
        tempList = np.array([0.1])
        minSim, maxSim = 0, 50
        massq  = lambda te: 1.

        V    = lambda x: 0.5*x**2.
        Vinv = lambda x: - 0.5*x**2.
        dV   = lambda x: x

    elif case=='plus':
        nTimeMAX = 1048576
        tempList = np.array([0.2])
        minSim = 0
        maxSim = 10
        massq  = lambda te: 1. + te*3./2.

        V     = lambda x:   0.5*x**2. + 0.25*x**4.
        Vinv  = lambda x: - 0.5*x**2. - 0.25*x**4.
        dV    = lambda x:       x     +      x**3.

    elif case=='minus':
        nTimeMAX = 262144
        tempList = np.array([0.10, 0.115, 0.13, 0.2])
        minSim = 0
        maxSim = 2000
        massq  = lambda te: 1. - te*3./2.

        V     = lambda x:   0.5*x**2. - 0.25*x**4. + x**6. * 1e-4
        Vinv  = lambda x: - 0.5*x**2. + 0.25*x**4. - x**6. * 1e-4
        dV    = lambda x:       x     -      x**3. + x**5. * 6e-4

    right_Vmax = sco.minimize_scalar(Vinv, bounds=(0., 2.), method='bounded')
    right_Vmax = right_Vmax.x
    return tempList, massq, right_Vmax, V, dV, Vinv, nTimeMAX, minSim, maxSim


def get_model(tempList, massq, right_Vmax, V, dV, Vinv, nTimeMAX, minSim, maxSim, tmp):
    temp = tempList[tmp]
    m2 = massq(temp)
    sigmafld = stdev(temp, m2)
    return temp, m2, sigmafld


### Paths to files
root_dir      = '/gpfs/dpirvu/prefactor/'
root_sim_dir  = '/gpfs/dpirvu/prefactor/bubble_formation_'

batch_params  = lambda nL,m2,te: 'x'+str(int(nL))+'_m2eff'+str('%.4f'%m2)+'_T'+str('%.4f'%te) 
triage_pref   = lambda minS,maxS,nTM: '_minSim'+str(minS)+'_maxSim'+str(maxS)+'_up_to_nTMax'+str(nTM)
sims_interval = lambda minS,maxS: '_minSim'+str(minS)+'_maxSim'+str(maxS)

solution_sim_location  = lambda nL,m2,te,sim: root_dir + batch_params(nL,m2,te) + '_sim' +str(sim)+'instanton_critical_fields.dat'
precursor_sim_location = lambda nL,m2,te,sim: root_dir + batch_params(nL,m2,te) + '_sim' +str(sim)+'instanton_sub_critical_fields.dat'

sim_location       = lambda nL,m2,te,sim: root_dir + batch_params(nL,m2,te) + '_sim'      +str(sim)+'_fields.dat'
clean_sim_location = lambda nL,m2,te,sim: root_dir + batch_params(nL,m2,te) + '_clean_sim'+str(sim)+'_fields.npy'
data_sim_location  = lambda nL,m2,te,sim: root_dir + batch_params(nL,m2,te) + '_data_sim' +str(sim)+'_fields.npy'
rest_sim_location  = lambda nL,m2,te,sim: root_dir + batch_params(nL,m2,te) + '_rest_sim' +str(sim)+'_fields.npy'

bubble_sim_location= lambda nL,m2,te,sim: root_sim_dir + batch_params(nL,m2,te) + '_sim'  +str(sim)+'_fields.dat'

directions_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_directions.npy'
velocities_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_velocitiesCOM.npy'
average_file    = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_average_bubble.npy'

sims_decayed_file   = lambda nL,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,te) + triage_pref(minS,maxS,nTM) + '_sims_decayed.npy'
sims_notdecayed_file= lambda nL,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,te) + triage_pref(minS,maxS,nTM) + '_sims_notdecayed.npy'
decay_times_file    = lambda nL,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,te) + triage_pref(minS,maxS,nTM) + '_timedecays.npy'
decaytimes4real_file= lambda nL,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,te) + triage_pref(minS,maxS,nTM) + '_timedecays4real.npy'
init_cond_file      = lambda nL,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,te) + triage_pref(minS,maxS,nTM) + '_initconds.npy'
prebubble_cond_file = lambda nL,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,te) + triage_pref(minS,maxS,nTM) + '_prebubbleconditions.npy'
continuetevol_cond_file = lambda nL,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,te) + triage_pref(minS,maxS,nTM) + '_continuetevolconditions.npy'
bubble_cond_file    = lambda nL,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,te) + triage_pref(minS,maxS,nTM) + '_bubbleconditions.npy'

generate_bubble_seeds_file = lambda nL,m2,te,sim: root_dir + batch_params(nL,m2,te) + '_sim' +str(sim)+'_bubble_seeds.txt'
generate_continuetevol_seeds_file = lambda nL,m2,te,sim: root_dir + batch_params(nL,m2,te) + '_sim' +str(sim)+'_continuation_seeds.txt'

powspec_tlist_file= lambda nL,m2,te,minS,maxS: root_dir + batch_params(nL,m2,te) + sims_interval(minS,maxS) + '_powspec.npy'
varians_tlist_file= lambda nL,m2,te,minS,maxS: root_dir + batch_params(nL,m2,te) + sims_interval(minS,maxS) + '_variances.npy'
stdinit_file      = lambda nL,m2,te,minS,maxS: root_dir + batch_params(nL,m2,te) + sims_interval(minS,maxS) + '_fld_init_std.npy'
emt_tlist_file    = lambda nL,m2,te,minS,maxS: root_dir + batch_params(nL,m2,te) + sims_interval(minS,maxS) + '_emt.npy'
stdemt0_tlist_file= lambda nL,m2,te,minS,maxS: root_dir + batch_params(nL,m2,te) + sims_interval(minS,maxS) + '_stdemt0.npy'
toten_tlist_file  = lambda nL,m2,te,minS,maxS: root_dir + batch_params(nL,m2,te) + sims_interval(minS,maxS) + '_toten.npy'

instanton_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_instanton_profile.npy'
subcrit_ansol_instanton_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_subinstanton_profile.npy'
ansol_instanton_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_instanton_profile.npy'
crittimes_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_critical_timeslice.npy'
critenerg_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_critical_energy.npy'
critfield_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_critical_field_and_momentum.npy'

# field, momentum
normal = np.array([phieq, 0., 0.])

###############################
# Functions below are not debugged #

def get_gradient(fld, lenLat, nLat):
    nnx = nLat//2+1
    dx  = lenLat/nLat
    dk  = 2.*np.pi/lenLat

    lattice = np.arange(nLat)
    klist   = np.roll((lattice - nLat//2+1)*dk, nLat//2+1)
    keffsq  = (2./dx**2.) * (1. - np.cos(klist * dx) )

    fftfldr= np.fft.rfft(fld, axis=-1, norm=None)
    fftgrd = (keffsq[:nLat//2+1]**0.5 * 1.j )[None,:] * fftfldr
    grd = np.fft.irfft(fftgrd, axis=-1, norm=None)
    return fftgrd, grd

def get_laplacian(fld, lenLat, nLat):
    nnx = nLat//2+1
    dx  = lenLat/nLat
    dk  = 2.*np.pi/lenLat

    lattice = np.arange(nLat)
    klist   = np.roll((lattice - nLat//2+1)*dk, nLat//2+1)
    keffsq  = (2./dx**2.) * (1. - np.cos(klist * dx) )

    fftfldr =  np.fft.rfft(fld, axis=-1, norm=None)
    fftlap  = -keffsq[:nLat//2+1][None,:] * fftfldr
    lap     =  np.fft.irfft(fftlap, axis=-1, norm=None)
    return fftlap, lap

def get_simulation_energy(real, nLat, lenLat, V):
    nC, nT, nN    = np.shape(real)
    fld, mom = real[0], real[1]

    _, grd = get_gradient(fld, lenLat, nLat)

    KEN_data = dx * np.sum(0.5*mom**2., axis=-1)
    GEN_data = dx * np.sum(0.5*grd**2., axis=-1)
    PEN_data = dx * np.sum(V(fld), axis=-1)
    TEN_data = KEN_data + GEN_data + PEN_data
    return KEN_data, GEN_data, PEN_data, TEN_data
