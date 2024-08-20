import os,sys
sys.path.append('/home/dpirvu/python_stuff/')
sys.path.append('/home/dpirvu/project/paper_prefactor/bubble_codes/')

from plotting import *
from bubble_tools import *
from experiment import *

# Classify decays
get_partial_stats = False
get_all_stats     = False
get_bubble_seeds  = False
get_bubble_from_sims = True

case = 'minus'
tmp=2

general = get_general_model(case)
tempList, massq, right_Vmax, V, dV, Vinv, nTimeMAX, minSim, maxSim = general

maxSim = (1000 if tmp == 0 else 2000 if tmp==1 else 3000 if tmp==2 else 2000)
nTimeMAX = (262144 if tmp!=1 else 131072)

temp, m2, sigmafld = get_model(*general, tmp)
exp_params = np.array([nLat, m2, temp])
print('General params', tempList, right_Vmax, nTimeMAX, minSim, maxSim)
print('Experiment', exp_params)

if get_partial_stats:
    aa=0
    div=1

    simList = np.array(np.linspace(minSim, maxSim, div+1), dtype='int')
    divdata = np.array([(ii,jj) for ii,jj in zip(simList[:-1], simList[1:])])
    asim, bsim = divdata[aa]

    for sim in np.arange(asim, bsim):
        path2sim = sim_location(*exp_params, sim)
        path2DATAsim = data_sim_location(*exp_params, sim)

        if os.path.exists(path2sim):
            tdecay, outcome, initcond, real, prebubble, bubble = get_realisation(nLat, nTimeMAX, path2sim)
            np.save(path2DATAsim, np.array([sim, tdecay, outcome, initcond, prebubble, bubble]), allow_pickle=True)
            print('Simulation', sim, ', outcome', outcome, ', tdecay:', tdecay)


if get_all_stats:
    ALLoutcomes, ALLdecaytimes, init_conditions, ALLbubbleSEEDS, ALLdetectedbubbles, reinit_conditions = [], [], [], [], [], []
    for sim in range(minSim, maxSim):
        
        path2DATAsim = data_sim_location(*exp_params, sim)
        if os.path.exists(path2DATAsim):
            sim, tdecay, outcome, initcond, prebubble, bubble = np.load(path2DATAsim, allow_pickle=True)
            print(sim, tdecay, outcome)

            ALLoutcomes.append([sim, outcome])
            ALLdecaytimes.append([sim, tdecay])

            init_conditions.append([sim, initcond[0,:], initcond[1,:]])
            if outcome!=2:
                ALLbubbleSEEDS.append([sim, prebubble[0,:], prebubble[1,:]])
                ALLdetectedbubbles.append([sim, bubble[0,:], bubble[1,:]])
            else:
                reinit_conditions.append([sim, bubble[0,:], bubble[1,:]])

    np.save(sims_decayed_file(*exp_params, minSim, maxSim, nTimeMAX),       ALLoutcomes,        allow_pickle=True)
    np.save(decay_times_file(*exp_params, minSim, maxSim, nTimeMAX),        ALLdecaytimes,      allow_pickle=True)
    np.save(init_cond_file(*exp_params, minSim, maxSim, nTimeMAX),          init_conditions,    allow_pickle=True)
    np.save(prebubble_cond_file(*exp_params, minSim, maxSim, nTimeMAX),     ALLbubbleSEEDS,     allow_pickle=True)
    np.save(bubble_cond_file(*exp_params, minSim, maxSim, nTimeMAX),        ALLdetectedbubbles, allow_pickle=True)
    np.save(continuetevol_cond_file(*exp_params, minSim, maxSim, nTimeMAX), reinit_conditions,  allow_pickle=True)
    print('All saved.')

if get_bubble_seeds:
    ALLbubbleSEEDS = np.load(prebubble_cond_file(*exp_params, minSim, maxSim, nTimeMAX), allow_pickle=True)
    for ii, (sim, bfld, bmom) in enumerate(ALLbubbleSEEDS):
        print(sim, ii, np.shape(bfld))
        filename = generate_bubble_seeds_file(*exp_params, sim)
        data2save = np.array([bfld, bmom]).T
        save_txt_file(filename, data2save)
        print('Created', sim, generate_bubble_seeds_file(*exp_params, sim))

    ALLcontinuationSEEDS = np.load(continuetevol_cond_file(*exp_params, minSim, maxSim, nTimeMAX), allow_pickle=True)
    for ii, (sim, bfld, bmom) in enumerate(ALLcontinuationSEEDS):
        print(sim, ii, np.shape(bfld))
        filename = generate_continuetevol_seeds_file(*exp_params, sim)
        data2save = np.array([bfld, bmom]).T
        save_txt_file(filename, data2save)
        print('Created', sim, generate_continuetevol_seeds_file(*exp_params, sim))

if get_bubble_from_sims:
    ALLoutcomes   = np.load(sims_decayed_file(*exp_params, minSim, maxSim, nTimeMAX), allow_pickle=True)
    ALLdecaytimes = np.load(decay_times_file(*exp_params, minSim, maxSim, nTimeMAX),  allow_pickle=True)

    ALLdecays4real = []
    for ii, (sim, outcome) in enumerate(ALLoutcomes):
        print(sim, outcome)
        path2sim = bubble_sim_location(*exp_params, sim)
        path2CLEANsim = clean_sim_location(*exp_params, sim)

        try:
            real, outcomenew = get_bubble_realisation(nLat, path2sim)

            tdecay = get_decay_time(real)
            real = centre_bubble(real, tdecay)
            np.save(path2CLEANsim, real, allow_pickle=True)

            tprevious = ALLdecaytimes[ii][-1]
            tprevious = (tprevious // 1024) * 1024
            ALLdecays4real.append([sim, tdecay + tprevious])
            print('Simulation', sim, 'saved!', outcome, outcomenew, 'tdecay, tprevious, sum', tdecay, tprevious, tdecay + tprevious)
        
        except:
            continue

    np.save(decaytimes4real_file(*exp_params, minSim, maxSim, nTimeMAX), ALLdecays4real, allow_pickle=True)

print('ALL DONE.')
