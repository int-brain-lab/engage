import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from one.api import ONE
from collections import Counter
import ssm
from sklearn.preprocessing import scale
import numpy.random as npr
from collections import defaultdict
from brainwidemap import bwm_query
from matplotlib.ticker import MultipleLocator

########
'''
Code adapted from 
https://github.com/zashwood/glm-hmm/tree/main

install ssm from Zoe's fork like so:
https://github.com/zashwood/ssm
'''
########



one = ONE()
pth_eng = Path(one.cache_dir, 'engaged')
pth_eng.mkdir(parents=True, exist_ok=True)


###########
### utils
###########

def compute_dwell_times_with_states(state_probs_list, congru=True):
    """
    Compute dwell times and their corresponding states from posterior state probabilities.

    Parameters:
        state_probs_list : list of np.ndarray
            Each array is (n_trials, n_states), rows sum to 1.

        congru: if True, compute dwell times only for congruent states
                if false, compute for incongruent states,
                if none, compute for all states

    Returns:
        dwell_info : np.ndarray of shape (n_dwell_periods, 2)
            Column 0: dwell time (int)
            Column 1: corresponding state (int)
    """
    dwell_info = []

    for session_probs in state_probs_list:
        state_seq = np.argmax(session_probs, axis=1)

        current_state = state_seq[0]
        count = 1

        for s in state_seq[1:]:
            if s == current_state:
                count += 1
            else:
                dwell_info.append([count, current_state])
                current_state = s
                count = 1
        dwell_info.append([count, current_state])  # final dwell period

    return np.array(dwell_info)


def compute_dwell_times_from_df(df, congru=None):
    """
    Computes dwell times from a DataFrame with state probabilities and trial info.
    
    Parameters:
        df : pd.DataFrame
            Must include columns: ['p_state1', ..., 'rewarded', 'probabilityLeft', ...]
        congru : True, False, or None
            If True: keep only congruent choices
            If False: keep only incongruent
            If None: use all
    Returns:
        dwell_info : np.ndarray of shape (n_runs, 2)
            column 0: dwell time
            column 1: state index (int)
    """
    p_cols = sorted([col for col in df.columns if col.startswith('p_state')],
                    key=lambda x: int(x.split('p_state')[1]))

    # Assign most likely state to each trial
    df = df.copy()
    df['state'] = df[p_cols].values.argmax(axis=1)

    # Determine choice direction
    df['choice_right'] = (df['contrastLeft'].isna() & (df['rewarded'] == 1)) | \
                         (df['contrastRight'].isna() & (df['rewarded'] == -1))

    if congru is not None:
        congruent = (
            ((df['probabilityLeft'] == 0.8) & (~df['choice_right'])) |
            ((df['probabilityLeft'] == 0.2) & (df['choice_right']))
        )
        if congru is True:
            df = df[congruent]
        elif congru is False:
            df = df[~congruent]

    # Compute dwell times
    state_seq = df['state'].values
    dwell_info = []
    current_state = state_seq[0]
    count = 1
    for s in state_seq[1:]:
        if s == current_state:
            count += 1
        else:
            dwell_info.append([count, current_state])
            current_state = s
            count = 1
    dwell_info.append([count, current_state])  # final run

    return np.array(dwell_info)


def partition_data_by_session(inpt, y, mask, session):
    '''
    Partition inpt, y, mask by session
    :param inpt: arr of size TxM
    :param y:  arr of size T x D
    :param mask: Boolean arr of size T indicating if element is violation or
    not
    :param session: list of size T containing session ids
    :return: list of inpt arrays, data arrays and mask arrays, where the
    number of elements in list = number of sessions and each array size is
    number of trials in session
    '''
    inputs = []
    datas = []
    indexes = np.unique(session, return_index=True)[1]
    unique_sessions = [session[index] for index in sorted(indexes)]
    counter = 0
    masks = []
    for sess in unique_sessions:
        idx = np.where(np.array(session) == sess)[0]
        counter += len(idx)
        inputs.append(inpt[idx, :])
        datas.append(y[idx, :])
        masks.append(mask[idx, :])
    assert counter == inpt.shape[0], "not all trials assigned to session!"
    return inputs, datas, masks


def create_violation_mask(violation_idx, T):
    """
    Return indices of nonviolations and also a Boolean mask for inclusion (1
    = nonviolation; 0 = violation)
    :param test_idx:
    :param T:
    :return:
    """
    mask = np.array([i not in violation_idx for i in range(T)])
    nonviolation_idx = np.arange(T)[mask]
    mask = mask + 0
    assert len(nonviolation_idx) + len(
        violation_idx
    ) == T, "violation and non-violation idx do not include all dta!"
    return nonviolation_idx, np.expand_dims(mask, axis=1)


def get_animal_name(eid):
    details = one.get_details(eid)
    animal = details['subject']
    return animal


def get_raw_data(eid):
    animal = get_animal_name(eid)
    trials = one.load_object(eid, 'trials')
    return (animal, str(one.eid2path(eid)), trials['contrastLeft'], 
        trials['contrastRight'], trials['feedbackType'], trials['choice'],
        trials['probabilityLeft'])

##########
### BWM processing; i.e. load and process trial data from multiple sessions for one mouse
##########


def process_bwm_mouse(animal, rerun=False):
    '''
    Load and process trial data from multiple sessions for one mouse.
    Normalize contrast column across all sessions (column 0).
    
    Parameters:
        eids : list of str
            List of experiment IDs for the same animal.

    Returns:
        inpt : ndarray (T, 3)
            Input matrix [normalized contrast, prev_choice, WSLS].
        y : ndarray (T, 1)
            Remapped choices.
        session : list of str
            Session labels per trial.

    Get eids for animal from BWM set
    E.g. animal = 'NYU-11' for biased trials (all trials)
    
    '''

    # check if animal file
    pss = Path(pth_eng, 'per_animal_input', f'{animal}.npy')
    if pss.exists() and not rerun:
        data = np.load(pss, allow_pickle=True).flat[0]
        return data['inpt'], data['y'], data['session']
    else:    
        print('processing new animal data for ', animal)

        oo = bwm_query()
        eids = np.unique(oo[oo['subject'] == animal]['eid'].values)

        all_inpt = []
        all_y = []
        all_session = []

        kk = 1
        for eid in eids:
            animal, session_id, stim_left, stim_right, rewarded, choice, bias_probs = get_raw_data(eid)

            mask = np.ones(len(choice), dtype=bool)

            stim_left = np.nan_to_num(stim_left[mask], nan=0)
            stim_right = np.nan_to_num(stim_right[mask], nan=0)
            stim = stim_right - stim_left
            T = len(stim)

            # Remap choice: {1:0, -1:1, 0:-1}
            choice_map = {1: 0, -1: 1, 0: -1}
            choice_masked = np.array([choice_map[c] for c in choice[mask]])

            # Previous choice
            prev_choice = np.hstack([choice_masked[0], choice_masked[:-1]])
            prev_choice[prev_choice == -1] = npr.choice([0, 1])
            prev_choice_bin = 2 * prev_choice - 1

            # WSLS (Win-Stay, Lose-Shift) covariate
            reward = rewarded[mask]
            prev_reward = np.hstack([reward[0], reward[:-1]])
            wsls = prev_reward * prev_choice_bin
            wsls[wsls == 0] = -1  # handle ambiguous or violated trials

            inpt = np.column_stack([stim, prev_choice_bin, wsls])
            y = choice_masked[:, None].astype(int)
            session = [session_id] * T

            all_inpt.append(inpt)
            all_y.append(y)
            all_session.extend(session)

            print(f'Processed {kk} of {len(eids)} eids for {animal}')
            kk += 1
        
        if len(all_inpt) == 0:
            return np.zeros((0, 3)), np.zeros((0, 1)), []

        # Concatenate and normalize contrast column (column 0)
        inpt = np.vstack(all_inpt)
        inpt[:, 0] = scale(inpt[:, 0])  # z-score normalization
        y = np.vstack(all_y)
        session = all_session

        np.save(pss, {'inpt': inpt, 'y': y, 'session': session},allow_pickle=True)
    
        return inpt, y, session

############
### Model fitting and result plotting
############

def model_single_mouse(animal, run_description='K_2', unbiased=False):
    
    '''
    Train model on single mouse using parameters from notebook as
    starting weights
    animal = 'CSHL_001'
    animals = np.load('/home/mic/glm-hmm/data/ibl/data_for_cluster'
                      '/data_by_animal/animal_list.npz')['arr_0']
    run_description: 'K_2' for 2-state model, state 1 is engaged
    unbiased: to reproduce Zoe's results on unbiased trials, see her repo


    '''
    if unbiased:
        # for unbiased trials
        animal_folder = '/home/mic/glm-hmm/data/ibl/data_for_cluster/data_by_animal/'
        animal_file = (f'/home/mic/glm-hmm/data/ibl/data_for_cluster/data_by_animal/{animal}_processed.npz')
        
        container = np.load(animal_file, allow_pickle=True)
        data = [container[key] for key in container]
        inpt = data[0]
        y = data[1]
        session = data[2]

    else:
        # for biased trials
        inpt, y, session = process_bwm_mouse(animal)

    # append bias to input (currently contrast, last choice, Win–Stay / Lose–Shift covariate)
    inpt = np.hstack((inpt, np.ones((len(inpt), 1))))

    # Identify violations for exclusion:
    violation_idx = np.where(y == -1)[0]

    # y not used for anything (due to violation mask) 
    y[np.where(y == -1), :] = 1
    _, mask = create_violation_mask(violation_idx, inpt.shape[0])

    # Partition inputs by session
    inputs, datas, masks = partition_data_by_session(inpt, y, mask, session)
                  
    # Model parameters
    num_states = int(run_description[-1])             # K
    obs_dim = 1                 # D (for binary choice)
    input_dim = inpt.shape[1]   # M
    num_categories = 2          # C

    N_em_iters = 200           # Number of EM iterations
    transition_alpha = 2       # Concentration parameter for transitions
    prior_sigma = 2            # Prior std dev for weights
    global_fit = False         # Not using shared GLM weights
     

    # # paper
    glmhmm = ssm.HMM(num_states, obs_dim, input_dim, 
                           observations="input_driven_obs",
                           observation_kwargs=dict(C=num_categories,
                                                   prior_sigma=prior_sigma),
                           transitions="sticky",
                           transition_kwargs=dict(alpha=transition_alpha,
                                                  kappa=0))


    ## from paper scripts global fit
    full_params = [[np.array([-0.52493862, -1.64298306, -1.53708898])],
    [np.array([[-0.02608457, -4.25327563, -4.46282725],
            [-3.04799552, -0.05351097, -5.370778  ],
            [-3.09435783, -5.83956581, -0.04941527]])],
    np.array([[[-7.20614670e+00, -3.12209531e-01, -1.61175163e-03,
            1.43813500e-01]],
    
            [[-1.22616681e+00, -3.61431579e-01, -1.80390841e-01,
            1.70656106e+00]],
    
            [[-1.20628874e+00, -3.20944513e-01, -1.98757460e-01,
            -1.62621352e+00]]])]

    
    glmhmm.params = [
        [full_params[0][0][:num_states]],  # truncate pi0
        [full_params[1][0][:num_states, :num_states]],  # square transition matrix
        full_params[2][:num_states]  # GLM weights
    ]

    glmhmm.fit(datas,
                inputs=inputs,
                masks=masks,
                method="em",
                num_iters=N_em_iters,
                initialize=False,
                tolerance=10 ** -4)


    # Get expected states:
    posterior_probs = [glmhmm.expected_states(data=data, input=inpt)[0]
                    for data, inpt
                    in zip(datas, inputs)]      

    # save model
    if unbiased:
        gg = '/home/mic/glm-hmm/results/individual_fit_own'
        sss = Path(gg, run_description, 'params')
    else:
        sss = Path(pth_eng, run_description, 'params')

    sss.mkdir(parents=True, exist_ok=True)

    np.save(Path(sss, f'{animal}_model_params.npy'), 
                {'gen_weights': glmhmm.observations.params,
                'trans_mat': np.exp(glmhmm.transitions.log_Ps),
                'params': glmhmm.params,
                'posterior_probs': posterior_probs})


def plot_model_params(animal, run_description='K_2', unbiased=False):

    '''
    Plot model parameters for a single animal.
    e.g. animal = 'NYU-11' for unbiased = False. 
    '''

    # Load model parameters
    if unbiased:
        ttt = '/home/mic/glm-hmm/results/individual_fit_own'
    else:
        ttt = pth_eng

    animal_file = Path(ttt, run_description, 'params',
                       f'{animal}_model_params.npy')

    model_params = np.load(animal_file, allow_pickle=True).flat[0]

    gen_weights     = model_params['gen_weights']
    trans_mat       = model_params['trans_mat']
    posterior_probs = model_params['posterior_probs']

    # session & trial info
    n_sessions = len(posterior_probs)
    trials_per_session = [arr.shape[0] for arr in posterior_probs]
    trials_str = ", ".join(str(n) for n in trials_per_session)

    # get which state on each trial
    posterior_concat    = np.concatenate(posterior_probs)
    state_seq           = np.argmax(posterior_concat, axis=1)

    num_states = gen_weights.shape[0]
    # build full-length occupancy array
    vals, counts        = np.unique(state_seq, return_counts=True)
    state_occupancies   = np.zeros(num_states, dtype=int)
    state_occupancies[vals] = counts

    dwell_info           = compute_dwell_times_with_states(posterior_probs)

    #num_states = gen_weights.shape[0]
    input_dim  = gen_weights.shape[2]
    inpts      = ['stim','bias','prev_choice','WSLS']
    cols       = ['#ff7f00','#4daf4a','#377eb8'][:num_states]

    fig = plt.figure(figsize=(14, 3), dpi=80)

    def clean_spines(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 1: Generative weights
    ax1 = plt.subplot(1,5,1)
    for k in range(num_states):
        ax1.plot(gen_weights[k][0], marker='o', color=cols[k], lw=1.5)
    ax1.axhline(0, color='k', alpha=0.5, ls='--')
    ax1.set_xticks(range(input_dim))
    ax1.set_xticklabels(inpts, rotation=45, fontsize=10)
    ax1.set_ylabel("GLM weight")
    ax1.set_title("Generative weights")
    clean_spines(ax1)

    # 2: Transition matrix
    ax2 = plt.subplot(1,5,2)
    ax2.imshow(trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(num_states):
        for j in range(num_states):
            ax2.text(j,i,f"{trans_mat[i,j]:.2f}", ha='center', va='center', fontsize=10)
    ax2.set_xticks(range(num_states)); ax2.set_yticks(range(num_states))
    ax2.set_xticklabels([str(i+1) for i in range(num_states)], fontsize=9)
    ax2.set_yticklabels([str(i+1) for i in range(num_states)], fontsize=9)
    ax2.set_xlabel("t+1"); ax2.set_ylabel("t")
    ax2.set_title("Transition matrix")
    clean_spines(ax2)

    # 3: Occupancy
    ax3 = plt.subplot(1,5,3)
    ax3.bar(range(num_states), state_occupancies, color=cols, width=0.8)
    ax3.set_xticks(range(num_states))
    ax3.set_xticklabels([str(i+1) for i in range(num_states)], fontsize=10)
    ax3.set_xlabel("state"); ax3.set_ylabel("occupancy")
    ax3.set_title("State occupancy")
    clean_spines(ax3)

    # 4: Empirical dwell
    ax4 = plt.subplot(1,5,4)
    all_d = dwell_info[:,0]
    bins = np.histogram_bin_edges(all_d, bins=30)
    for k in range(num_states):
        dk = dwell_info[dwell_info[:,1]==k][:,0]
        ax4.hist(dk, bins=bins, histtype='step', color=cols[k], lw=1.5)
        ax4.axvline(np.median(dk), color=cols[k], ls='--', lw=1.2)
    ax4.set_xlabel("dwell (trials)"); ax4.set_title("Empirical dwell")
    clean_spines(ax4)

    # 5: Expected dwell
    ax5 = plt.subplot(1,5,5)
    P = np.exp(trans_mat) if np.all(trans_mat <= 0) else trans_mat
    exp_d = 1/(1 - np.diag(P))
    ax5.bar(range(num_states), exp_d, color=cols, width=0.8)
    ymax = exp_d.max() * 1.15
    ax5.set_ylim(0, ymax)
    for k,val in enumerate(exp_d):
        ax5.text(k, val + ymax*0.02, f"{val:.1f}", ha='center', fontsize=9)
    ax5.set_xticks(range(num_states))
    ax5.set_xticklabels([str(i+1) for i in range(num_states)], fontsize=10)
    ax5.set_ylabel("trials"); ax5.set_title("Expected dwell")
    clean_spines(ax5)

    plt.tight_layout(rect=[0,0,1,0.88])
    plt.suptitle(f"{animal}: {n_sessions} sessions | trials/session = [{trials_str}]",
                 fontsize=14)

    # Save
    if unbiased:
        save_dir = Path('/home/mic/glm-hmm/results/individual_fit_own',
                        run_description, 'pngs')
    else:
        save_dir = Path(pth_eng, run_description, 'pngs')
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f'{animal}_model_params.png', bbox_inches='tight')


def do_for_all(run_description='K_2', unbiased=False):
    '''
    Run model fitting and plotting for all animals
    run_description: 'standard_init' for standard initialization
    '''

    if unbiased:
        animals = np.load('/home/mic/glm-hmm/data/ibl/data_for_cluster'
                        '/data_by_animal/animal_list.npz')['arr_0']
    else:
        oo = bwm_query()
        animals = oo['subject'].unique()

    plt.ioff()
    k = 0
    for animal in animals:
        print(f'Processing {animal}')
        model_single_mouse(animal, run_description=run_description, unbiased=unbiased)
        plot_model_params(animal, run_description=run_description, unbiased=unbiased)
        plt.close()
        k += 1
        print(f'Processed {k} out of {len(animals)} animals')


def pool_behavioral_data():
    '''
    For all BWM mice, load trial data, save as dataframe.
    
    Returns:
        df_all: pd.DataFrame
            Dataframe with columns: 
            ['animal', 'eid', 'stimLeft', 'stimRight', 
             'rewarded', 'probabilityLeft', 'probabilityRight']
    '''
    columns = ['animal', 'eid', 'stimLeft', 'stimRight', 
               'rewarded', 'probabilityLeft', 'probabilityRight']

    data = []
    oo = bwm_query()
    all_rows = []

    k = 0
    for animal in oo['subject'].unique():
        eids = np.unique(oo[oo['subject'] == animal]['eid'].values)

        for eid in eids:
            try:
                trials = one.load_object(eid, 'trials')
            except Exception as e:
                print(f"Skipping {eid} due to load error: {e}")
                continue

            # Extract variables
            stimLeft = trials['contrastLeft']
            stimRight = trials['contrastRight']
            rewarded = trials['feedbackType']
            probLeft = trials['probabilityLeft']
            probRight = 1 - probLeft

            n_trials = len(stimLeft)

            for i in range(n_trials):
                row = {
                    'animal': animal,
                    'eid': eid,
                    'contrastLeft': stimLeft[i],
                    'contrastRight': stimRight[i],
                    'rewarded': rewarded[i],
                    'probabilityLeft': probLeft[i]
                }
                all_rows.append(row)
        
        print(f"Processed {animal} k={k+1} of {len(oo['subject'].unique())}")
        k += 1

    df_all = pd.DataFrame(all_rows, columns=columns)

    df_all.to_parquet(Path(pth_eng, 'bwm_behavioral_data.pqt'))



#######################
### meta analyses
#######################

def plot_dwell_times_hist_all(unbiased=False, run_description='K_2'):
    '''
    For all animals, load params file, compute dwell times
    and plot a histogram per state. Also show expected dwell times
    computed from transition matrix diagonals.
    '''

    if unbiased:
        # for unbiased trials
        param_dir = Path('/home/mic/glm-hmm/results/individual_fit_own/'
                         f'{run_description}/params')
    else:
        # for biased trials
        param_dir = Path(pth_eng, run_description, 'params')

    animal_files = sorted(param_dir.glob("*_model_params.npy"))                   
    if not animal_files:
        print("No model parameter files found.")
        return

    all_dwell_times = defaultdict(list)
    expected_dwell = defaultdict(list)

    for file in animal_files:
        animal = file.stem.replace('_model_params', '')
        model_params = np.load(file, allow_pickle=True).flat[0]
        posterior_probs = model_params['posterior_probs']
        trans_mat = model_params['trans_mat']

        # Compute empirical dwell times
        dwell_info = compute_dwell_times_with_states(posterior_probs)
        for dwell, state in dwell_info:
            all_dwell_times[state].append(dwell)

        # Compute expected dwell times
        P = np.exp(trans_mat) if np.all(trans_mat <= 0) else trans_mat
        for k in range(P.shape[0]):
            tau = 1 / (1 - P[k, k])
            expected_dwell[k].append(tau)

    num_states = len(all_dwell_times)
    colors = ['#ff7f00', '#4daf4a', '#377eb8'][:num_states]

    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # ---------- Empirical dwell time histogram ----------
    all_durations = np.concatenate([np.array(all_dwell_times[s]) for s in all_dwell_times])
    bins_empirical = np.arange(1, np.max(all_durations) + 2) - 0.5

    for state in sorted(all_dwell_times):
        data = np.array(all_dwell_times[state])
        axs[0].hist(data, bins=bins_empirical, histtype='step',
                    color=colors[state], linewidth=1.5,
                    label=f"State {state + 1}", density=True)

        med = np.median(data)
        axs[0].axvline(med, linestyle=':', color=colors[state], linewidth=1.2)

    axs[0].set_ylim(top=axs[0].get_ylim()[1] * 1.1)

    offsets = [0.97, 0.90, 0.83]
    for idx, state in enumerate(sorted(all_dwell_times)):
        data = np.array(all_dwell_times[state])
        med = np.median(data)
        axs[0].text(med + 0.5, axs[0].get_ylim()[1] * offsets[idx],
                    f"{med:.0f}", color=colors[state],
                    ha='left', va='top', fontsize=9) 

    axs[0].set_xlabel("Empirical dwell time (trials)", fontsize=11)
    axs[0].set_ylabel("Density", fontsize=11)
    axs[0].set_title("Pooled empirical dwell time per state", fontsize=12)
    axs[0].legend(fontsize=9)
    # despine
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)


    # ---------- Expected dwell time histogram ----------
    max_expected = np.max([np.max(expected_dwell[s]) for s in expected_dwell])
    bins_expected = np.arange(0, max_expected + 5, 2)

    for state in sorted(expected_dwell):
        data = np.array(expected_dwell[state])
        axs[1].hist(data, bins=bins_expected, histtype='step',
                    color=colors[state], linewidth=1.5,
                    label=f"State {state + 1}", density=True)
        med_exp = np.median(data)
        axs[1].axvline(med_exp, linestyle=':', color=colors[state], linewidth=1.2)

    axs[1].set_ylim(top=axs[1].get_ylim()[1] * 1.1)

    for idx, state in enumerate(sorted(expected_dwell)):
        data = np.array(expected_dwell[state])
        med_exp = np.median(data)
        axs[1].text(med_exp + 0.5, axs[1].get_ylim()[1] * offsets[idx],
                    f"{med_exp:.1f}", color=colors[state],
                    ha='left', va='top', fontsize=9)

    axs[1].set_xlabel("Expected dwell time (trials)", fontsize=11)
    axs[1].set_ylabel("Density", fontsize=11)
    axs[1].set_title("Expected dwell time across animals", fontsize=12)
    axs[1].legend(fontsize=9)
     # despine   
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)


    plt.tight_layout()

    fig_out = param_dir.parent / 'pngs' / 'dwell_times_hist_all_states__empirical_vs_expected.png'
    fig_out.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_out, bbox_inches='tight', dpi=300)


def plot_emp_dwell_times_congru(run_description='K_2'):
    '''
    3 panels:
    - all empirical dwell times pooled
    - dwell times for congruent runs (choices aligned with block bias)
    - dwell times for incongruent runs
    '''

    df = merge_frames(run_description=run_description)
    all_dwell = compute_dwell_times_from_df(df, congru=None)
    congru_dwell = compute_dwell_times_from_df(df, congru=True)
    incongru_dwell = compute_dwell_times_from_df(df, congru=False)

    dwell_sets = [all_dwell, congru_dwell, incongru_dwell]
    labels = ['All trials', 'Congruent choices', 'Incongruent choices']

    num_states = int(df[[c for c in df.columns if c.startswith("p_state")]].shape[1])
    colors = ['#ff7f00', '#4daf4a', '#377eb8'][:num_states]

    fig, axs = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    offsets = [0.97, 0.90, 0.83]

    for ax, dwell_data, label in zip(axs, dwell_sets, labels):
        max_duration = np.max(dwell_data[:, 0])
        bins = np.arange(1, max_duration + 2) - 0.5
        for state in range(num_states):
            data = dwell_data[dwell_data[:, 1] == state][:, 0]
            ax.hist(data, bins=bins, histtype='step', color=colors[state],
                    linewidth=1.5, density=True, label=f"State {state + 1}")
            med = np.median(data)
            ax.axvline(med, linestyle=':', color=colors[state], linewidth=1.2)
            ax.text(med + 0.5, ax.get_ylim()[1] * offsets[state],
                    f"{med:.0f}", color=colors[state], fontsize=9,
                    ha='left', va='top')

        n_trials = dwell_data[:, 0].sum()
        ax.set_title(f"{label} (n = {n_trials} trials)", fontsize=12)
        ax.set_ylabel("Density", fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9)

    axs[-1].set_xlabel("Empirical dwell time (trials)", fontsize=11)
    axs[-1].set_xlim(left=0.5)
    plt.tight_layout()

    fig_out = Path(pth_eng) / run_description / 'pngs' / 'dwell_times_hist__empirical_all_congru_incongru.png'
    fig_out.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_out, bbox_inches='tight', dpi=300)


def collect_trial_posteriors(unbiased=False, run_description='K_2'):
    """
    Assemble trial-wise posterior state probabilities across animals.
    Output: DataFrame with columns [animal, eid, p_state1, p_state2, p_state3]
    """
    if unbiased:
        base_dir = Path('/home/mic/glm-hmm/results/individual_fit_own')
        data_dir = Path('/home/mic/glm-hmm/data/ibl/data_for_cluster/data_by_animal')
    else:
        base_dir = Path(pth_eng)

    param_dir = base_dir / run_description / 'params'
    out_file = base_dir / run_description / 'trial_posteriors_all_animals.csv'

    if out_file.exists():
        return pd.read_csv(out_file)

    animal_files = sorted(param_dir.glob("*_model_params.npy"))
    all_rows = []

    for i, file in enumerate(animal_files):
        animal = file.stem.replace('_model_params', '')
        print(f"Processing animal {i + 1} of {len(animal_files)}: {animal}")

        # Load model parameters
        model_params = np.load(file, allow_pickle=True).flat[0]
        posterior_probs = model_params['posterior_probs']
        posterior_concat = np.concatenate(posterior_probs)

        # Load session info
        if unbiased:
            dat = np.load(data_dir / f'{animal}_processed.npz', allow_pickle=True)
            session = dat['arr_2']
        else:
            _, _, session = process_bwm_mouse(animal)

        assert len(session) == len(posterior_concat), \
            f"Mismatch in trial count for {animal}: session={len(session)}, posterior={len(posterior_concat)}"

        # Map session paths to eids
        session_paths = list(dict.fromkeys(session))
        eids = one.path2eid(session_paths)
        path_to_eid = dict(zip(session_paths, map(str, eids)))

        # Create dataframe rows
        for sess_path in session_paths:
            eid = path_to_eid[sess_path]
            trial_idx = np.where(np.array(session) == sess_path)[0]
            for p in posterior_concat[trial_idx]:
                row = {
                    'animal': animal,
                    'eid': eid,
                }
                for i, prob in enumerate(p):
                    row[f'p_state{i+1}'] = prob
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    return df


def plot_state_probabilities_last_trials(n_last_trials=400, run_description='K_2'):
    """
    For each session (eid), extract last N trials and average
    posterior probabilities across sessions.
    Plot how p(state1), p(state2), ..., p(stateN) change near session end.

    Parameters:
        n_last_trials : int, number of trials from session end to consider
        run_description : str, model name to load posteriors from
    """
    df = collect_trial_posteriors(run_description=run_description)
    eids = df['eid'].unique()

    # Automatically detect the number of states from column names
    state_cols = sorted([col for col in df.columns if col.startswith('p_state')],
                        key=lambda x: int(x.split('p_state')[1]))

    prob_arrays = {s: [] for s in state_cols}

    for eid in eids:
        df_eid = df[df['eid'] == eid]
        if len(df_eid) < n_last_trials:
            continue
        df_tail = df_eid.iloc[-n_last_trials:]  # take last N trials
        n = len(df_tail)
        for s in state_cols:
            probs = df_tail[s].values
            prob_arrays[s].append(probs)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(-n_last_trials, 0)

    palette = sns.color_palette('Set2', n_colors=len(state_cols))
    for s, color in zip(state_cols, palette):
        dat = np.array(prob_arrays[s])
        mean_prob = np.nanmean(dat, axis=0)
        ax.plot(x, mean_prob, label=s, color=color)

    ax.set_xlabel('Trial number from session end')
    ax.set_ylabel('Average posterior probability')
    ax.set_title(f"State probabilities in final {n_last_trials} trials")
    ax.legend()
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show() 


def merge_frames(run_description='as_paper_scripts', rerun=False):
    '''
    Merge the behavioral frame with the posterior probabilities frame.
    
    Ensures exact row-wise match by animal and eid.
    Assumes trial order in both files is preserved.
    
    Returns:
        df_merged : pd.DataFrame
    '''

    # load if file exists
    base_dir = Path(pth_eng)
    out_file = base_dir / run_description / 'merged_behavioral_and_states.csv'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and not rerun:
        return pd.read_csv(out_file)

    else:    
        print('Merging behavioral and state data...')
        
        state_file = base_dir / run_description / 'trial_posteriors_all_animals.csv'
        behav_file = base_dir / 'bwm_behavioral_data.pqt'

        # Load data
        df_states = pd.read_csv(state_file)
        df_behavior = pd.read_parquet(behav_file)

        # Sort both frames for consistent merge
        df_states = df_states.sort_values(by=['animal', 'eid']).reset_index(drop=True)
        df_behavior = df_behavior.sort_values(by=['animal', 'eid']).reset_index(drop=True)

        # Check that both frames have the same length
        assert len(df_states) == len(df_behavior), \
            f"Length mismatch: {len(df_states)} state rows vs {len(df_behavior)} behavioral rows"

        # Check row-wise equality of animal and eid columns
        assert np.all(df_states['animal'].values == df_behavior['animal'].values), \
            "Mismatch in 'animal' column order"
        assert np.all(df_states['eid'].values == df_behavior['eid'].values), \
            "Mismatch in 'eid' column order"

        # Merge column-wise
        df_merged = pd.concat([df_behavior.reset_index(drop=True), 
                            df_states.drop(columns=['animal', 'eid']).reset_index(drop=True)], axis=1)

        # create signed ccontrast column
        cl = df_merged['contrastLeft'].fillna(0).values
        cr = df_merged['contrastRight'].fillna(0).values    
        signed_contrast = cr - cl
        df_merged['signed_contrast'] = signed_contrast

        # if key probabilityRight exists, delete it
        if 'probabilityRight' in df_merged.columns:
            df_merged.drop(columns=['probabilityRight'], inplace=True)

        # save
        df_merged.to_csv(out_file, index=False)

        return df_merged


def plot_psychometric_curves(run_description='K_3'):
    '''
    Psychometric curves with multiple panels:
    Panel 1: All states combined
    Panel 2+: One panel per state, filtered by dominant state
    '''   

    df = merge_frames(run_description=run_description)
    df = df.dropna(subset=['signed_contrast', 'rewarded'])

    # Identify states
    state_cols = sorted([col for col in df.columns if col.startswith('p_state')],
                        key=lambda x: int(x.split('p_state')[1]))
    n_states = len(state_cols)

    # Compute choice_right
    right_correct = df['contrastLeft'].isna() & (df['rewarded'] == 1)
    right_incorrect = df['contrastRight'].isna() & (df['rewarded'] == -1)
    df['choice_right'] = right_correct | right_incorrect

    # Block info
    df['block_id'] = df.groupby('eid')['probabilityLeft'].transform(
        lambda x: (x != x.shift(1)).cumsum()
    )
    df['block_key'] = df['eid'].astype(str) + '_block' + df['block_id'].astype(str)

    color_map = {0.2: 'red', 0.8: 'blue'}
    fig, axes = plt.subplots(1, n_states + 1, figsize=(5 * (n_states + 1), 5), sharey=True, sharex=True)

    # forinset sharey sharex
    inset_axes = []
    common_xlim = [-1, 1]
    diff_curves = []

    for idx, state_only in enumerate([None] + list(range(n_states))):
        ax = axes[idx]
        if state_only is not None:
            target = f'p_state{state_only + 1}'
            others = [col for col in state_cols if col != target]
            df_plot = df[df[target] > df[others].max(axis=1)]
        else:
            df_plot = df.copy()

        block_vals = []
        for block_key, block_df in df_plot.groupby('block_key'):
            p_left = block_df['probabilityLeft'].iloc[0]
            for contrast, contrast_df in block_df.groupby('signed_contrast'):
                if len(contrast_df) < 5:
                    continue
                frac_right = contrast_df['choice_right'].mean()
                block_vals.append({
                    'signed_contrast': contrast,
                    'frac_right': frac_right,
                    'probabilityLeft': p_left
                })

        df_blocks = pd.DataFrame(block_vals)
        jitter_strength = 0.02
        rng = np.random.default_rng(seed=0)
        avg_curves = {}

        for p_left in [0.2, 0.8]:
            dat = df_blocks[df_blocks['probabilityLeft'] == p_left].copy()
            jitter = rng.uniform(-jitter_strength, jitter_strength, size=len(dat))
            dat['jittered_contrast'] = dat['signed_contrast'] + jitter

            ax.scatter(dat['jittered_contrast'], dat['frac_right'],
                       color=color_map[p_left], alpha=0.2, s=10)

            avg = dat.groupby('signed_contrast')['frac_right'].agg(['mean', 'sem']).reset_index()
            avg_curves[p_left] = avg
            ax.errorbar(avg['signed_contrast'], avg['mean'], yerr=avg['sem'],
                        color=color_map[p_left], capsize=3,
                        label=f'{"Right" if p_left == 0.2 else "Left"} blocks')

        ax.set_xlabel("Signed contrast")
        if idx == 0:
            ax.set_ylabel("Proportion right choices")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        ax.legend()
        state_title = f"State {state_only + 1}" if state_only is not None else "All states"
        ax.set_title(f"{run_description} - {state_title}")
        sns.despine(ax=ax, top=True, right=True)

        # Inset: difference curve
        if all(k in avg_curves for k in [0.2, 0.8]):
            ax_inset = ax.inset_axes([0.6, 0.1, 0.35, 0.35])
            avg_r = avg_curves[0.2].set_index('signed_contrast')
            avg_l = avg_curves[0.8].set_index('signed_contrast')
            diff = avg_r['mean'] - avg_l['mean']
            diff_curves.append(diff)
            inset_axes.append(ax_inset)
            ax_inset.plot(diff.index, diff.values, color='black')
            ax_inset.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_inset.set_title('Right - Left', fontsize=8)
            ax_inset.tick_params(axis='both', which='major', labelsize=6)
            ax_inset.yaxis.tick_right()
            sns.despine(ax=ax_inset, top=True, right=True)
            ax_inset.text(.55, diff.max(), f'{diff.max():.2f}', fontsize=6, va='center')
            ax_inset.patch.set_alpha(0)
            ax_inset.set_facecolor('none')

    common_ylim = [
        min(diff.min() for diff in diff_curves),
        max(diff.max() for diff in diff_curves)
    ]

    for ax_inset in inset_axes:
        ax_inset.set_xlim(common_xlim)
        ax_inset.set_ylim(common_ylim)

    fig.tight_layout()
    plt.show()



####################
### Alberto's results analysis
####################


def load_engagement_data(keep_mask=True):

    '''
    Load Alberto's results
    '''

    # Load the data
    df = pd.read_parquet(Path(pth_eng, 'all_trials_with_engagement.pqt'))
    ntrials_raw = len(df)

    # Filter out trials with no engagement scores
    if keep_mask:
        dropped = (~df['mask']).sum()
        df = df[df['mask']]
        print(f'Dropped {dropped} trials of {ntrials_raw} with mask False')

    # Compute reaction time ('rt') and engagement probabilities
    df['rt'] = df['firstMovement_times'] - df['stimOn_times']
    df['p_eng_2'] = df['glm-hmm_2'].apply(lambda x: x[0])
    df['p_eng_4'] = df['glm-hmm_4'].apply(lambda x: x[0]+x[3])


    # Truncate RT as in BWM paper
    df = df[(df['rt'] >= 0.08) & (df['rt'] <= 2)]

    # # Drop rows with missing values
    # df = df[['rt', 'engagement_K_2', 'engagement_K_4']].dropna()

    return df


def delta_plots(min_400=False):
    '''
    using violin+strip plots for binary engagement variables.

    versus contrast of first trial in pair
    '''

    df = load_engagement_data(keep_mask=True)
    df['contrast'] = df['contrastLeft'].combine_first(df['contrastRight'])
    eids = df['eid'].unique()

    if min_400:
        eids = [eid for eid in eids if len(df[df['eid'] == eid]) >= 400]

    rows = []

    for eid in eids:
        df_eid = df[df['eid'] == eid].copy()

        # Get values and identify gaps between consecutive trials
        level_vals = df_eid['level_0'].values
        diffs = np.diff(level_vals)

        # Create a group ID that increments where gaps > 1 occur
        group_ids = np.zeros(len(level_vals), dtype=int)
        group_ids[1:] = np.cumsum(diffs > 1)

        # Assign group ID to each row
        df_eid['group'] = group_ids       

        for _, group_df in df_eid.groupby('group'):
            if len(group_df) > 1:
                # Compute diffs
                d2 = np.diff(group_df['p_eng_2'].values)
                d4 = np.diff(group_df['p_eng_4'].values)

                # get contrast of first trial in pair
                c = group_df['contrast'].values[:-1]
                #dc = np.diff(group_df['contrast'].values)

                # Append rows for each pair
                for i in range(len(d2)):
                    rows.append({
                        'eid': eid,
                        'd_p_eng_2': np.abs(d2[i]),
                        'd_p_eng_4': np.abs(d4[i]),
                        'contr. 1st trial': np.abs(c[i])
                    })

    # Create final DataFrame
    df_diffs = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    pairs = [
        ('d_p_eng_2', 'd_p_eng_4', 'K=2 vs K=4'),
        ('contr. 1st trial', 'd_p_eng_2', 'K=2 vs Contrast'),
        ('contr. 1st trial', 'd_p_eng_4', 'K=4 vs Contrast'),
    ]

    for ax, (x, y, title) in zip(axes, pairs):
        hb = ax.hexbin(df_diffs[x], df_diffs[y], gridsize=40, cmap='Blues', mincnt=1)
        ax.set_xlabel(f'{"_".join(x.split("_"))}')
        ax.set_ylabel(f'{"_".join(y.split("_"))}')
        #ax.set_title(title)
        fig.colorbar(hb, ax=ax, label='# trial pairs', 
            shrink=0.6, aspect=10)


    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()


def plot_engagement_prob_vs_rt():

    '''
    Plot engagement probability (k2, k4, a panel each)
    versus reaction time in a hex plot
    add pearson correlation coefficient
    '''

    df = load_engagement_data(keep_mask=True)

    fig, axes = plt.subplots(1, 3, figsize=(8, 4))

    hb = axes[0].hexbin(df['rt'], df['p_eng_2'], gridsize=40, 
                        cmap='Blues',mincnt=1)
    axes[0].set_xlabel('Reaction Time (s)')
    axes[0].set_ylabel('Engagement Probability (K=2)')

    axes[1].hexbin(df['rt'], df['p_eng_4'], gridsize=40, 
                   cmap='Blues', mincnt=1)
    axes[1].set_xlabel('Reaction Time (s)')
    axes[1].set_ylabel('Engagement Probability (K=4)')

    # plot p_eng_2 vs p_eng_4
    hb = axes[2].hexbin(df['p_eng_2'], df['p_eng_4'], gridsize=40,
                        cmap='Blues', mincnt=1)
    axes[2].set_xlabel('Engagement Probability (K=2)')
    axes[2].set_ylabel('Engagement Probability (K=4)')

    fig.colorbar(hb, ax=axes, label='# trials', shrink=0.6, aspect=10)


def re_plot_rt_hist(keep_mask=True):
    '''
    replot histogram of reaction times
    '''
    
    df = load_engagement_data(keep_mask=True)

    # Plot histogram
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(df['rt'], bins=150, ax=ax, color='k', alpha=0.5) 
    ax.set_xlabel('Reaction Time (s)')
    ax.set_ylabel('Count')
    ax.set_title('0.08 < rt < 2s')

    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()


def plot_tr_number_vs_scores():
    '''
    For each eid, plot the last 400 trials versus rt and engagement score
    '''

    df = load_engagement_data(keep_mask=True)
    
    # only keep eids which have at least 400 trials
    eids = df['eid'].unique()
    eids = [eid for eid in eids if len(df[df['eid'] == eid]) >= 400]

    # for each eid, plot a line plot, last 400 trials versus rt
    fig, ax = plt.subplots(figsize=(6, 4))
    l = []
    le2 = []
    le4 = []

    for eid in eids:
        df_eid = df[df['eid'] == eid]
        ntrials = len(df_eid)
        df_eid = df_eid.iloc[-400:]
        l.append(df_eid['rt'].values)
        le2.append(df_eid['engagement_K_2'].values)
        le4.append(df_eid['engagement_K_4'].values)

    ax.plot(np.arange(-400, 0), np.mean(l,axis=0), alpha=0.5, label='rt')

    # on a twin axis, plot engagement scores
    ax2 = ax.twinx()
    ax2.set_ylabel('Engagement Score')
    ax2.plot(np.arange(-400, 0), np.mean(le2,axis=0), 
        alpha=0.5, color='orange', label='engagement_K_2')
    ax2.plot(np.arange(-400, 0), np.mean(le4,axis=0), 
        alpha=0.5, color='green', label='engagement_K_4')

    # merge legends from ax and ax2
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines += lines2
    labels += labels2
    # remove duplicates
    by_label = dict(zip(labels, lines))

    ax.legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
        ncol=3,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02)  # X=centered, Y=just above the axes
    )

    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Reaction Time (s)')
    #ax.set_title('Last 400 trials vs Reaction Time')
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()


 
