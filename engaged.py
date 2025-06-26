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


one = ONE()
pth_eng = Path(one.cache_dir, 'engaged')
pth_eng.mkdir(parents=True, exist_ok=True)


###########
### utils
###########

def compute_dwell_times_with_states(state_probs_list):
    """
    Compute dwell times and their corresponding states from posterior state probabilities.

    Parameters:
        state_probs_list : list of np.ndarray
            Each array is (n_trials, n_states), rows sum to 1.

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

def model_single_mouse(animal, run_description='as_paper_scripts', unbiased=False):
    
    '''
    Train model on single mouse using parameters from notebook as
    starting weights
    animal = 'CSHL_001'
    animals = np.load('/home/mic/glm-hmm/data/ibl/data_for_cluster'
                      '/data_by_animal/animal_list.npz')['arr_0']
    run_description: 'standard_init' for standard initialization


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
    num_states = 3              # K
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
    glmhmm.params = [[np.array([-0.52493862, -1.64298306, -1.53708898])],
    [np.array([[-0.02608457, -4.25327563, -4.46282725],
            [-3.04799552, -0.05351097, -5.370778  ],
            [-3.09435783, -5.83956581, -0.04941527]])],
    np.array([[[-7.20614670e+00, -3.12209531e-01, -1.61175163e-03,
            1.43813500e-01]],
    
            [[-1.22616681e+00, -3.61431579e-01, -1.80390841e-01,
            1.70656106e+00]],
    
            [[-1.20628874e+00, -3.20944513e-01, -1.98757460e-01,
            -1.62621352e+00]]])]


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


def plot_model_params(animal, run_description='as_paper_scripts', unbiased=False):

    # Load model parameters
    if unbiased:
        ttt = '/home/mic/glm-hmm/results/individual_fit_own'
    else:
        ttt = pth_eng

    animal_file = Path(ttt, run_description, 'params',
                       f'{animal}_model_params.npy')
    model_params = np.load(animal_file, allow_pickle=True).flat[0]

    gen_weights = model_params['gen_weights']
    trans_mat = model_params['trans_mat']
    posterior_probs = model_params['posterior_probs']

    posterior_probs_concat = np.concatenate(posterior_probs)
    state_max_posterior = np.argmax(posterior_probs_concat, axis=1)
    _, state_occupancies = np.unique(state_max_posterior, return_counts=True)

    dwell_info = compute_dwell_times_with_states(posterior_probs)

    num_states = gen_weights.shape[0]
    input_dim = gen_weights.shape[2]

    inpts_names = ['stim', 'bias', 'prev_choice', 'WSLS']
    cols = ['#ff7f00', '#4daf4a', '#377eb8'][:num_states]

    fig = plt.figure(figsize=(14, 3), dpi=80, facecolor='w', edgecolor='k')  

    # Panel 1: generative weights
    plt.subplot(1, 5, 1)
    for k in range(num_states):
        plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                 color=cols[k], linestyle='-', lw=1.5, label=f"state {k+1}")
    plt.yticks(fontsize=10)
    plt.ylabel("GLM weight", fontsize=12)
    plt.xlabel("covariate", fontsize=12)
    plt.xticks(range(input_dim), inpts_names, fontsize=10, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.title("Generative weights", fontsize=13)

    # Panel 2: transition matrix
    plt.subplot(1, 5, 2)
    plt.imshow(trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(num_states):
        for j in range(num_states):
            plt.text(j, i, str(np.around(trans_mat[i, j], 2)), ha="center", va="center",
                     color="k", fontsize=10)
    plt.xticks(range(num_states), [str(i+1) for i in range(num_states)], fontsize=9)
    plt.yticks(range(num_states), [str(i+1) for i in range(num_states)], fontsize=9)
    plt.xlim(-0.5, num_states - 0.5)
    plt.ylim(num_states - 0.5, -0.5)
    plt.ylabel("state t", fontsize=12)
    plt.xlabel("state t+1", fontsize=12)
    plt.title("Transition matrix", fontsize=13)

    # Panel 3: state occupancy
    plt.subplot(1, 5, 3)
    for z, occ in enumerate(state_occupancies):
        plt.bar(z, occ, width=0.8, color=cols[z])
    plt.xticks(range(num_states), [str(i+1) for i in range(num_states)], fontsize=10)
    plt.xlabel('state', fontsize=12)
    plt.ylabel('occupancy', fontsize=12)
    plt.title("State occupancy", fontsize=13)

    # Panel 4: empirical dwell time histogram
    plt.subplot(1, 5, 4)
    all_dwell_times = dwell_info[:, 0]
    bins = np.histogram_bin_edges(all_dwell_times, bins=30)
    for k in range(num_states):
        dwell_k = dwell_info[dwell_info[:, 1] == k][:, 0]
        plt.hist(dwell_k, bins=bins, histtype='step', 
                 label=f'state {k+1}', color=cols[k], linewidth=1.5)
        plt.axvline(np.median(dwell_k), color=cols[k], linestyle='--', linewidth=1.2)
    plt.xlabel("Dwell time (trials)", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.title("Empirical dwell", fontsize=13)

    # Panel 5: expected dwell time from trans_mat
    plt.subplot(1, 5, 5)
    P = np.exp(trans_mat) if np.all(trans_mat <= 0) else trans_mat
    dwell_exp = 1 / (1 - np.diag(P))
    for k in range(num_states):
        plt.bar(k, dwell_exp[k], color=cols[k], width=0.8)
        plt.text(k, dwell_exp[k] + 1, f"{dwell_exp[k]:.1f}", ha='center', fontsize=9)
    plt.xticks(range(num_states), [f'{i+1}' for i in range(num_states)], fontsize=10)
    plt.ylabel("Trials", fontsize=10)
    plt.title("Expected dwell", fontsize=13)

    plt.tight_layout()

    # Save
    if unbiased:
        sss = Path('/home/mic/glm-hmm/results/individual_fit_own',
                   run_description, 'pngs')
    else:
        sss = Path(pth_eng, run_description, 'pngs')
    sss.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(sss, f'{animal}_model_params.png'), bbox_inches='tight')


def do_for_all(run_description='as_paper_scripts', unbiased=False):
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


#######################
### meta analyses
#######################


def plot_dwell_times_hist_all(unbiased=True):
    '''
    For all animals, load params file, compute dwell times
    and plot a histogram per state. Also show expected dwell times
    computed from transition matrix diagonals.
    '''

    if unbiased:
        # for unbiased trials
        param_dir = Path('/home/mic/glm-hmm/results/individual_fit_own/'
                         'as_paper_scripts/params')
    else:
        # for biased trials
        param_dir = Path(pth_eng, 'as_paper_scripts', 'params')

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
        axs[0].axvline(np.median(data), linestyle=':', color=colors[state], linewidth=1.2)

    axs[0].set_xlabel("Empirical dwell time (trials)", fontsize=11)
    axs[0].set_ylabel("Density", fontsize=11)
    axs[0].set_title("Pooled empirical dwell time per state", fontsize=12)
    axs[0].legend(fontsize=9)

    # ---------- Expected dwell time histogram ----------
    max_expected = np.max([np.max(expected_dwell[s]) for s in expected_dwell])
    bins_expected = np.arange(0, max_expected + 5, 2)

    for state in sorted(expected_dwell):
        data = np.array(expected_dwell[state])
        axs[1].hist(data, bins=bins_expected, histtype='step',
                    color=colors[state], linewidth=1.5,
                    label=f"State {state + 1}", density=True)
        axs[1].axvline(np.median(data), linestyle=':', color=colors[state], linewidth=1.2)

    axs[1].set_xlabel("Expected dwell time (trials)", fontsize=11)
    axs[1].set_ylabel("Density", fontsize=11)
    axs[1].set_title("Expected dwell time across animals", fontsize=12)
    axs[1].legend(fontsize=9)

    plt.tight_layout()

    fig_out = param_dir.parent / 'pngs' / 'dwell_times_hist_all_states__empirical_vs_expected.png'
    fig_out.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_out, bbox_inches='tight', dpi=300)



def collect_trial_posteriors(unbiased=False, run_description='as_paper_scripts'):
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
                all_rows.append({
                    'animal': animal,
                    'eid': eid,
                    'p_state1': p[0],
                    'p_state2': p[1],
                    'p_state3': p[2],
                })

    df = pd.DataFrame(all_rows)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    return df



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