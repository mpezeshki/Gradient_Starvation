import numpy as np
import math, csv, argparse, pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from matplotlib import pyplot as plt

#################################
### RANDOM FEATURE GENERATION ###
#################################

def sample_from_sphere(n, d):
    x = np.random.multivariate_normal(np.zeros(d), np.identity(d), n)
    x = (x.T/np.linalg.norm(x, axis=1)).T
    return x

def get_projections(x, W):
    # Skipped normalization because I normalized W instead
    return np.maximum(x @ W, 0)

def get_random_features(train_x, test_x, N):
    if isinstance(train_x, tuple):
        assert len(N)==len(train_x)
    elif N==0:
        return train_x, test_x
    else:
        train_x = (train_x,)
        test_x = (test_x,)
        N = (N,)

    train_x_projected = []
    test_x_projected = []
    for train_x_component, test_x_component, N_component in zip(train_x, test_x, N):
        if N_component == 0:
            continue
        d = train_x_component.shape[1] # number of original features
        W = sample_from_sphere(d, N_component)
        # Train
        train_x_projected.append(get_projections(train_x_component, W))
        # Test
        test_x_projected.append(get_projections(test_x_component, W))
    return np.hstack(train_x_projected), np.hstack(test_x_projected)

#########################
### SAMPLING EXAMPLES ###
#########################

def oversample(g, n_groups):
    group_counts = []
    for group_idx in range(n_groups):
        group_counts.append((g==group_idx).sum())
    resampled_idx = []
    for group_idx in range(n_groups):
        idx, = np.where(g==group_idx)
        if group_counts[group_idx] < max(group_counts):
            for _ in range(max(group_counts)//group_counts[group_idx]):
                resampled_idx.append(idx)
            resampled_idx.append(np.random.choice(idx,
                                                  max(group_counts) % group_counts[group_idx],
                                                  replace=False))
        else:
            resampled_idx.append(idx)
    resampled_idx = np.concatenate(resampled_idx)
    return resampled_idx

def undersample(g, n_groups):
    group_counts = []
    for group_idx in range(n_groups):
        group_counts.append((g==group_idx).sum())
    resampled_idx = []
    for group_idx in range(n_groups):
        idx, = np.where(g==group_idx)
        resampled_idx.append(np.random.choice(idx,
                                              min(group_counts),
                                              replace=False))
    resampled_idx = np.concatenate(resampled_idx)
    return resampled_idx

##############
### MODELS ###
##############

def fit_logistic_regression(X, y, Lambda=None):
    penalty_args = {}
    if Lambda:
        penalty_args['penalty'] = 'l2'
        n = y.size
        penalty_args['C'] = 1/(n*Lambda)
    else:
        penalty_args['penalty'] = 'none'
    model = LogisticRegression(**penalty_args, fit_intercept=False, solver='lbfgs', max_iter=1e8)
    model.fit(X,y)
    return model

def fit_ridge_regression(X, y, Lambda=1e-6):
    n = y.size
    model = Ridge(alpha=Lambda*n, fit_intercept=False)
    model.fit(X,y)
    return model

#########################
### ERROR COMPUTATION ###
#########################

def zero_one_error(model, X, y):
    if isinstance(model, LogisticRegression):
        return 1 - model.score(X, y)
    elif isinstance(model, Ridge):
        yhat_zero_one = model.predict(X)>0
        yhat = -1*(1-yhat_zero_one) + (yhat_zero_one)
        return np.mean(yhat!=y)

def squared_error(model, X, y):
    assert isinstance(model, Ridge), 'squared error supported only for Ridge models'
    return np.mean((model.predict(X)-y)**2)

def compute_error(full_data, model, n_groups, error_fn, resample_idx = None, verbose=True):
    error_log = {}

    (train_x, train_y, train_g), (test_x, test_y, test_g) = full_data
    # get group counts based on full data
    group_count = []
    for g in range(n_groups):
        g_mask = (train_g==g)
        group_count.append(g_mask.sum())

    # get train accuracies based on resampled data
    if resample_idx is not None:
        train_x, train_y, train_g = train_x[resample_idx,:], train_y[resample_idx], train_g[resample_idx]
    group_train_error = []
    for g in range(n_groups):
        g_mask = (train_g==g)
        error = error_fn(model, train_x[g_mask,:], train_y[g_mask])
        error_log[f'train_error_group:{g}'] = error
        group_train_error.append(error)
    error_log['robust_train_error'] = max(group_train_error)
    error_log['avg_train_error'] = np.array(group_train_error) @ (np.array(group_count)/sum(group_count))

    group_test_error = []
    for g in range(n_groups):
        g_mask = (test_g==g)
        error = error_fn(model, test_x[g_mask,:], test_y[g_mask])
        group_test_error.append(error)
        error_log[f'test_error_group:{g}'] = error
    error_log['robust_test_error'] = max(group_test_error)
    error_log['avg_test_error'] = np.array(group_test_error) @ (np.array(group_count)/sum(group_count))

    if verbose:
        print(f'Average train error: {error_log["avg_train_error"]}')
        print(f'Average test error: {error_log["avg_test_error"]}')
        print(f'Robust train error: {error_log["robust_train_error"]}')
        print(f'Robust test error: {error_log["robust_test_error"]}')

    return error_log

##################
### EXPERIMENT ###
##################

def run_no_projection_model(data_generation_fn, data_args, N, fit_model_fn, error_fn, model_kwargs={}, seed=None, verbose=True,
                            model_file=None):
    # set seed
    if seed is not None:
        np.random.seed(seed)

    # print settings
    if verbose:
        print(f'Model fit function: {fit_model_fn.__name__}')
        if len(model_kwargs)>0:
            print('Model kwargs:')
            for k,v in model_kwargs.items():
                print(f'\t{k}: {v}')
        print(f'Number of random features: {N}')
        print(f'Seed: {seed}')

    if data_generation_fn.__name__=='generate_toy_data_no_projections':
        data_args = data_args.copy()
        data_args['d_noise'] = N
    # data
    train_data, n_groups = data_generation_fn(**data_args, train=True) 
    test_data, n_groups = data_generation_fn(**data_args, train=False)
    data = (train_data, test_data)
    (train_x, train_y, train_g), (test_x, test_y, test_g) = data

    erm_error_log, res_error_log = {}, {}
    # ERM
    if verbose: print('\nERM')
    erm_model = fit_model_fn(train_x, train_y, **model_kwargs)
    erm_error = compute_error(data, erm_model, n_groups, error_fn, verbose=verbose)
    # OVER
    if verbose: print('\nOversampling')
    resample_idx = oversample(train_g, n_groups)
    over_model = fit_model_fn(train_x[resample_idx,:], train_y[resample_idx], **model_kwargs)
    over_error = compute_error(data, over_model, n_groups, error_fn, resample_idx=resample_idx, verbose=verbose)
    # UNDER
    if verbose: print('\nUndersampling')
    resample_idx = undersample(train_g, n_groups)
    under_model = fit_model_fn(train_x[resample_idx,:], train_y[resample_idx], **model_kwargs)
    under_error = compute_error(data, under_model, n_groups, error_fn, resample_idx=resample_idx, verbose=verbose)

    if model_file:
        model_dict = {'erm_model': erm_model, 'over_model': over_model, 'under_model': under_model}
        pickle.dump(model_dict, open(model_file, "wb" ))

    return erm_error, over_error, under_error


def run_random_features_model(full_data, n_groups, N, fit_model_fn, error_fn, model_kwargs={}, seed=None, verbose=True):
    # set seed
    if seed is not None:
        np.random.seed(seed)

    # print settings
    if verbose:
        print(f'Model fit function: {fit_model_fn.__name__}')
        if len(model_kwargs)>0:
            print('Model kwargs:')
            for k,v in model_kwargs.items():
                print(f'\t{k}: {v}')
        print(f'Number of random features: {N}')
        print(f'Seed: {seed}')

    # data
    (train_x, train_y, train_g), (test_x, test_y, test_g) = full_data
    proj_train_x, proj_test_x = get_random_features(train_x, test_x, N)
    projected_data = (proj_train_x, train_y, train_g), (proj_test_x, test_y, test_g)

    erm_error_log, res_error_log = {}, {}
    # ERM
    if verbose: print('\nERM')
    model = fit_model_fn(proj_train_x, train_y, **model_kwargs)
    erm_error = compute_error(projected_data, model, n_groups, error_fn, verbose=verbose)
    # OVER
    if verbose: print('\nOversampling')
    resample_idx = oversample(train_g, n_groups)
    model = fit_model_fn(proj_train_x[resample_idx,:], train_y[resample_idx], **model_kwargs)
    over_error = compute_error(projected_data, model, n_groups, error_fn, resample_idx=resample_idx, verbose=verbose)
    # UNDER
    if verbose: print('\nUndersampling')
    resample_idx = undersample(train_g, n_groups)
    model = fit_model_fn(proj_train_x[resample_idx,:], train_y[resample_idx], **model_kwargs)
    under_error = compute_error(projected_data, model, n_groups, error_fn, resample_idx=resample_idx, verbose=verbose)

    return erm_error, over_error, under_error

def save_error_logs(outfile, error_log_dict_list, opt_type_list):
    writer = None
    with open(outfile, 'w') as f:
        for opt_type, error in zip(opt_type_list,
                                   error_log_dict_list):
            error['opt_type'] = opt_type
            if writer is None:
                writer = csv.DictWriter(f, fieldnames = error.keys())
                writer.writeheader()
            writer.writerow(error)

########################
### ANALYSIS HELPERS ###
#######################

def read_error_logs(path_dict,
                    opt_types=['ERM', 'oversample', 'undersample'],
                    fields=['avg_train_error', 'avg_test_error', 'robust_train_error', 'robust_test_error',
                            'train_error_group:0','train_error_group:1','train_error_group:2','train_error_group:3',
                            'test_error_group:0','test_error_group:1','test_error_group:2','test_error_group:3']):
    errors = {}
    for field in fields:
        errors[field] = {}
        for opt_type in opt_types:
            errors[field][opt_type] = []

    x_axis_values = sorted(path_dict.keys())
    for x in x_axis_values:
        for opt_type in opt_types:
            for field in fields:
                errors[field][opt_type].append([])
        for path in path_dict[x]:
            error_df = pd.read_csv(path)
            for opt_type in opt_types:
                error_row = error_df[error_df['opt_type']==opt_type]
                for field in fields:
                    errors[field][opt_type][-1].append(error_row[field].values[0])

    for field in fields:
        for opt_type in opt_types:
            errors[field][opt_type] = np.array(errors[field][opt_type])
    return np.array(x_axis_values), errors

def print_errors(key_list, error_log, opt_types=['ERM', 'oversample', 'undersample']):

    # Average errors
    train_accs = error_log['avg_train_error']
    test_accs = error_log['avg_test_error']

    print("---------Average train accuracy--------")
    data = {'type': key_list, 'ERM': np.ravel(train_accs['ERM']),
            'oversample': np.ravel(train_accs['oversample']),
            'undersample': np.ravel(train_accs['undersample'])}
    print(pd.DataFrame(data))

    print("---------Average test accuracy--------")
    data = {'type': key_list, 'ERM': np.ravel(test_accs['ERM']),
            'oversample': np.ravel(test_accs['oversample']),
            'undersample': np.ravel(test_accs['undersample'])}
    print(pd.DataFrame(data))

    # Robust errors
    train_accs = error_log['robust_train_error']
    test_accs = error_log['robust_test_error']

    print("---------Robust train accuracy--------")
    data = {'type': key_list, 'ERM': np.ravel(train_accs['ERM']),
            'oversample': np.ravel(train_accs['oversample']),
            'undersample': np.ravel(train_accs['undersample'])}
    print(pd.DataFrame(data))

    print("---------Robust test accuracy--------")
    data = {'type': key_list, 'ERM': np.ravel(test_accs['ERM']),
            'oversample': np.ravel(test_accs['oversample']),
            'undersample': np.ravel(test_accs['undersample'])}
    print(pd.DataFrame(data))

def plot_double_descent(x_axis_values, error_log, xlabel,
                        opt_types=['ERM', 'oversample', 'undersample'],
                        robust=True, print_values=True, figure=None, 
                        train_label='Train', test_label='Test',
                        train_color='grey', test_color='black'):
    if robust:
        train_accs = error_log['robust_train_error']
        test_accs = error_log['robust_test_error']
    else:
        train_accs = error_log['avg_train_error']
        test_accs = error_log['avg_test_error']

    if figure is not None:
        fig, ax = figure
    else:
        fig, ax = plt.subplots(2, len(opt_types), figsize=(8,4), sharex=True, sharey='row')
    for i, opt_type in enumerate(opt_types):
        ax[0,i].semilogx(x_axis_values, np.mean(train_accs[opt_type], axis=1),
                         color=train_color, label=train_label)
        ax[1,i].semilogx(x_axis_values, np.mean(test_accs[opt_type], axis=1),
                         color=test_color, label=test_label)
        train_label = None
        test_label = None
        for replicate_idx in range(train_accs[opt_type].shape[1]):
            ax[0,i].scatter(x_axis_values, train_accs[opt_type][:,replicate_idx],
                            color=train_color, marker='.', alpha=0.5)
            ax[1,i].scatter(x_axis_values, test_accs[opt_type][:,replicate_idx],
                            color=test_color, marker='.', alpha=0.5)
        ax[0,i].set_title(opt_type)
        ax[1,i].set_xlabel(xlabel)
        for row in range(2):
            if robust:
                ylabel='Robust Error'
            else:
                ylabel='Average Error'
            ax[row,i].set_ylabel(ylabel)
        if print_values:
            print(opt_type)
            print(np.mean(train_accs[opt_type], 1))
            print(np.mean(test_accs[opt_type], 1))
    plt.tight_layout()
    return fig, ax

def plot_results(get_filepath_dict, args_list, N_list, robust, verbose=False, legend_args={}):
    cmap=plt.get_cmap('tab20')
    fig, ax = plt.subplots(2, 3, figsize=(12,4), sharex=True, sharey='row')
    for i, args in enumerate(args_list):
        label = ', '.join([f'{k}={v}' for k,v in args.items()])
        paths = get_filepath_dict(**args, N_list=N_list)
        N_list, error_logs = read_error_logs(paths)
        plot_double_descent(N_list, error_logs, 'N', robust=robust,
                            figure=(fig, ax), train_label=None, test_label=label,
                            train_color=cmap((i*2+1)%20), test_color=cmap((i*2)%20), 
                            print_values=verbose)
    fig.legend(**legend_args)
