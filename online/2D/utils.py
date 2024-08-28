import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os
import csv
import matplotlib.cm as cm
import imageio
from io import BytesIO
from matplotlib.patches import Ellipse
from model import predict_batch_placecell, softmax

def get_statespace(num=51):
    x = np.linspace(-1,1,num)
    xx,yy = np.meshgrid(x,x)
    xs = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    return xs

def plot_place_cells(params, startcoord, goalcoord=None, goalsize=0.1, title='2D', envsize=1, obstacles=False):
    pc_centers, pc_sigmas, pc_constant, actor_weights, critic_weights = params
    num = len(pc_centers)

    plt.figure(figsize=(4, 3))  # Adjust figure size
    plt.title(title)
    
    if goalcoord is not None:
        circle = plt.Circle(goalcoord, goalsize, color='r', fill=True, zorder=2)
        plt.gca().add_patch(circle)
    
    if obstacles:
        from matplotlib.patches import Rectangle
        plt.gca().add_patch(Rectangle((-0.6, 0.3), 0.2, 0.7, facecolor='grey'))  # top left
        plt.gca().add_patch(Rectangle((0.4, 0.3), 0.2, 0.7, facecolor='grey'))  # top right
        plt.gca().add_patch(Rectangle((-0.6, -0.3), 0.2, -0.7, facecolor='grey'))  # bottom left
        plt.gca().add_patch(Rectangle((0.4, -0.3), 0.2, -0.7, facecolor='grey'))  # bottom right

    # Check if sigma has a shape for 2D ellipses
    for i in range(num):
        if pc_sigmas[i].shape[0] > 1:
            # Ellipses for 2D Gaussians
            width = 2 * np.sqrt(pc_sigmas[i][0, 0])
            height = 2 * np.sqrt(pc_sigmas[i][1, 1])
            ellipse = Ellipse(xy=pc_centers[i], width=width, height=height, angle=0, edgecolor='g', fc='None', zorder=1)
            plt.gca().add_patch(ellipse)
            plt.scatter(pc_centers[i][0], pc_centers[i][1], s=5, color='purple', zorder=3)
        else:
            # Circles for isotropic Gaussians
            circle = plt.Circle(pc_centers[i], np.sqrt(2 * pc_sigmas[i][0, 0]), color='g', fill=False, zorder=1)
            plt.gca().add_patch(circle)
            plt.scatter(pc_centers[i][0], pc_centers[i][1], s=5, color='purple', zorder=3)

    plt.scatter(startcoord[0], startcoord[1], s=5, color='blue', label='Start', zorder=3)  # Mark start coordinate
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(np.linspace(-1, 1, 3))
    plt.yticks(np.linspace(-1, 1, 3))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # plt.legend()
    plt.tight_layout()
    plt.show()

def plot_maps(actor_weights,critic_weights, env, npc, title=None):
    npcs = int(npc**0.5)
    plt.figure(figsize=(3,2))
    plt.imshow(critic_weights.reshape([npcs, npcs]), origin='lower')
    plt.colorbar()
    dirction = np.matmul(actor_weights, env.onehot2dirmat)
    xx, yy = np.meshgrid(np.arange(npcs), np.arange(npcs))
    plt.quiver(xx.reshape(-1),yy.reshape(-1), dirction[:,0], dirction[:,1], color='k', scale_units='xy')
    plt.gca().set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Value & Policy maps')
    plt.tight_layout()


def store_csv(csv_file, args, score, drift):
    # Extract all arguments from args namespace
    arg_dict = vars(args)
    
    # Add score and drift to the dictionary
    arg_dict['score'] = score
    arg_dict['drift'] = drift

    # Create csv_columns from the keys of arg_dict
    csv_columns = list(arg_dict.keys())

    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a' if file_exists else 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(arg_dict)


def evaluate_loss(latencys, threshold=35, stability_window=10000, w1=1,w2=1, w3=1):
    loss_vector = np.array(moving_average(latencys,20))
    # Calculate convergence speed
    try:
        convergence_epoch = next(i for i, v in enumerate(loss_vector) if v < threshold)
    except StopIteration:
        convergence_epoch = len(loss_vector)
    
    # Calculate stability
    stability = np.std(loss_vector[-stability_window:]) if len(loss_vector) >= stability_window else np.std(loss_vector)
    
    # Final loss value
    final_loss = loss_vector[-1]

    score = convergence_epoch*final_loss*stability
    
    return score

# Define the functions to fit
def linear(x, a, b):
    return a * x + b

def exponential(x, a, b, c):
    return a * np.exp(-b * x) + c

def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

def power_law(x, a, b, c):
    return a * np.power(x, -b) + c

# Function to fit the model
def fit_model(x, y, func_type='linear', initial_guess=None):
    if func_type == 'linear':
        func = linear
        if initial_guess is None: initial_guess = [1, 1]
    elif func_type == 'exp':
        func = exponential
        if initial_guess is None: initial_guess = [100, 0, 10]
    elif func_type == 'sigmoid':
        func = sigmoid
        if initial_guess is None: initial_guess = [1, 1, 1]
    elif func_type == 'power':
        func = power_law
        if initial_guess is None: initial_guess = [1, 1, 100]
    else:
        raise ValueError("Unsupported function type. Choose from 'linear', 'exp', 'sigmoid', or 'power'.")

    popt, _ = curve_fit(func, x, y, p0=initial_guess, maxfev=10000)
    return popt


def plot_model_fit(x, y, func_type):
    plt.scatter(x, y)
    popt, func = fit_model(x, y, func_type)
    plt.plot(x, func(x, *popt), label=f'Fitted: {func_type}\nParams: {np.round(popt, 3)}', color='red')
    plt.legend(frameon=False, fontsize=6)
    plt.show()

def get_1D_fva_density_corr(allcoords, logparams, end, gap=25, bins=15, delta_t=1, end2=None):
    bins = np.linspace(-1, 1, bins)
    fx = []
    dx = []
    xs = []
    vx = []
    ax = []
    for g in range(gap-1):
        coord = allcoords[end-1-g].flatten()
        velocity = (coord[1:] - coord[:-1]) / delta_t
        coord = coord[:-1]  # Exclude the last coordinate because it doesn't have a corresponding velocity
        acceleration = (velocity[1:] - velocity[:-1]) / delta_t
        velocity = velocity[:-1]  # Exclude the last velocity because it doesn't have a corresponding acceleration
        coord = coord[:-1]  # Exclude the last coordinate because it doesn't have a corresponding acceleration

        frequency, x = np.histogram(coord, bins=bins)
        visits = x[:-1] + (x[1] - x[0]) / 2

        if end2 is None:
            end2 = end

        param = logparams[end2-1-g]
        pcacts = predict_batch_placecell(param, visits)
        density = np.sum(pcacts, axis=1)
        
        # Bin the velocities and accelerations
        bin_indices = np.digitize(coord, bins) - 1
        sum_velocity = np.zeros(len(bins) - 1)
        sum_acceleration = np.zeros(len(bins) - 1)
        counts = np.zeros(len(bins) - 1)

        for i in range(len(velocity)):
            if 0 <= bin_indices[i] < len(sum_velocity):
                sum_velocity[bin_indices[i]] += velocity[i]
                sum_acceleration[bin_indices[i]] += acceleration[i]
                counts[bin_indices[i]] += 1

        avg_velocity = np.zeros(len(sum_velocity))
        avg_acceleration = np.zeros(len(sum_acceleration))
        nonzero_bins = counts != 0
        avg_velocity[nonzero_bins] = sum_velocity[nonzero_bins] / counts[nonzero_bins]
        avg_acceleration[nonzero_bins] = sum_acceleration[nonzero_bins] / counts[nonzero_bins]
        
        fx.append(frequency)
        dx.append(density)
        xs.append(visits)
        vx.append(avg_velocity)
        ax.append(avg_acceleration)
    
    fx = np.array(fx)
    dx = np.array(dx)
    xs = np.array(xs)
    vx = np.array(vx)
    ax = np.array(ax)

    xs = np.mean(xs, axis=0)
    dx = np.mean(dx, axis=0)
    fx = np.mean(fx, axis=0)
    vx = np.mean(vx, axis=0)
    ax = np.mean(ax, axis=0)

    R_fd, pval_fd = stats.pearsonr(fx, dx)
    R_vd, pval_vd = stats.pearsonr(vx, dx)
    R_ad, pval_ad = stats.pearsonr(ax, dx)
    return xs, fx, dx, vx, ax, R_fd, pval_fd, R_vd, pval_vd, R_ad, pval_ad


def get_1D_freq_density_corr(allcoords, logparams, trial, gap=25, bins=15):
    bins = np.linspace(-0.8,0.8,bins)
    fx = []
    dx = []
    xs = []

    for g in range(gap):
        coord = allcoords[trial-g-1]
        #x = np.linspace(-1.0,1.0,bins+1)
        #coord = np.concatenate([coord[:,0], x],axis=0)
        frequency,x = np.histogram(coord, bins=bins)
        visits = x[:-1] + (x[1] - x[0])/2

        param = logparams[trial-g-1]
        pcacts = predict_batch_placecell(param, visits)
        density = np.sum(pcacts,axis=1)
    
        fx.append(frequency)
        dx.append(density)
        xs.append(visits)
    
    fx = np.array(fx)
    dx = np.array(dx)
    xs = np.array(xs)

    xs = np.mean(xs,axis=0)
    dx = np.mean(dx,axis=0)
    fx = np.mean(fx,axis=0)
    R,pval = stats.pearsonr(fx, dx)
    return xs, fx, dx, R, pval

def plot_fxdx_trials(allcoords, logparams, trials,gap, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    
    Rs = []
    for trial in trials:
        visits, frequency, density, R, pval = get_1D_freq_density_corr(allcoords, logparams, trial, gap=gap)
        Rs.append(R)
    ax.plot(trials, Rs, marker='o')
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(trials).reshape(-1), np.array(Rs).reshape(-1))
    regression_line = slope * np.array(trials).reshape(-1) + intercept
    ax.plot(np.array(trials).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    ax.legend(frameon=False, fontsize=6)
    ax.set_title('Correlation with learning')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Correlation')


def plot_fx_dx(allcoords, logparams, trial, title,gap,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    
    visits, frequency, density, R, pval = get_1D_freq_density_corr(allcoords, logparams, trial, gap=gap)
    ax.scatter(frequency, density)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(frequency).reshape(-1), np.array(density).reshape(-1))
    regression_line = slope * np.array(frequency).reshape(-1) + intercept
    ax.plot(np.array(frequency).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    ax.legend(frameon=False, fontsize=6)
    ax.set_title(title)
    ax.set_xlabel('Frequency $f(x)$')
    ax.set_ylabel('Density $d(x)$')

def plot_freq_density_corr(visits, frequency, density, title):
    plt.figure(figsize=(6, 2))
    plt.suptitle(title)
    plt.subplot(121)
    plt.plot(visits, (frequency - np.min(frequency)) / (np.max(frequency) - np.min(frequency)), label='Frequency')
    plt.plot(visits, (density - np.min(density)) / (np.max(density) - np.min(density)), label='Density')
    plt.xlabel('Location (x)')
    plt.ylabel('Norm value')
    plt.axvline(-0.75, color='g', linestyle='--', label='Start')
    plt.axvline(0.5, color='r', linestyle='--', label='Goal')
    plt.legend(frameon=False, fontsize=6)

    plt.subplot(122)
    plt.scatter(frequency, density)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(frequency).reshape(-1), np.array(density).reshape(-1))
    regression_line = slope * np.array(frequency).reshape(-1) + intercept
    plt.plot(np.array(frequency).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    plt.legend(frameon=False, fontsize=6)
    plt.title('Correlation')
    plt.xlabel('Frequency f(x)')
    plt.ylabel('Density d(x)')
    plt.tight_layout()


def flatten(xss):
    return np.array([x for xs in xss for x in xs],dtype=np.float32)

def random_zach_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):
    pc_cent =  np.linspace(-envsize,envsize,npc) 
    pc_sigma = np.random.gamma(1, sigma/1,size=npc)
    pc_constant = np.random.uniform(0, alpha,size=npc)
    #pc_constant /= np.max(pc_constant)*alpha
    
    return [np.array(pc_cent), np.array(pc_sigma), np.array(pc_constant), 
    1e-5 * np.random.normal(size=(npc,nact)), 1e-5 * np.random.normal(size=(npc,1))]

def random_gamma_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):
    pc_cent =  np.linspace(-envsize,envsize,npc) 
    pc_sigma = np.ones(npc)*sigma
    np.random.seed(seed)
    pc_constant = np.random.gamma(1, 0.1/1,size=npc)
    #pc_constant /= np.max(pc_constant)*alpha
    
    return [np.array(pc_cent), np.array(pc_sigma), np.array(pc_constant), 
    1e-5 * np.random.normal(size=(npc,nact)), 1e-5 * np.random.normal(size=(npc,1))]



def plot_analysis(logparams,latencys, allcoords, stable_perf, exptname=None , rsz=0.025):
    f, axs = plt.subplots(7,3,figsize=(12,21))
    total_trials = len(latencys)
    gap = 25

    #latency 
    score = plot_latency(latencys, ax=axs[0,0])

    plot_pc(logparams, 0,ax=axs[0,1], title='Before Learning', goalsize=rsz)

    plot_pc(logparams, total_trials,ax=axs[0,2], title='After Learning', goalsize=rsz)


    plot_value(logparams, [gap,total_trials//4, total_trials], ax=axs[3,0], goalsize=rsz)


    plot_velocity(logparams,  [gap,total_trials//4,total_trials],ax=axs[1,0], goalsize=rsz)



    ## high d at reward
    dx = plot_density(logparams,  total_trials, ax=axs[1,1], goalsize=rsz)

    fx = plot_frequency(allcoords,  total_trials, ax=axs[1,2], gap=gap, goalsize=rsz)


    #### done till here

    plot_fx_dx(allcoords, logparams, gap,'Before Learning', ax=axs[2,0], gap=gap)

    plot_fx_dx(allcoords, logparams, total_trials,'After Learning', ax=axs[2,1], gap=gap)

    plot_fxdx_trials(allcoords, logparams, np.linspace(gap, total_trials,dtype=int, num=31), ax=axs[2,2], gap=gap)

    # change in field area
    plot_field_area(logparams, np.linspace(0, total_trials, num=51, dtype=int), ax=axs[3,1])

    # change in field location
    plot_field_center(logparams, np.linspace(0, total_trials, num=51, dtype=int), ax=axs[3,2])


    ## drift
    trials, pv_corr,rep_corr, startxcor, endxcor = get_pvcorr(logparams, stable_perf, total_trials, num=1001)

    plot_rep_sim(startxcor, stable_perf, ax=axs[4,0])

    plot_rep_sim(endxcor, total_trials, ax=axs[4,1])
    
    drift = (np.std(pv_corr))/(np.std(np.array(latencys)[np.linspace(stable_perf, total_trials-1, num=1001, dtype=int)]))

    plot_pv_rep_corr(trials, pv_corr, rep_corr,title=f"D={drift:.3f}",ax=axs[4,2])

    param_delta = get_param_changes(logparams, total_trials)
    plot_param_variance(param_delta, total_trials, stable_perf,axs=axs[5])

    plot_l1norm(param_delta[2], ax=axs[5,2].twinx(), stable_perf=stable_perf)

    # plot_policy(logparams,ax=axs[6,0])

    plot_com(logparams,[0.75,0.0],total_trials//2-1, ax=axs[6,0])

    # dlambda = np.mean(np.std(param_delta[0][stable_perf:],axis=0))
    # dalpha= np.mean(np.std(param_delta[2][stable_perf:],axis=0))

    plot_active_frac(logparams, total_trials, num=total_trials//1000, threshold=0.1,ax=axs[6,2])

    # for trial in np.linspace(0,total_trials, num=3, dtype=int):
    #     axs[6,1].hist(logparams[trial][2],alpha=0.25, label=f'T={trial+1}')

    plot_amplitude_drift(logparams, total_trials, stable_perf, ax=axs[6,1])

    f.text(0.001,0.001, exptname, fontsize=5)
    f.tight_layout()
    return f, score, drift

def plot_policy(logparams,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    im = ax.imshow(logparams[-1][3],aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Action')
    ax.set_ylabel('PF')
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels([-1,1])

def plot_active_frac(logparams, train_episodes,num=100, threshold=1.0,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    trials = np.linspace(0,train_episodes-1,num, dtype=int)
    active_frac = []
    for trial in trials:
        amp = logparams[trial][2]**2
        active_frac.append(amp)
    active_frac = np.array(active_frac)

    ax.plot(trials, np.sum(active_frac>=threshold,axis=1)/active_frac.shape[1])
    ax.set_xlabel('Trial')
    ax.set_ylabel(f'Active Fraction > {threshold}')


def get_param_changes(logparams, total_trials, stable_perf=0):

    lambdas = []
    sigmas = []
    alphas = []
    episodes = np.arange(stable_perf, total_trials)
    for e in episodes:
        lambdas.append(logparams[e][0])
        sigmas.append(logparams[e][1])
        alphas.append(logparams[e][2])
    lambdas = np.array(lambdas)
    sigmas = np.array(sigmas)
    alphas = np.array(alphas)
    return [lambdas, sigmas, alphas]

def plot_param_variance(param_change, total_trials, stable_perf,num=10,axs=None):
    if axs is None:
        f,axs = plt.subplots(nrows=1, ncols=3)
    [lambdas, sigmas, alphas] = param_change
    # Assuming `lambdas` is your T x N matrix
    variances = np.var(alphas[stable_perf:], axis=0)
    # Get indices of the top 10 variances
    top_indices = np.argsort(variances)[-num:][::-1]
    episodes = np.arange(0, total_trials)

    labels = [r'$\lambda$', r'$\sigma$',r'$\alpha$']
    for i, param in enumerate([lambdas, sigmas, alphas]):
        for n in top_indices:
            axs[i].plot(episodes[stable_perf:], param[stable_perf:,n])
        axs[i].set_xlabel('Trial')
        axs[i].set_ylabel(labels[i])

def plot_pv_rep_corr(trials, pv_corr, rep_corr,title,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    ax.plot(trials, pv_corr,label='$\phi(t)$')
    ax.plot(trials, rep_corr,label=r'$\phi(t)^\top\phi(t)$')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Correlation')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6) 

def plot_latency(latencys,ax=None, window=20):
    if ax is None:
        f,ax = plt.subplots()
    ax.plot(latencys)
    ma = moving_average(latencys, window)
    ax.plot(ma)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Latency (Steps)')
    #plt.xscale('log')
    score = evaluate_loss(latencys)
    ax.set_title(f'Latency: {np.round(ma[-1]):.0f}, Score: {score:.3f}')

    # x=np.arange(len(latencys)+1)
    # ap,bp,cp = fit_model(x, ma, func_type='exp')
    # ax.plot(exponential(x,ap,bp,cp), color='r', label=f'{ap:.3g}e(-{bp:.3g}x)+{cp:.3g}')
    # plt.legend(frameon=False)
    return score

def plot_l1norm(alpha_delta,stable_perf=0, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    ax.set_ylabel('$|\\alpha|_1$')
    l1norm = np.linalg.norm(alpha_delta,ord=1, axis=1)
    ax.plot(np.arange(len(alpha_delta))[stable_perf:], l1norm[stable_perf:], color='k',linewidth=3)

def plot_amplitude_drift(logparams, total_trials, stable_perf, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    param_delta = get_param_changes(logparams, total_trials, stable_perf)
    mean_amplitude = np.mean(param_delta[2]**2,axis=0)
    # delta_lambda = np.std(param_delta[0],axis=0)
    # delta_alpha = np.std(param_delta[2],axis=0)
    deltas = np.sum(np.std(np.array(param_delta),axis=1),axis=0)
    ax.scatter(mean_amplitude, deltas)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(mean_amplitude).reshape(-1), np.array(deltas).reshape(-1))
    regression_line = slope * np.array(mean_amplitude).reshape(-1) + intercept
    ax.plot(np.array(mean_amplitude).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    ax.legend(frameon=False)
    ax.set_xlabel('Mean Amplitude')
    ax.set_ylabel('$\sum var(\theta)$')

def plot_rep_sim(xcor,trial, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    im = ax.imshow(xcor)
    plt.colorbar(im)
    ax.set_xlabel('Location (x)')
    ax.set_ylabel('Location (x)')
    idx = np.array([0,500,1000])
    ax.set_xticks(np.arange(1001)[idx], np.linspace(-1,1,1001)[idx])
    ax.set_yticks(np.arange(1001)[idx], np.linspace(-1,1,1001)[idx])
    ax.set_title(f'T={trial}')

def plot_value(logparams, trial, goalcoord=[0.5,0.5], startcoord=[-0.75,-0.75], goalsize=0.05, envsize=1, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    num = 31
    xs = get_statespace(num)
    pcacts = predict_batch_placecell(logparams[trial], xs)
    value = pcacts @ logparams[trial][4] 
    im = ax.imshow(value.reshape(num,num), origin='lower')
    plt.colorbar(im,ax=ax)

    reward_grid = gaussian(xs, goalcoord, goalsize).reshape(num, num)
    start_grid = gaussian(xs, startcoord, goalsize).reshape(num, num)
    ax.imshow(reward_grid, cmap='OrRd', origin='lower')
    ax.imshow(start_grid, cmap='YlGn', origin='lower')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xticks([],[])
    ax.set_yticks([],[])



def plot_field_area(logparams, trials,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    areas = []
    for trial in trials:
        area = np.trapz(predict_batch_placecell(logparams[trial], np.linspace(-1,1,1001)),axis=0)
        areas.append(area)
    areas = np.array(areas)
    norm_area = areas/areas[0]

    ax.errorbar(trials, np.mean(norm_area,axis=1), np.std(norm_area,axis=1)/np.sqrt(len(logparams[0][0])), marker='o')
    ax.set_ylabel('Norm Field Area')
    ax.set_xlabel('Trial')
    return norm_area

def plot_field_center(logparams, trials,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    lambdas = []
    for trial in trials:
        lambdas.append(logparams[trial][0])
    lambdas = np.array(lambdas)
    norm_lambdas = lambdas-lambdas[0]
    ax.errorbar(trials, np.mean(norm_lambdas,axis=1), np.std(norm_lambdas,axis=1)/np.sqrt(len(logparams[0][0])), marker='o')
    ax.set_ylabel('Centered Field Center')
    ax.set_xlabel('Trial')
    return norm_lambdas



def plot_velocity(logparams, trial, goalcoord=[0.5,0.5], startcoord=[-0.7,-0.75], goalsize=0.05, envsize=1, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    num=31
    xs = get_statespace(num)

    pcacts = predict_batch_placecell(logparams[trial], xs)
    actout = pcacts @ logparams[trial][3] 
    aprob = softmax(actout)
    onehot2dirmat = np.array([
    [0,1],  # up
    [1,0],  # right
    [0,-1],  # down
    [-1,0]  # left
    ])
    vel = np.matmul(aprob, onehot2dirmat * 0.1)
    xx, yy = np.meshgrid(np.arange(num), np.arange(num))
    ax.quiver(xx.reshape(-1),yy.reshape(-1), vel[:,0], vel[:,1], color='k', scale_units='xy', zorder=2)

    reward_grid = gaussian(xs, goalcoord, goalsize).reshape(num, num)
    start_grid = gaussian(xs, startcoord, goalsize).reshape(num, num)
    ax.imshow(reward_grid, cmap='OrRd', origin='lower', zorder=2)
    ax.imshow(start_grid, cmap='YlGn', origin='lower',zorder=2)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    

def plot_pc(logparams, trial,pi=None, title='', ax=None, goalcoord=[0.5,0.5], startcoord=[-0.75,-0.75], goalsize=0.05, envsize=1, ):
    num = 31
    xs = get_statespace(num)
    pcacts = predict_batch_placecell(logparams[trial], xs)

    reward_grid = gaussian(xs, goalcoord, goalsize).reshape(num, num)
    start_grid = gaussian(xs, startcoord, goalsize).reshape(num, num)

    num_curves = pcacts.shape[1]
    yidx = xidx = int(num_curves**0.5)
    if ax is None:
        f,axs = plt.subplots(yidx, xidx, figsize=(12,12))
        pcidx = np.arange(num_curves)
        axs = axs.flatten()
    else:
        if pi is None:
            pcidx = [num_curves//2]
        else:
            pcidx = [pi]
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
    
    for i in pcidx:
        if len(pcidx)>1:
            ax = axs[i]
        ax.imshow(pcacts[:, i].reshape(num, num), origin='lower')
        ax.imshow(reward_grid, cmap='OrRd', origin='lower')
        ax.imshow(start_grid, cmap='YlGn', origin='lower')

        ax.set_xticks([],[])
        ax.set_yticks([],[])
        max_value = np.max(pcacts[:, i])
        ax.text(1.0, 0.0, f'{i+1}-{max_value:.2f}', transform=ax.transAxes,
                fontsize=6, color='yellow', ha='right')

    if ax is None:
        f.suptitle(title)
        f.tight_layout()

def plot_com(logparams,goalcoords,stable_perf, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    x = logparams[stable_perf][0]
    y = logparams[stable_perf*2][0]
    plt.plot(np.linspace(np.min(x),np.max(x),1000),np.linspace(np.min(y),np.max(y),1000), color='k')
    indexes = np.where((x >= goalcoords[0]-0.1) & (x <= goalcoords[0]+0.1))[0]
    values_in_x = x[indexes]
    values_in_y = y[indexes]
    ax.scatter(x, y)
    ax.scatter(values_in_x, values_in_y, color='g')
    ax.axvline(0.5, color='r')
    ax.axhline(0.0, color='r')
    ax.set_xlabel('Before')
    ax.set_ylabel('After')

def plot_density(logparams, trial, ax=None, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.025, envsize=1, ):
    if ax is None:
        f,ax = plt.subplots()

    num = 15
    xs = get_statespace(num)
    pcacts = predict_batch_placecell(logparams[trial], xs)
    dx = np.sum(pcacts,axis=1)
    im = ax.imshow(dx.reshape(num,num), origin='lower')
    plt.colorbar(im,ax=ax)
    reward_grid = gaussian(xs, goalcoord, goalsize).reshape(num, num)
    start_grid = gaussian(xs, startcoord, goalsize).reshape(num, num)
    ax.imshow(reward_grid, cmap='OrRd', origin='lower', zorder=2)
    ax.imshow(start_grid, cmap='YlGn', origin='lower',zorder=2)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    return dx.reshape(num,num)

def plot_frequency(allcoords, trial, gap=20, bins=15, goalcoord=[0.5,0.5], startcoord=[-0.75,-0.75], goalsize=0.05, ax=None):
    if ax is None:
        f,ax = plt.subplots()

    coord = []
    for t in range(gap):
        for c in allcoords[trial-t-1]:
            coord.append(c)
    coord = np.array(coord)

    x = np.linspace(-1,1,bins+1)
    xx,yy = np.meshgrid(x,x)
    x = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    coord = np.concatenate([coord, x],axis=0)

    hist, x_edges, y_edges = np.histogram2d(coord[:, 0], coord[:, 1], bins=[bins, bins])

    xs = x_edges[:-1] + (x_edges[1] - x_edges[0])/2 
    ys = y_edges[:-1] + (y_edges[1] - y_edges[0])/2 

    xx,yy = np.meshgrid(xs,ys)
    visits = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    freq = hist.reshape(-1)

    im = ax.imshow(freq.reshape(bins,bins), origin='lower')
    plt.colorbar(im, ax=ax)


    reward_grid = gaussian(xs, goalcoord, goalsize).reshape(bins, bins)
    start_grid = gaussian(xs, startcoord, goalsize).reshape(bins, bins)
    ax.imshow(reward_grid, cmap='OrRd', origin='lower', zorder=2)
    ax.imshow(start_grid, cmap='YlGn', origin='lower',zorder=2)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xticks([],[])
    ax.set_yticks([],[])

    return freq.reshape(bins,bins)


def reward_func(xs,goal, rsz, threshold=1e-2):
    rx =  np.exp(-0.5 * np.sum(((xs - goal) / rsz) ** 2, axis=1))
    return rx * (rx>threshold)

def gaussian(xs, center, sigma):
    values = np.exp(-0.5 * np.sum(((xs - center) / sigma) ** 2, axis=1))
    values[values < 0.01] = np.nan  # Convert values less than 0.01 to NaN
    return values

def moving_average(signal, window_size):
    # Pad the signal to handle edges properly
    padded_signal = np.pad(signal, (window_size//2, window_size//2), mode='edge')
    
    # Apply the moving average filter
    weights = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(padded_signal, weights, mode='valid')
    
    return smoothed_signal

def saveload(filename, variable, opt):
    import pickle
    if opt == 'save':
        with open(f"{filename}.pickle", "wb") as file:
            pickle.dump(variable, file)
        print('file saved')
    else:
        with open(f"{filename}.pickle", "rb") as file:
            return pickle.load(file)
    

def get_pvcorr(params, start, end, num):
    xs = np.linspace(-1,1,1001)
    startpcs = predict_batch_placecell(params[start], xs)
    startvec = startpcs.flatten()
    trials = np.linspace(start, end-1, num, dtype=int)
    startxcor = startpcs@startpcs.T

    pv_corr = []
    rep_corr = []
    for i in trials:
        endpcs = predict_batch_placecell(params[i], xs)
        endvec = endpcs.flatten()
        R = np.corrcoef(startvec, endvec)[0, 1]
        pv_corr.append(R)

        endxcor = endpcs@endpcs.T
        R_rep = np.corrcoef(startxcor.flatten(), endxcor.flatten())[0, 1]
        rep_corr.append(R_rep)
    return trials, pv_corr,rep_corr, startxcor, endxcor

def get_learning_rate(initial_lr, final_lr, total_steps):
    steps = np.arange(total_steps + 1)
    decay_rate = (final_lr / initial_lr) ** (1 / total_steps)
    learning_rates = initial_lr * (decay_rate ** steps)
    return learning_rates


def plot_gif(logparams, startcoord=[-0.75], goalcoord=[0.5], goalsize=0.025, envsize=1, gif_name='place_cells.gif', num_frames=100, duration=5):
    frames = []
    xs = np.linspace(-envsize, envsize, 1001)
    
    # Select indices for frames to use
    frames_to_use = np.linspace(0, len(logparams) - 1, num_frames, dtype=int)
    
    for p in frames_to_use:
        params = logparams[p]
        pcacts = []
        velocity = []
        
        for x in xs:
            pc = predict_placecell(params, x)
            actout = np.matmul(pc, params[3])
            aprob = softmax(2 * actout)
            if params[3].shape[1] == 3:
                vel = np.matmul(aprob, np.array([[-1], [1], [0]]))
            else:
                vel = np.matmul(aprob, np.array([[-1], [1]]))
            pcacts.append(pc)
            velocity.append(np.clip(vel,-1,1) * 0.1)
        
        pcacts = np.array(pcacts)
        velocity = np.array(velocity)

        plt.figure(figsize=(4, 4))
        for i in range(pcacts.shape[1]):
            plt.plot(xs, pcacts[:, i])
        plt.hlines(xmin=-envsize, xmax=envsize, y=0, colors='k')
        plt.axvline(startcoord[0],ymin=0, ymax= 6.01, color='g', linestyle='--', label='Start', linewidth=2)
        plt.fill_betweenx(np.linspace(0,  6.01), goalcoord[0] - goalsize, goalcoord[0] + goalsize, color='r', alpha=0.25)
        plt.ylabel('Tuning curves $\phi(x)$')
        plt.xlabel('Location (x)')
        plt.title(f'T={p}')
        plt.ylim(-0.01, 6.01)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

    # Save GIF
    imageio.mimsave(gif_name, frames, duration=duration)

# Example usage:
# plot_gif(logparams, startcoord, goalcoord, goalsize)


# Example usage:
# plot_gif(logparams, startcoord, goalcoord, goalsize)
