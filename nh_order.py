import numpy as np
import MDAnalysis
import argparse
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
from scipy.optimize import differential_evolution, curve_fit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument('top', type=str, help='tpr file')
parser.add_argument('traj', type=str, help='trajectory file')
parser.add_argument('sele1', type=str, help='main selection 1')
parser.add_argument('sele2', type=str, help='main selection 2')
parser.add_argument('--mode', default=0,required=False, type=int, help='0 acf. 1 iRED')
parser.add_argument('out', type=str, help='output file')
args = parser.parse_args()

@jit(nopython=True)
def p2_angle(vec1,vec2):
    '''
    Computes the second order legendre polynomial between two vectors
    '''
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    p2 = 0.5*(3*dot_product**2-1)
    return(p2)

@jit(nopython=True)
def acf(vectors,residue,frames):
    '''
    Coputes the autocorrelation function same as GROMACS using P2 to itself
    '''
    M = int(frames/2)
    C_p = np.zeros(M+1)
    for j in range(M+1):
        p2_sum = 0
        for i in range(frames-M):
            p2_sum += p2_angle(vectors[i][residue],vectors[i+j][residue])
        C_p[j] = p2_sum/M
    return(C_p/C_p[0])

@jit(nopython=True)
def angles(vectors,num_residues,residue,frames):
    '''
    Computes the correlation to every other residue using P2 needed to generate the matrix for iRED
    '''
    angs = np.zeros(num_residues)
    for k in range(num_residues):
        if k >= residue:
            for i in range(frames):
                angs[k] +=  p2_angle(vectors[i][residue],vectors[i][k])
    return(angs/frames)

def exp_model(x,A0,A1):
    '''
    Exponential model for the fit of the autocorrelation
    '''
    return(A0+(1-A0)*np.exp(-x/A1))

def plot_mat(mat,residues):
    '''
    Plot the covariance matrix with the correct labels
    '''
    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.set_xticks(np.arange(0,len(residues),len(residues)//10))
    ax.set_yticks(np.arange(0,len(residues),len(residues)//10))
    x_labels = np.array(ax.get_xticks().tolist())
    residues = np.array(residues)
    x_labels = residues[x_labels]
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(x_labels)
    fig = plt.imshow(mat)
    plt.colorbar(fig,fraction=0.046, pad=0.02)
    plt.savefig(f'{args.out}ired_mat.png',dpi=300,bbox_inches='tight')

def ired(mat):
    '''
    Computes the main modes and calculates the order parameter as a sum of all modes but the first 5 (3 tran, 2 rot)
    '''
    size = np.shape(mat)[0]
    deltas = np.zeros((size,size))
    ls, vecs = np.linalg.eigh(mat)
    ls = np.flipud(ls)
    vecs = np.fliplr(vecs)

    # Calculate the delta S^2 as the outer product of |i><i|_jj 
    # multiplied by the corresponding eigen value lambda_i
    for m in range(size):
        tmp = np.outer(vecs[:,m],vecs[:,m])
        for i in range(size):
            deltas[i,m] = ls[m]*tmp[i,i]

    # Sum all the contributions of the deltas from i=5 (since the first 
    # largest modes correspond to global tumbling) to n then S^2=1-sum(deltas)
    #  and map it to their corresponding residue.
    S = np.zeros(size)
    for i in range(size):
        S[i] = 1-np.sum(deltas[i,5:])
    return(S)

def run():
    u = MDAnalysis.Universe(args.top,args.traj)
    len_traj = len(u.trajectory)

    dt = (u.trajectory[1].time-u.trajectory[0].time)*1e-3

    print(f'The number of frames are:\t\t\t{len_traj:8d}')
    print(f'The calculated time step is:\t\t\t{dt:8.4f} ns')

    sel1 = u.select_atoms(f'{args.sele1}')
    sel2 = u.select_atoms(f'{args.sele2}')
    
    num_atoms_sel1 = len(sel1)
    num_atoms_sel2 = len(sel2)
    
    print(f'The number of atoms in selection 1:\t\t{num_atoms_sel1:8d}')
    print(f'The number of atoms in selection 2:\t\t{num_atoms_sel2:8d}')
    
    sel1_resnums = list(sel1.atoms.resnums)
    sel2_resnums = list(sel2.atoms.resnums)
    
    print(f'The first and last resnums for selection 1:\t{sel1_resnums[0]:5d}{sel1_resnums[-1]:5d}')
    print(f'The first and last resnums for selection 2:\t{sel2_resnums[0]:5d}{sel2_resnums[-1]:5d}')

    vectors = []
    for ts in tqdm(u.trajectory,colour='green',desc='Frames'):
        nh_vector = sel1.positions - sel2.positions
        vectors.append(nh_vector)

    shape = np.shape(vectors)    
    frames,residues = shape[0],shape[1]

    if args.mode == 0:
        fig = plt.figure(figsize=(7,6))
        time = np.arange(int(frames/2)+1)*dt
        max_x = max(time)
        min_x = min(time)
        order = []
        residue_ids = []
        for residue in tqdm(range(residues),colour='green',desc='Residues'):
            C_p = acf(vectors,residue,frames)
            result = list(zip(time,C_p))
            np.savetxt(f'{args.out}{sel1[residue].resid}.dat',result,fmt=['%15.3f','%10.5f'],header='{:>13s}{:>11s}'.format('Time (ns)','C_p(t)'))
            fittedParameters, pcov = curve_fit(exp_model, time, C_p, bounds = ([-0.5,min_x],[1,max_x]))
            A0,A1 = fittedParameters
            y_fit = exp_model(time, A0,A1)
            plt.scatter(time, C_p, s=1, alpha = 0.8)
            plt.plot(time, y_fit)
            order.append(A0)
            residue_ids.append(sel1[residue].resid)
        plt.xlim(time[0],time[-1])
        plt.xlabel(r'$\tau$ (ns)')
        plt.ylabel(r'$C$($\tau$)')
        plt.savefig(f'{args.out}acf.png',dpi=300,bbox_inches='tight')
        result = list(zip(residue_ids,order))
        np.savetxt(f'{args.out}order_acf.dat',result,fmt=['%8d','%8.5f'],header='{:7s}{:>8s}'.format('Residue','S2'))

    
    elif args.mode == 1:
        mat = np.zeros((residues,residues))
        residue_ids = []
        for residue in tqdm(range(residues),colour='green',desc='Residues'):
            mat[:][residue] = angles(vectors,residues,residue,frames)
            residue_ids.append(sel1[residue].resid)
        mat = mat+mat.T
        np.fill_diagonal(mat, mat.diagonal()/2.0)
        np.savetxt(f'{args.out}ired_mat.dat',mat,fmt='%9.6f')
        plot_mat(mat,residue_ids)
        S2 = ired(mat)
        result = list(zip(residue_ids,S2))
        np.savetxt(f'{args.out}order_ired.dat',result,fmt=['%8d','%8.5f'],header='{:7s}{:>8s}'.format('Residue','S2'))

if __name__ == '__main__':
    run()
