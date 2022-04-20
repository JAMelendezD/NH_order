import numpy as np
import MDAnalysis as mda
import argparse
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
from scipy.optimize import curve_fit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument('top', type=str, help='tpr file')
parser.add_argument('traj', type=str, help='trajectory file')
parser.add_argument('first', type=int, help='First frame starts at 0')
parser.add_argument('last', type=int, help='Last frame inclusive')
parser.add_argument('sele1', type=str, help='main selection 1')
parser.add_argument('sele2', type=str, help='main selection 2')
parser.add_argument('P', type=int, help='legendre polynomial only options 1 or 2.')
parser.add_argument('--mode', default=0,required=False, type=int, help='0 acf. 1 acf in blocks. 2 iRED. 3 computes angle against vector')
parser.add_argument('--lenacf', default=None,required=False, type=int, help='Length of the ACF maximum frames/2')
parser.add_argument('--vec', nargs='+', help='Vector to compute angle for mode 3', required=False)
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
def p1_angle(vec1,vec2):
    '''
    Computes the first order legendre polynomial between two vectors
    '''
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    p1 = np.dot(unit_vec1, unit_vec2)
    return(p1)

@jit(nopython=True)
def acf(vectors,residue,frames,length,func):
    '''
    Computes the autocorrelation function same as GROMACS using P2 to itself
    '''
    if length != None:
        if length <= int(frames/2):
            M = length
        else:
            M = int(frames/2)
    else:
        M = int(frames/2)

    C_p = np.zeros(M+1)
    for j in range(M+1):
        p_sum = 0
        for i in range(frames-M):
            p_sum += func(vectors[i][residue],vectors[i+j][residue])
        C_p[j] = p_sum/M
    return(C_p/C_p[0])

@jit(nopython=True)
def acf_blocks(vectors,residue,frames,length,func):
    '''
    Computes the autocorrelation function same as GROMACS using P2 to itself in blocks
    '''
    if length != None:
        if length <= int(frames/2):
            M = length
        else:
            M = int(frames/2)
    else:
        M = int(frames/2)

    blocks = int(frames/length)
    C_p = np.zeros(M+1)
    for j in range(M+1):
        p_sum = 0
        for i in range(blocks-1):
            p_sum += func(vectors[i*M][residue],vectors[i*M+j][residue])
        C_p[j] = p_sum/blocks
    return(C_p/C_p[0])

@jit(nopython=True)
def angles(vectors,num_residues,residue,frames,func):
    '''
    Computes the correlation to every other residue using P2 needed to generate the matrix for iRED
    '''
    angs = np.zeros(num_residues)
    for k in range(num_residues):
        if k >= residue:
            for i in range(frames):
                angs[k] +=  func(vectors[i][residue],vectors[i][k])
    return(angs/frames)

@jit(nopython=True)
def angle_vec(vectors,vec,residue,frames,func):
    '''
    Computes the angle of every vector to the z-axis
    '''
    angs = 0
    for i in range(frames):
        angs +=  func(vectors[i][residue],vec)
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
    plt.savefig(f'{args.out}mat_ired_{args.P}.png',dpi=300,bbox_inches='tight')

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
    print(f'MDA version: {mda.__version__}')
    u = mda.Universe(args.top,args.traj)
    len_traj = len(u.trajectory[args.first:args.last])

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
    for ts in tqdm(u.trajectory[args.first:args.last],colour='green',desc='Frames'):
        nh_vector = sel1.positions - sel2.positions
        vectors.append(nh_vector)

    shape = np.shape(vectors)    
    frames,residues = shape[0],shape[1]

    if args.P == 1:
        func = p1_angle
    elif args.P == 2:
        func = p2_angle

    if args.mode == 0 or args.mode == 1:
        if args.lenacf != None:
            if args.lenacf <= int(frames/2):
                time = np.arange(args.lenacf+1)*dt
            else:
                time = np.arange(int(frames/2)+1)*dt
        else:
            time = np.arange(int(frames/2)+1)*dt
        max_x = max(time)
        min_x = min(time)
        order = []
        residue_ids = []
        fig = plt.figure(figsize=(7,6))
        for residue in tqdm(range(residues),colour='green',desc='Residues'):
            if args.mode == 0:
                C_p = acf(vectors,residue,frames,args.lenacf,func)
            elif args.mode == 1:
                C_p = acf_blocks(vectors,residue,frames,args.lenacf,func)
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
        plt.savefig(f'{args.out}acf_{args.mode}_{args.lenacf}_P{args.P}.png',dpi=300,bbox_inches='tight')
        result = list(zip(residue_ids,order))
        np.savetxt(f'{args.out}order_acf_{args.mode}_{args.lenacf}_P{args.P}.dat',result,fmt=['%8d','%8.5f'],header='{:7s}{:>8s}'.format('Residue','S2'))

    elif args.mode == 2:
        mat = np.zeros((residues,residues))
        residue_ids = []
        for residue in tqdm(range(residues),colour='green',desc='Residues'):
            mat[:][residue] = angles(vectors,residues,residue,frames,func)
            residue_ids.append(sel1[residue].resid)
        mat = mat+mat.T
        np.fill_diagonal(mat, mat.diagonal()/2.0)
        np.savetxt(f'{args.out}mat_ired_{args.P}.dat',mat,fmt='%9.6f')
        plot_mat(mat,residue_ids)
        S2 = ired(mat)
        result = list(zip(residue_ids,S2))
        np.savetxt(f'{args.out}order_ired_{args.P}.dat',result,fmt=['%8d','%8.5f'],header='{:7s}{:>8s}'.format('Residue','S2'))

    elif args.mode == 3:
        if args.vec == None:
            raise argparse.ArgumentError('mode 3 require also a vector with the --vec flag')
        else:
            if len(args.vec) != 3:
                raise argparse.ArgumentError('vector must be 3 dimensional')
            else:
                with open(f'{args.out}order_axis.dat', 'a') as f:
                    order = 0
                    vec = np.array(args.vec,dtype=np.float32)
                    for residue in tqdm(range(residues),colour='green',desc='Residues'):
                        order += angle_vec(vectors,vec,residue,frames,func)
                    f.write(f'{sel1.names[0]:>8s}{order/residues:10.5f}\n')

if __name__ == '__main__':
    run()
