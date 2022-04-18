import numpy as np
import MDAnalysis
import argparse
from numba import jit
from tqdm import tqdm
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument('top', type=str, help='tpr file')
parser.add_argument('traj', type=str, help='trajectory file')
parser.add_argument('sele1', type=str, help='main selection 1')
parser.add_argument('sele2', type=str, help='main selection 2')
parser.add_argument('dt', type=float, help='Time interval between frames')
parser.add_argument('out', type=str, help='output file')
args = parser.parse_args()

@jit(nopython=True)
def p2_angle(vec1,vec2):
	unit_vec1 = vec1 / np.linalg.norm(vec1)
	unit_vec2 = vec2 / np.linalg.norm(vec2)
	dot_product = np.dot(unit_vec1, unit_vec2)
	p2 = 0.5*(3*dot_product**2-1)
	return(p2)

@jit(nopython=True)
def acf(vecs,residue,frames):
    M =  int(frames/2)
    C_p = np.zeros(M+1)
    for j in range(M+1):
        p2_sum = 0
        for i in range(frames-M):
            p2_sum += p2_angle(vecs[i][residue],vecs[i+j][residue])
        C_p[j] = p2_sum/M
    return(C_p/C_p[0])

def run():
    u = MDAnalysis.Universe(args.top,args.traj)
    len_traj = len(u.trajectory)

    print(f'The number of frames are:\t\t\t{len_traj:8d}')

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
    M =  int(frames/2)
    time = np.arange(frames)*args.dt
    for residue in tqdm(range(residues),colour='green',desc='Residues'):
        C_p = acf(vectors,residue,frames)
        result = list(zip(time,C_p))
        np.savetxt(f'{args.out}{sel1[residue].resid}.dat',result,fmt=['%15.3f','%10.5f'],header='{:>13s}{:>11s}'.format('Time (ps)','C_p(t)'))

if __name__ == '__main__':
    run()
