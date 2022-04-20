# NH Generalized Order Parameter

Calculates the generalized order parameter using the autocorrelation function of the NH vectors (requires well-fitted trajectory) or using the iRED procedure applied to the NH vectors. 

<p align="center">
  <img width="400" src="images/nh_vectors.png">
</p>

## Autocorrelation

<p align="center">
  <img width="300" src="images/no_block.png">
</p>

```
python nh_order.py protein.tpr protein_nj_fit.xtc 5000 10000 "name N and not resname PRO and not resid 1" "name H and not resname PRO and not resid 1" 2 ./data/ --mode 0 --lenacf 100
```

<p align="center">
  <img width="400" src="images/acf_0_100_2.png">
  <img width="400" src="images/acf_0_1000_2.png">
</p>

<p align="center">
  <img width="300" src="images/block.png">
</p>


```
python nh_order.py protein.tpr protein_nj_fit.xtc 5000 10000 "name N and not resname PRO and not resid 1" "name H and not resname PRO and not resid 1" 2 ./data/ --mode 1 --lenacf 100
```

<p align="center">
  <img width="400" src="images/acf_1_100_2.png">
  <img width="400" src="images/acf_1_1000_2.png">
</p>

## iRED

```
python nh_order.py protein.tpr protein_nj_fit.xtc 5000 10000 "name N and not resname PRO and not resid 1" "name H and not resname PRO and not resid 1" 2 ./data/ --mode 2
```

<p align="center">
  <img width="400" src="images/mat_ired_1.png">
  <img width="400" src="images/mat_ired_2.png">
</p>

## Order parameter

<p align="center">
  <img width="800" src="images/NH_order_models.png">
  <img width="800" src="images/NH_order_all.png">
  <img width="800" src="images/NH_order_block.png">
</p>

## Order parameter to any axis useful for membranes

```
python nh_order.py mem.tpr mem_mol.xtc 5000 10000 "name C3" "name H3" 2 ./data/ --mode 3 --vec 0 0 1
```

<p align="center">
  <img width="400" src="images/order.png">
</p>