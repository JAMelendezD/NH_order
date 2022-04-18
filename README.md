# NH Generalized order parameter

Calculates the generalized order parameter using the autocorrelation function of the NH vectors (requires well-fitted trajectory) or using the iRED procedure applied to the NH vectors. 

<p align="center">
  <img width="400" src="images/nh_vectors.png">
</p>

## Autocorrelation

```
python nh_order.py protein.tpr protein.xtc 0.1 "name N and not resname PRO and not resid 1" "name H and not resname PRO and not resid 1" 10 ./results
```

<p align="center">
  <img width="800" src="images/acf.png">
</p>

## iRED

```
python nh_order.py protein.tpr protein.xtc 0.1 "name N and not resname PRO and not resid 1" "name H and not resname PRO and not resid 1" 10 ./results
```

## Ordered parameter

<p align="center">
  <img width="800" src="images/order.png">
</p>