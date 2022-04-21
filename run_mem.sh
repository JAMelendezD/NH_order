

carbons=(C22 C23 C24 C25 C26 C27 C28 C29 C210 C211 C212 C213 C214 C215 C216 C217)
hydrogens=(H2R H3R H4R H5R H6R H7R H8R H91 H101 H11R H12R H13R H14R H15R H16R H17R)

for i in ${!carbons[@]};
do
    python nh_order.py ./data_mem/mem.tpr ./data_mem/mem_mol.xtc 5000 10000 "name ${carbons[i]}" "name ${hydrogens[i]}" 2 ./data_mem/ --mode 3 --vec 0 0 -1
done

carbons=(C32 C33 C34 C35 C36 C37 C38 C39 C310 C311 C312 C313 C314 C315)
hydrogens=(H2X H3X H4X H5X H6X H7X H8X H9X H10X H11X H12X H13X H14X H15X)

for i in ${!carbons[@]};
do
    python nh_order.py ./data_mem/mem.tpr ./data_mem/mem_mol.xtc 5000 10000 "name ${carbons[i]}" "name ${hydrogens[i]}" 2 ./data_mem/ --mode 3 --vec 0 0 -1
done
