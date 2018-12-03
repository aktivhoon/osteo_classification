# read out epochs with best metric
cat outs/fold_DenseNet_acc_m2/log.txt | grep Val | awk -F [:' '] '{print $2 ": " $4 "\tResult: " $7 + 1.5*$16}' | sort -k 3 -r | head
