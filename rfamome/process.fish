# parallel -j 20 /home/ubuntu/repos/LinAliFold-CentroidLinAliFold/bin/LinAliFold -i  {} \> {}.lina ::: (ls *.fasta)
parallel -j 20 /home/ubuntu/.myconda/miniconda3/envs/biofilm100/bin/R-scape --nofigures --gapthresh 1.0 --outmsa --outdir ./test/ {} ::: (ls *stk)
