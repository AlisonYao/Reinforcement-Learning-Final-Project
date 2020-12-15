# Reinforcement-Learning-Final-Project

CSCI-SHU 375 Reinforcement Learning | Fall2020 | Final Project

## environment setup

https://docs.google.com/document/d/1DvSsVHmf2KVlRmHJ9uwAikbcP4nykbNGGKisaPkKAX4/edit?usp=sharing
https://drive.google.com/drive/folders/1BFHZBoMicYDM4pWs9ohJ1yKYJcAqIvt4?usp=sharing

## hpc

### run & check job & cancel job

sbatch XXX.sh
squeue -u yy2564
scancel _job_id_

### plotting

command line: (drl) Alisons-MacBook-Pro:hw1 zhuang$ python -m spinup.run plot -s 10 SAC_ep500_rs1000_hopper-v2/ SAC_ep500_rs10000_hopper-v2/ SAC_ep500_rs100000_hopper-v2/ SAC_ep500_rs1000000_hopper-v2/ --value **Performance**

command line: (drl) Alisons-MacBook-Pro:hw1 zhuang$ python -m spinup.run plot -s 10 SAC_ep500_deter_hopper-v2/ SAC_ep500_rs1000000_hopper-v2/ --value Performance --legend deterministic stochastic

command line: (drl) Alisons-MacBook-Pro:hw1 zhuang$ python -m spinup.run plot -s 10 SAC_ep500_p0-5_hopper-v2/ SAC_ep500_p0-95_hopper-v2/ SAC_ep500_rs1000000_hopper-v2/ --value Performance --legend polyak0.5 polyak0.95 default

command line: (drl) Alisons-MacBook-Pro:hw1 zhuang$ python -m spinup.run plot -s 10 SAC_ep500_p0-5_hopper-v2/ SAC_ep500_p0-95_hopper-v2/ SAC_ep500_rs1000000_hopper-v2/ --value **AverageQ1Vals** --legend polyak0.5 polyak0.95 default

## project instructions

https://docs.google.com/document/d/16dfPlTdcFnjsxMW02LhVGd6Ommpnekb84wiUCNHwl_8/edit?usp=sharing
