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

### project hpc

python -m spinup.run plot -s 10 projectMSAC_ep1000_k1_ant-v2/ projectMSAC_ep1000_k5_ant-v2/ projectMSAC_ep1000_sv_k1_ant-v2/ projectMSAC_ep1000_sv_k5_ant-v2/ --value Performance --legend SAC MSAC5 SAC-Single MSAC-Single5 --ylabel Performance_Ant-v2

python -m spinup.run plot -s 10 projectMSAC_ep1000_k1_ant-v2/ projectMSAC_ep1000_k5_ant-v2/ projectMSAC_ep1000_sv_k1_ant-v2/ projectMSAC_ep1000_sv_k5_ant-v2/ --value AverageQ1Vals --legend SAC MSAC5 SAC-Single MSAC-Single5 --ylabel AverageQ1Vals_Ant-v2

python -m spinup.run plot -s 10 projectMSAC_ep1000_k1_ant-v2/ projectMSAC_ep1000_k5_ant-v2/ projectMSAC_ep1000_sv_k1_ant-v2/ projectMSAC_ep1000_sv_k5_ant-v2/ --value AverageQBias --legend SAC MSAC5 SAC-Single MSAC-Single5 --ylabel AverageQBias_Ant-v2

python -m spinup.run plot -s 10 projectMSAC_ep1000_k1_hopper-v2/ projectMSAC_ep1000_k5_hopper-v2/ projectMSAC_ep1000_sv_k1_hopper-v2/ projectMSAC_ep1000_sv_k5_hopper-v2/ --value Performance --legend SAC MSAC5 SAC-Single MSAC-Single5 --ylabel Performance_Hopper-v2

python -m spinup.run plot -s 10 projectMSAC_ep1000_k1_hopper-v2/ projectMSAC_ep1000_k5_hopper-v2/ projectMSAC_ep1000_sv_k1_hopper-v2/ projectMSAC_ep1000_sv_k5_hopper-v2/ --value AverageQ1Vals --legend SAC MSAC5 SAC-Single MSAC-Single5 --ylabel AverageQ1Vals_Hopper-v2

python -m spinup.run plot -s 10 projectMSAC_ep1000_k1_hopper-v2/ projectMSAC_ep1000_k5_hopper-v2/ projectMSAC_ep1000_sv_k1_hopper-v2/ projectMSAC_ep1000_sv_k5_hopper-v2/ --value AverageQBias --legend SAC MSAC5 SAC-Single MSAC-Single5 --ylabel AverageQBias_Hopper-v2

---

python -m spinup.run plot -s 10 projectMSAC_ep1000_sv_k1_ant-v2/ projectMSAC-single_ep1000_sv_k2_ant-v2/ projectMSAC_ep1000_sv_k5_ant-v2/ projectMSAC-single_ep1000_sv_k8_ant-v2/ --value Performance --legend SAC-Single MSAC-Single2 MSAC-Single5 MSAC-Single8 --ylabel Performance_Ant-v2

python -m spinup.run plot -s 10 projectMSAC_ep1000_sv_k1_ant-v2/ projectMSAC-single_ep1000_sv_k2_ant-v2/ projectMSAC_ep1000_sv_k5_ant-v2/ projectMSAC-single_ep1000_sv_k8_ant-v2/ --value AverageQ1Vals --legend SAC-Single MSAC-Single2 MSAC-Single5 MSAC-Single8 --ylabel AverageQ1Vals_Ant-v2

python -m spinup.run plot -s 10 projectMSAC_ep1000_sv_k1_ant-v2/ projectMSAC-single_ep1000_sv_k2_ant-v2/ projectMSAC_ep1000_sv_k5_ant-v2/ projectMSAC-single_ep1000_sv_k8_ant-v2/ --value AverageQBias --legend SAC-Single MSAC-Single2 MSAC-Single5 MSAC-Single8 --ylabel AverageQBias_Ant-v2

python -m spinup.run plot -s 10 projectMSAC_ep1000_sv_k1_hopper-v2/ projectMSAC-single_ep1000_sv_k2_hopper-v2/ projectMSAC_ep1000_sv_k5_hopper-v2/ projectMSAC-single_ep1000_sv_k8_hopper-v2/ --value Performance --legend SAC-Single MSAC-Single2 MSAC-Single5 MSAC-Single8 --ylabel Performance_Hopper-v2

python -m spinup.run plot -s 10 projectMSAC_ep1000_sv_k1_hopper-v2/ projectMSAC-single_ep1000_sv_k2_hopper-v2/ projectMSAC_ep1000_sv_k5_hopper-v2/ projectMSAC-single_ep1000_sv_k8_hopper-v2/ --value AverageQ1Vals --legend SAC-Single MSAC-Single2 MSAC-Single5 MSAC-Single8 --ylabel AverageQ1Vals_Hopper-v2

python -m spinup.run plot -s 10 projectMSAC_ep1000_sv_k1_hopper-v2/ projectMSAC-single_ep1000_sv_k2_hopper-v2/ projectMSAC_ep1000_sv_k5_hopper-v2/ projectMSAC-single_ep1000_sv_k8_hopper-v2/ --value AverageQBias --legend SAC-Single MSAC-Single2 MSAC-Single5 MSAC-Single8 --ylabel AverageQBias_Hopper-v2

---

python -m spinup.run plot -s 10 projectMSAC_ep1000_k1_ant-v2/ projectMSAC_ep1000_k5_ant-v2/ projectMSAC_ep1000_sv_k1_ant-v2/ projectMSAC_ep1000_sv_k5_ant-v2/ projectMSAC-single_ep1000_sv_k2_ant-v2/ projectMSAC_ep1000_sv_k5_ant-v2/ projectMSAC-single_ep1000_sv_k8_ant-v2/ --value Time --legend SAC MSAC5 SAC-Single MSAC-Single5 MSAC-Single2 MSAC-Single5 MSAC-Single8 --ylabel Time_Ant-v2

python -m spinup.run plot -s 10 projectMSAC_ep1000_k1_hopper-v2/ projectMSAC_ep1000_k5_hopper-v2/ projectMSAC_ep1000_sv_k1_hopper-v2/ projectMSAC_ep1000_sv_k5_hopper-v2/ projectMSAC-single_ep1000_sv_k2_hopper-v2/ projectMSAC_ep1000_sv_k5_hopper-v2/ projectMSAC-single_ep1000_sv_k8_hopper-v2/ --value Time --legend SAC MSAC5 SAC-Single MSAC-Single5 MSAC-Single2 MSAC-Single5 MSAC-Single8 --ylabel Time_Hopper-v2

---

python -m spinup.run plot -s 10 projectMSAC_ep1000_k1_ant-v2/ projectMSAC_ep1000_k5_ant-v2/ projectMSAC_ep1000_sv_k1_ant-v2/ projectMSAC_ep1000_sv_k5_ant-v2/ projectMSAC-single_ep1000_sv_k2_ant-v2/ projectMSAC_ep1000_sv_k5_ant-v2/ projectMSAC-single_ep1000_sv_k8_ant-v2/ --value NumDatapoints --legend SAC MSAC5 SAC-Single MSAC-Single5 MSAC-Single2 MSAC-Single5 MSAC-Single8 --ylabel Datapoints_Ant-v2

python -m spinup.run plot -s 10 projectMSAC_ep1000_k1_hopper-v2/ projectMSAC_ep1000_k5_hopper-v2/ projectMSAC_ep1000_sv_k1_hopper-v2/ projectMSAC_ep1000_sv_k5_hopper-v2/ projectMSAC-single_ep1000_sv_k2_hopper-v2/ projectMSAC_ep1000_sv_k5_hopper-v2/ projectMSAC-single_ep1000_sv_k8_hopper-v2/ --value NumDatapoints --legend SAC MSAC5 SAC-Single MSAC-Single5 MSAC-Single2 MSAC-Single5 MSAC-Single8 --ylabel Datapoints_Hopper-v2
