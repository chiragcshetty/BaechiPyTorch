#!/bin/bash
## Experiments: ("single_gpu" "baechi") X ("etf" "sct" "topo") X (list of perturb values)
##---------------------------------------------------
cd ../main_training

rm -f results/step_times.txt

perturb_list=("0.2") # perturbation factor
exp_type_list=("single_gpu" "baechi")
#exp_type_list=("baechi")

sch_list=("etf" "sct" "topo") # algorithms to be used
#sch_list=("sct")

model_name="inception_v3"
batch_size="64"

N=12 # No of run per combination of exp_type_list and sch_list

for exp_type in "${exp_type_list[@]}"
do
    echo $exp_type; echo "******-------------*****"
    if [ $exp_type ==  "baechi" ]
    then
        for p_factor in "${perturb_list[@]}"
        do
            for sch in "${sch_list[@]}";
            do
                counter=1
                while [ $counter -le $N ]
                do
                    echo $counter
                    python main_baechi.py --sch $sch --perturb-factor $p_factor -m $model_name -b $batch_size
                    ((counter++))
                    echo "Sleep for 2 sec"
                    sleep 2
                done
                echo All Done
                python result_summary.py -t $exp_type --sch $sch --perturb-factor $p_factor -m $model_name -b $batch_size
            done
        done
    else
        counter=1
        while [ $counter -le $N ]
        do
            echo $counter
            python main_single_gpu.py -m $model_name -b $batch_size
            ((counter++))
            echo "Sleep for 2 sec"
            sleep 2
        done
        echo All Done
        python result_summary.py -t $exp_type -m $model_name -b $batch_size
    fi
done
