#!/bin/bash
## Using this script, following experiments can be performed N times each:
## Experiment type:    "single_gpu", "baechi"
## Scheme with baechi: "etf", "sct", "topoo"
## Averaged results are stored in the results folder
cd ../main_training

rm -f results/step_times.txt
#------------------------------------------------------------------------
exp_type_list=("single_gpu" "baechi")
#exp_type_list=("baechi")

sch_list=("etf" "sct" "topo")
#sch_list=("etf")

N=10
model_name="inception_v3"
batch_size="64"
#-----------------------------------------------------------------------

echo "Experiment Settings:"
echo "- Model ${model_name} with batch size ${batch_size}."
echo "- Other settings (e.g: no. of training steps in each run etc) can be changed in config.py"
echo "- Each experiment will be run ${N} times"
echo "- Average result of each run will be recorded in the results folder.
--- For Baechi, a folder with model name and exp setting is created.
--- For single gpu, result is stored as single-gpu-model.txt"
echo "- Average result across all ${N} is recorded in result_details.txt. For each setting,
these are the numbers quoted in the paper"
echo ""; echo ""
echo "########################################################################################";
echo "Starting the experiments->>>>>>>"
echo "Performing the following experiments:"
for exp_type in "${exp_type_list[@]}"
do
    if [ $exp_type ==  "baechi" ]
    then
        for i in "${sch_list[@]}";
            do echo "-- Baechi with m-${i}";
            done
    else
        echo "-- Single GPU";
    fi
done
echo "########################################################################################";
echo "";echo ""

#----------------------------------------------------------------------


for exp_type in "${exp_type_list[@]}"
do
    echo "Starting Experiment: ${exp_type}"; echo "******------***********-------*****"
    if [ $exp_type ==  "baechi" ]
    then
        for sch in "${sch_list[@]}";
        do
            counter=1
            echo "Scheme: m-${sch}"; echo "***------*****-------***"
            while [ $counter -le $N ]
            do
                echo "Run no ${counter} of m-${sch}"; echo "***------*****-------***"
                python main_baechi.py --sch $sch -m $model_name -b $batch_size
                ((counter++))
                echo "Sleep for 2 sec";
                echo "########################################################################################";
                echo "";echo ""
                sleep 2
            done
            echo All Done
            echo "Saving results of m-${sch} across all ${N} runs"; echo "***------*****-------***";
            python result_summary.py -t $exp_type --sch $sch -m $model_name -b $batch_size
            echo "########################################################################################";
            echo "";echo ""
        done
    else
        counter=1
        while [ $counter -le $N ]
        do
            echo "Run no ${counter} of single gpu experiment "; echo "***------*****-------***"
            python main_single_gpu.py -m $model_name -b $batch_size
            ((counter++))
            echo "Sleep for 2 sec";echo "";echo ""
            sleep 2
        done
        echo All Done
        echo "Saving results of single-gpu across all ${N} runs"; echo "***------*****-------***";
        python result_summary.py -t $exp_type -m $model_name -b $batch_size
        echo "########################################################################################";
        echo "";echo ""
    fi
done