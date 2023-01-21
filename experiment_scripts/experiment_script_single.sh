#!/bin/bash
## Use this experiment to perform one experiment N times
## Averaged results will be stored in the results folder
cd ../main_training

rm -f results/step_times.txt

exp_type="baechi" #"single_gpu"

sch="etf" #"sct", "topo"

model_name="inception_v3"
batch_size="64"

N=4
counter=1

echo "Experiment Settings:"
echo "- Model ${model_name} with batch size ${batch_size}."
echo "- Other settings (e.g: no. of training steps in each run etc) can be changed in config.py"
echo "- The experiment will be run ${N} times"
echo "- Average results of the run will be recorded in the results folder.
--- For Baechi, a folder with model name and exp setting is created.
--- For single gpu, result is stored as single-gpu-model.txt"
echo "- Average result across all ${N} is recorded in result_details.txt. For each setting,
these are the numbers quoted in the paper"
echo ""; echo ""
echo "########################################################################################";
echo "Starting the experiments->>>>>>>"
echo "Performing the following experiment:"

if [ $exp_type ==  "baechi" ]
then
    echo "-- Baechi with m-${sch}";
    
else
    echo "-- Single GPU";
fi

echo "########################################################################################";
echo "";echo ""

if [ $exp_type ==  "baechi" ]
then
    while [ $counter -le $N ]
    do
        echo "Run no ${counter} of m-${sch}"; echo "***------*****-------***"
        python main_baechi.py --sch $sch -m $model_name -b $batch_size
        ((counter++))
        echo "Sleep for 2 sec"
        sleep 2
done
    echo All done
    echo "Saving results of m-${sch} across all ${N} runs"; echo "***------*****-------***";
    python result_summary.py -t $exp_type --sch $sch -m $model_name -b $batch_size
else
    while [ $counter -le $N ]
    do
        echo "Run no ${counter} of single gpu experiment "; echo "***------*****-------***"
        python main_single_gpu.py -m $model_name -b $batch_size
        ((counter++))
        echo "Sleep for 2 sec"
        sleep 2
done
    echo All done
    echo "Saving results of single-gpu across all ${N} runs"; echo "***------*****-------***";
    python result_summary.py -t $exp_type -m $model_name -b $batch_size
fi
