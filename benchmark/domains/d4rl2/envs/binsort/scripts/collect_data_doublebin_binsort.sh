NUM_OBJ=2
ENV=BinSort-v0
OUTPUT=/nfs/kun2/users/asap7772/bridge_data_exps/sim_data/minibullet/0811_addednoise_2obj
# OUTPUT=/home/asap7772/bridge_data_exps/sim_data/minibullet/06013_multitask_binsortneutral
timesteps=80
# policies=(binsortneutral)
policies=(binsortneutralmultstored)
accept=all_placed
# accept=place_success_target
num_parallel=1
counter=0
dry_run=false

amounts=(2000 5000 10000)

for ((i=0; i<${NUM_OBJ} ; i++)); do
    for ((j=0; j<${NUM_OBJ} ; j++)); do
        for policy in "${policies[@]}"; do
            for amount in "${amounts[@]}"; do
                if [ $i -lt $j ]; then
                    COMMAND="python scripts/scripted_collect.py -e ${ENV} -pl ${policy} -n ${amount} -t ${timesteps} \
                    -d ${OUTPUT}_${amount}_${policy}/task_${i}_${j} -a ${accept}  --noise 0.1 --specific_task_id 1 --desired_task_id $i $j --save-all &"
                    
                    echo $COMMAND

                    if [ $dry_run = false ]; then
                        if [ $counter -lt $num_parallel ]; then
                            eval $COMMAND
                            counter=$((counter+1))
                            sleep 1
                        else
                            wait
                            counter=0
                        fi
                    fi
                fi
            done
        done
    done
done


# python scripts/scripted_collect.py -e BinSort-v0 -pl binsortmult -n 250 -t 80 -d /tmp/test -a place_success_target --noise 0.1 --specific_task_id 1 --desired_task_id 0 1 --gui