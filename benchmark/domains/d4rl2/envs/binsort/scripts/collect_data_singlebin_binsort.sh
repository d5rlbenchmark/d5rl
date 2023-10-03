NUM_OBJ=10
ENV=BinSortSingleBin-v0
OUTPUT=/nfs/kun2/users/asap7772/bridge_data_exps/sim_data/minibullet/0611_multitask_binsort_singleobj
amount=500
timesteps=80
policy=binsortneutralmult

num_parallel=5
counter=0

for ((i=0; i<${NUM_OBJ} ; i++)); do
    for ((j=0; j<${NUM_OBJ} ; j++)); do
        if [ $i -lt $j ]; then
            COMMAND="python scripts/scripted_collect.py -e ${ENV} -pl ${policy} -n ${amount} -t ${timesteps} \
            -d ${OUTPUT}/task_${i}_${j} -a all_placed --noise 0.1 --specific_task_id 1 --desired_task_id $i $j &"
            
            echo $COMMAND
            eval $COMMAND
    
            sleep 1
            counter=$((counter+1))
    
            if [ $counter -ge $num_parallel ]; then
                wait
                counter=0
            fi
        fi
    done
done