NUM_OBJ=5
ENV=PickPlaceInterferingDistractors-v0
OUTPUT=/nfs/kun2/users/asap7772/bridge_data_exps/sim_data/minibullet/0602_multitask_interfering
amount=250
timesteps=30

num_parallel=10
counter=0

for ((i=0; i<${NUM_OBJ} ; i++)); do
    for ((j=0; j<${NUM_OBJ} ; j++)); do
        if [ $i -ne $j ]; then
            COMMAND="python scripts/scripted_collect.py -e ${ENV} -pl pickplace -n ${amount} -t ${timesteps} \
            -d ${OUTPUT}/task_${i}_${j} -a place_success_target --noise 0.1 --specific_task_id 1 --desired_task_id $i $j &"
            
            echo $COMMAND
            eval $COMMAND
    
            sleep 10
            counter=$((counter+1))
    
            if [ $counter -ge $num_parallel ]; then
                wait
                counter=0
            fi
        fi
    done
done