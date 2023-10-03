NUM_TARGET=6
NUM_DISTRACTOR=5
ENV=PickPlaceDisjointDistractors-v0
OUTPUT=/nfs/kun2/users/asap7772/bridge_data_exps/sim_data/minibullet/0602_multitask_disjoint
amount=250
timesteps=30

num_parallel=5
counter=0


for ((i=0; i<${NUM_TARGET} ; i++)); do
    for ((j=0; j<${NUM_TARGET} ; j++)); do
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
    done
done