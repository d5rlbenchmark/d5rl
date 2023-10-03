envs=(BinSort-v0 BinSortSingleObjBinObj-v0)
perc=(0.05 0.1 0.2 0.5 0.8 1.0)
noise=(0.1 0.02)
timesteps=40
policy=(binsortneutral)
trunc=(2 4)

counter=0
num_parallel=4

for no in ${noise[@]}; do
    for tr in ${trunc[@]}; do
        for per in ${perc[@]}; do
            for env in ${envs[@]}; do
                for pi in ${policy[@]}; do
                    if [ $pi = "binsort" ]; then
                        num=1000
                    elif [ $pi = "stitching" ]; then
                        num=100
                    elif [ $pi = "binsortneutral" ]; then
                        num=1000
                    fi
                    path=/nfs/kun2/users/asap7772/binsort1k/${pi}_binper${per}_noise${no}_trunc${tr}/

                    command="python scripts/scripted_collect.py -e ${env} -pl ${pi} \
                    -n ${num} -t ${timesteps} -d ${path} -a place_success_target \
                    --noise ${no} --save-all --p_place_correct ${per} --trunc=${tr}"

                    echo ${command}
                    ${command} &

                    sleep 2.0
                    
                    counter=$((counter+1))
                    if [ $counter -ge $num_parallel ]; then
                        wait
                        counter=0
                    fi
                done
            done
        done
    done
done


