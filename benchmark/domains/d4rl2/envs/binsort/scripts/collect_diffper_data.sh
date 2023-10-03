envs=(BinSort-v0 BinSortSingleObjBinObj-v0)
perc=(0.2 0.5 0.1 1.0)
noise=(0.1 0.02)
timesteps=40
policy=(binsortneutralstored)
trunc=(2 4)
num=1000

counter=0
num_parallel=4

for env in ${envs[@]}; do
    for per in ${perc[@]}; do
        for pi in ${policy[@]}; do
            for no in ${noise[@]}; do
                for tr in ${trunc[@]}; do
                    path=/nfs/kun2/users/asap7772/diffper_data_binsort_fixed/${pi}_binper${per}_noise${no}_trunc${tr}/

                    command="python scripts/scripted_collect.py -e ${env} -pl ${pi} \
                    -n ${num} -t ${timesteps} -d ${path} -a place_success_target \
                    --noise ${no} --save-all --p_place_correct ${per} --trunc=${tr}"

                    echo ${command}
                    ${command} &

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


