envs=(BinSortSingleObjBinObj-v0)
perc=(0.2 0.5 0.1 1.0)
noise=(0.1 0.02)
timesteps=40
policy=(binsortneutral)
trunc=(2 4)
num=1000

for env in ${envs[@]}; do
    for per in ${perc[@]}; do
        for pi in ${policy[@]}; do
            for no in ${noise[@]}; do
                for tr in ${trunc[@]}; do
                    path=/nfs/kun2/users/asap7772/noterm_data_binsort/${pi}_binper${per}_noise${no}_trunc${tr}/

                    command="python scripts/scripted_collect.py -e ${env} -pl ${pi} \
                    -n ${num} -t ${timesteps} -d ${path} -a place_success_target \
                    --noise ${no} --save-all --p_place_correct ${per} --trunc=${tr} --ignore_done"

                    echo ${command}
                    ${command} &
                done
            done
            wait
        done
    done
done


