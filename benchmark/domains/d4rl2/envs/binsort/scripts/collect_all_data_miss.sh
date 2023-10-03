envs=(PutBallintoBowlDiverse)
noise=(0.1)
p_place_correct=(0.4 0.3 0.5 0.2 0.6)

num_parallel=4
counter=0
for env in ${envs[@]}; do
    for n in ${noise[@]}; do
        for p in ${p_place_correct[@]}; do
            python scripts/scripted_collect.py -e ${env}-v0 -pl pickplacemiss -n 1000 -t 25 \
            -d /nfs/kun2/users/asap7772/miss_data_diverse/placewith${p} -a place_success_target --noise ${n} --save-all --p_place_correct ${p} &

            counter=$((counter+1))
            if [ $((counter%num_parallel)) -eq 0 ]; then
                wait
            fi
        done
    done
done
