 python scripts/scripted_collect.py -e PutAerointoBowl-v0 -pl pickplace -n 250 -t 30 -d /mount/harddrive/trainingdata/minibullet/multitask_pickplacedata_noise0.1 -a place_success_target --noise 0.1  &
 sleep 1
 python scripts/scripted_collect.py -e PutT_cupintoBowl-v0 -pl pickplace -n 250 -t 30 -d /mount/harddrive/trainingdata/minibullet/multitask_pickplacedata_noise0.1 -a place_success_target --noise 0.1 & 
 sleep 1
 python scripts/scripted_collect.py -e Putcolunnade_topintoBowl-v0 -pl pickplace -n 250 -t 30 -d /mount/harddrive/trainingdata/minibullet/multitask_pickplacedata_noise0.1 -a place_success_target --noise 0.1 &
 sleep 1
 python scripts/scripted_collect.py -e Putbeehive_funnelintoBowl-v0 -pl pickplace -n 250 -t 30 -d /mount/harddrive/trainingdata/minibullet/multitask_pickplacedata_noise0.1 -a place_success_target --noise 0.1 &
 sleep 1
 python scripts/scripted_collect.py -e Putcrooked_lid_trash_canlintoBowl-v0 -pl pickplace -n 250 -t 30 -d /mount/harddrive/trainingdata/minibullet/multitask_pickplacedata_noise0.1 -a place_success_target --noise 0.1 &
 sleep 1
 python scripts/scripted_collect.py -e Putcbbongo_drum_bowlintoBowl-v0 -pl pickplace -n 250 -t 30 -d /mount/harddrive/trainingdata/minibullet/multitask_pickplacedata_noise0.1 -a place_success_target --noise 0.1 &
