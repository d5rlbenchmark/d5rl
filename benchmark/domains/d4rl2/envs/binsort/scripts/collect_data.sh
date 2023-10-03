# python scripts/scripted_collect.py -e PutBottleintoBowl-v0 -pl pickplace -n 1000 -t 25 -d /mount/harddrive/trainingdata/minibullet/pickplacedata_noise0.02 -a place_success_target --noise 0.02
# python scripts/scripted_collect.py -e PutBallintoBowl-v0 -pl pickplace -n 250 -t 30 -d /mount/harddrive/trainingdata/minibullet/pickplacedata_noise0.1 -a place_success_target --noise 0.1
python scripts/scripted_collect.py -e PutBallintoBowlDiverse-v0 -pl pickplace -n 1000 -t 25 -d ./data/fulldiverse_pickplacedata_noise0.1 -a place_success_target --noise 0.1 &
python scripts/scripted_collect.py -e PutBallintoBowlDiverseHalf-v0 -pl pickplace -n 1000 -t 25 -d ./data/halfdiverse_pickplacedata_noise0.1 -a place_success_target --noise 0.1 &
python scripts/scripted_collect.py -e PutBallintoBowlDiverseThird-v0 -pl pickplace -n 1000 -t 25 -d ./data/thirddiverse_pickplacedata_noise0.1 -a place_success_target --noise 0.1 &

python scripts/scripted_collect.py -e PutBallintoBowlDiverse-v0 -pl pickplace -n 1000 -t 25 -d ./data/failed_fulldiverse_pickplacedata_noise0.1 -a place_success_target --noise 0.1 --save_failonly
python scripts/scripted_collect.py -e PutBallintoBowlDiverseHalf-v0 -pl pickplace -n 1000 -t 25 -d ./data/failed_halfdiverse_pickplacedata_noise0.1 -a place_success_target --noise 0.1  --save_failonly
python scripts/scripted_collect.py -e PutBallintoBowlDiverseThird-v0 -pl pickplace -n 1000 -t 25 -d ./data/failed_thirddiverse_pickplacedata_noise0.1 -a place_success_target --noise 0.1  --save_failonly
