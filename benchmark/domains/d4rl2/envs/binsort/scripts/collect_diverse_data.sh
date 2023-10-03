python scripts/scripted_collect.py -e PutBallintoBowlCamera-v0 -pl pickplace -n 5000 -t 25 -d ./data/camera_large_data -a place_success_target --noise 0.1 --gui --save-all
python scripts/scripted_collect.py -e PutBallintoBowlDiverse-v0 -pl pickplace -n 5000 -t 25 -d ./data/diverse_bowl_pos_large_data -a place_success_target --noise 0.1 --gui --save-all


# python scripts/scripted_collect.py -e PutBallintoBowlCamera-v0 -pl pickplace -n 1000 -t 25 -d ./data/redmultview_pickplacedata_noise0.1 -a place_success_target --noise 0.1 & 
# python scripts/scripted_collect.py -e PutBallintoBowlCamera-v0 -pl pickplace -n 1000 -t 25 -d ./data/failed_redmultview_pickplacedata_noise0.02 -a place_success_target --noise 0.02 --save_failonly &
# python scripts/scripted_collect.py -e PutBallintoBowlCamera-v0 -pl pickplace -n 1000 -t 25 -d ./data/failed_redmultview_pickplacedata_noise0.1 -a place_success_target --noise 0.1 --save_failonly &