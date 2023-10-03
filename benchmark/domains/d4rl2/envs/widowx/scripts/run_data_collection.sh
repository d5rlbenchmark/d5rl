# Data collection scripts


# Grasping data for ambient objects

# Noisy data

# success : 10%
python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvBall-v0 -pl grasp_any -a any_object_place_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/grasping_data_final/ --num-tasks 1 --save-all --noise=0.1 -p 10

# sucess: 30-40%
python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl grasp_any -a any_object_place_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/grasping_data_final/ --num-tasks 1 --save-all --noise=0.1 -p 10

# success: 20-30%
python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvBallGatorade-v0 -pl grasp_any -a any_object_place_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/grasping_data_final/ --num-tasks 1 --save-all --noise=0.1 -p 10

# success: 40-50%
python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl grasp_any -a any_object_place_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/grasping_data_final/ --num-tasks 1 --save-all --noise=0.1 -p 10


# Expert-ish data
# success: 0 - 10%
python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvBall-v0 -pl grasp_any -a any_object_place_success -n 100 -t 30 -d /nfs/kun2/users/aviralkumar/grasping_data_final/ --num-tasks 1 --save-all --noise=0.02 -p 10

# success: 40 - 50%
python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl grasp_any -a any_object_place_success -n 100 -t 30 -d /nfs/kun2/users/aviralkumar/grasping_data_final/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvBallGatorade-v0 -pl grasp_any -a any_object_place_success -n 100 -t 30 -d /nfs/kun2/users/aviralkumar/grasping_data_final/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl grasp_any -a any_object_place_success -n 100 -t 30 -d /nfs/kun2/users/aviralkumar/grasping_data_final/ --num-tasks 1 --save-all --noise=0.02 -p 10


# Adversarial data
python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvBall-v0 -pl grasp_any -a any_object_place_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/grasping_data_final/ --num-tasks 1 --save-all --noise=0.05 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl grasp_any -a any_object_place_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/grasping_data_final/ --num-tasks 1 --save-all --noise=0.05 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvBallGatorade-v0 -pl grasp_any -a any_object_place_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/grasping_data_final/ --num-tasks 1 --save-all --noise=0.05 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl grasp_any -a any_object_place_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/grasping_data_final/ --num-tasks 1 --save-all --noise=0.05 -p 10 --suboptimal

#########################################################################################

############## Closing Drawers ##############

################ Main bottom #################


python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl second_top_close -a second_drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl second_top_close -a second_drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl second_top_close -a second_drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl second_top_close -a second_drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl second_top_close -a second_drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl second_top_close -a second_drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal


python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl second_top_close -a second_drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl second_top_close -a second_drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl second_top_close -a second_drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl second_top_close -a second_drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl second_top_close -a second_drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl second_top_close -a second_drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal



################## Main Top Close ###################
python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl main_top_close -a drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl main_top_close -a drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl main_top_close -a drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl main_top_close -a drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl main_top_close -a drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl main_top_close -a drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal


python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl main_top_close -a drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl main_top_close -a drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl main_top_close -a drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl main_top_close -a drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl main_top_close -a drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl main_top_close -a drawer_top_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal


################## Main Drawer ###########################
python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl main_drawer_close -a drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl main_drawer_close -a drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl main_drawer_close -a drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl main_drawer_close -a drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl main_drawer_close -a drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl main_drawer_close -a drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal


python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl main_drawer_close -a drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl main_drawer_close -a drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl main_drawer_close -a drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl main_drawer_close -a drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl main_drawer_close -a drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl main_drawer_close -a drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal


################# Second drawer close ###########################
python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl second_drawer_close -a second_drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl second_drawer_close -a second_drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl second_drawer_close -a second_drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl second_drawer_close -a second_drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl second_drawer_close -a second_drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl second_drawer_close -a second_drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal


python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl second_drawer_close -a second_drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl second_drawer_close -a second_drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl second_drawer_close -a second_drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl second_drawer_close -a second_drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl second_drawer_close -a second_drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl second_drawer_close -a second_drawer_closed_success -n 50 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal




################################################################################
######################     Drawer Open      ####################################
################################################################################

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl second_drawer_open -a second_drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl second_drawer_open -a second_drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl second_drawer_open -a second_drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl second_drawer_open -a second_drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl second_drawer_open -a second_drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl second_drawer_open -a second_drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal


python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl second_drawer_open -a second_drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl second_drawer_open -a second_drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl second_drawer_open -a second_drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl second_drawer_open -a second_drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl second_drawer_open -a second_drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl second_drawer_open -a second_drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal



##### Main drawer open ##############

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl main_drawer_open -a drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl main_drawer_open -a drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl main_drawer_open -a drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl main_drawer_open -a drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyShed-v0 -pl main_drawer_open -a drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnv-v0 -pl main_drawer_open -a drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal


python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl main_drawer_open -a drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl main_drawer_open -a drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.2 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl main_drawer_open -a drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl main_drawer_open -a drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.02 -p 10

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyGatorade-v0 -pl main_drawer_open -a drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal

python scripts/scripted_collect_parallel.py -e Widow250MultiDrawerMultiObjectEnvWithOnlyBall-v0 -pl main_drawer_open -a drawer_opened_success -n 200 -t 30 -d /nfs/kun2/users/aviralkumar/d4rl2_final_drawer_open_data/ --num-tasks 1 --save-all --noise=0.1 -p 10 --suboptimal