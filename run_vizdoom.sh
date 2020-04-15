### Training ###
epochs=30
workers=4
framestack=1
l2=0.00001
data_dir=doom_recordings
experiments_dir=experiments
save_freq=1
repetitions=3

width=80
height=60

### Training ###
model_dir=$experiments_dir/vizdoom-models
mkdir -p $model_dir
actions=3
for i in $(seq 1 $repetitions)
do
    python3 train.py $data_dir vizdoom_health_gathering_supreme $model_dir/vizdoom_health_gathering_supreme_${i} --epochs $epochs --workers $workers --framestack $framestack --l2 $l2 --save-freq $save_freq --json --width $width --height $height --actions $actions
    python3 plot_history.py $model_dir/vizdoom_health_gathering_supreme_${i}-history.json --save
done

actions=6
for i in $(seq 1 $repetitions)
do
    python3 train.py $data_dir vizdoom_deathmatch $model_dir/vizdoom_deathmatch_${i} --epochs $epochs --workers $workers --framestack $framestack --l2 $l2 --save-freq $save_freq --json --width $width --height $height --actions $actions
    python3 plot_history.py $model_dir/vizdoom_deathmatch_${i}-history.json --save
done

### Evaluation ###
savedir=$experiments_dir/vizdoom-results
action=sampling
processes=4
games=50

mkdir -p $savedir

# Only evaluate last three epochs for the final performance
for epoch in 28 29 30
do
    python3 play_vizdoom.py $model_dir/vizdoom_health_gathering_supreme*${epoch}.pt --config doom_scenarios/health_gathering_supreme.cfg --processes $processes --games $games --framestack $framestack --save $savedir --width $width --height $height
    python3 play_vizdoom.py $model_dir/vizdoom_deathmatch*${epoch}.pt --config doom_scenarios/deathmatch.cfg --processes $processes --games $games --framestack $framestack --save $savedir --width $width --height $height
done

# Play random results
python3 play_vizdoom.py $models_dir/vizdoom_deathmatch_all_1 --config doom_scenarios/deathmatch.cfg --processes $processes --games 100 --framestack $framestack --save $savedir-random --width $width --height $height --no-cuda --random
python3 play_vizdoom.py $models_dir/vizdoom_health_gathering_supreme_all_1 --config doom_scenarios/health_gathering_supreme.cfg --processes $processes --games 100 --framestack $framestack --save $savedir-random --width $width --height $height --no-cuda --random
