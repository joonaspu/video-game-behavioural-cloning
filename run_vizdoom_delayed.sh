### Training ###
epochs=30
workers=4
framestack=1
l2=0.00001
data_dir=doom_recordings
models_dir=experiments/vizdoom-delayed-models
save_freq=1
delays="-100 -10 -5 -2 2 5 10 100"

mkdir -p $models_dir

width=80
height=60

actions=3

for delay in $delays
do
    for i in 1 2 3
    do
        python3 train.py $data_dir vizdoom_health_gathering_supreme $models_dir/vizdoom_health_gathering_supreme_all_delay_${delay}_${i} --epochs $epochs --workers $workers --framestack $framestack --l2 $l2 --save-freq $save_freq --json --width $width --height $height --actions $actions --action-delay ${delay}
        python3 plot_history.py $models_dir/vizdoom_health_gathering_supreme_all_delay_${delay}_${i}-history.json --save
    done
done

actions=6
for delay in $delays
do
    for i in 1 2 3
    do
        python3 train.py $data_dir vizdoom_deathmatch $models_dir/vizdoom_deathmatch_all_delay_${delay}_${i} --epochs $epochs --workers $workers --framestack $framestack --l2 $l2 --save-freq $save_freq --json --width $width --height $height --actions $actions --action-delay ${delay}
        python3 plot_history.py $models_dir/vizdoom_deathmatch_all_delay_${delay}_${i}-history.json --save
    done
done


### Evaluation ###
savedir=experiments/vizdoom-delayed-results
action=sampling
processes=4
games=50

mkdir -p $savedir

# Only evaluate last three models
for epoch in 28 29 30
do
    python3 play_vizdoom.py $models_dir/vizdoom_health_gathering_supreme*${epoch}.pt --config doom_scenarios/health_gathering_supreme.cfg --processes $processes --games $games --framestack $framestack --save $savedir --width $width --height $height
    python3 play_vizdoom.py $models_dir/vizdoom_deathmatch*${epoch}.pt --config doom_scenarios/deathmatch.cfg --processes $processes --games $games --framestack $framestack --save $savedir --width $width --height $height
done

### Plotting ###
for delay in $delays
do
    python3 plot_atari_eval.py ${savedir} vizdoom_deathmatch_all_delay_${delay} --save --repeats 3
    python3 plot_atari_eval.py ${savedir} vizdoom_health_gathering_supreme_all_delay_${delay} --save --repeats 3
done
