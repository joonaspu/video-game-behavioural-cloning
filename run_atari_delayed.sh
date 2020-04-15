### Training ###
epochs=5
workers=16
framestack=2
l2=0.00001
data_dir=${ATARI_HEAD_DIR}
models_dir=models_action_delay
save_freq=1

# Atari-HEAD
for game in ms_pacman montezuma_revenge space_invaders
do
    for action_delay in -100 -10 -5 -2 0 2 5 10 100
    do
        # Do 3 iterations of each training
        for i in 1 2 3
        do
            # All data
            python3 train.py $data_dir/$game game $models_dir/${game}_${action_delay}_${i} --epochs $epochs --workers $workers --framestack $framestack --l2 $l2 --save-freq $save_freq --merge --atari-head --action-delay $action_delay
            python3 plot_history.py $models_dir/${game}_${action_delay}_${i}-history.json --save
        done
    done
done

#Atari GC
data_dir=${ATARI_GC_DIR}
for game in mspacman pinball qbert revenge spaceinvaders
do
    for action_delay in -100 -10 -5 -2 0 2 5 10 100
    do
        # Do 3 iterations of each training
        for i in 1 2 3
        do
            # Top 5%
            epochs=5
            percentile=95
            python3 train.py $data_dir $game $models_dir/${game}_${action_delay}_${i} --epochs $epochs --workers $workers --percentile $percentile --framestack $framestack --l2 $l2 --save-freq $save_freq --merge --action-delay $action_delay
            python3 plot_history.py $models_dir/${game}_${action_delay}_${i}-history --save
        done
    done
done

### Evaluation ###
savedir=results_action_delay/
action=sampling
processes=5
games=20
no_ops=30

# Atari-HEAD
python3 play_atari.py $models_dir/ms_pacman_* --env MsPacman-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
python3 play_atari.py $models_dir/space_invaders_* --env SpaceInvaders-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
python3 play_atari.py $models_dir/montezuma_revenge_* --env MontezumaRevenge-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge

# Atari GC
python3 play_atari.py $models_dir/mspacman_* --env MsPacman-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
python3 play_atari.py $models_dir/qbert_* --env Qbert-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
python3 play_atari.py $models_dir/spaceinvaders_* --env SpaceInvaders-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
python3 play_atari.py $models_dir/revenge_* --env MontezumaRevenge-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
python3 play_atari.py $models_dir/pinball_* --env VideoPinball-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge