### Training ###
epochs=10
workers=16
framestack=2
l2=0.00001
data_dir=${ATARI_HEAD_DIR}
models_dir=models_atari_head
save_freq=1

for game in ms_pacman montezuma_revenge space_invaders
do
    # Do 3 iterations of each training
    for i in 1 2 3
    do
        # All data
        python3 train.py $data_dir/$game game $models_dir/${game}_all_${i} --epochs $epochs --workers $workers --framestack $framestack --l2 $l2 --save-freq $save_freq --merge --atari-head
        python3 plot_history.py $models_dir/${game}_all_${i}-history.json --save
    done
done

### Evaluation ###
savedir=results_atari_head/
action=sampling
processes=5
games=20
no_ops=30

python3 play_atari.py $models_dir/ms_pacman_* --env MsPacman-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
python3 play_atari.py $models_dir/space_invaders_* --env SpaceInvaders-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
python3 play_atari.py $models_dir/montezuma_revenge_* --env MontezumaRevenge-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
