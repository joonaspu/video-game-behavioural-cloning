### Training ###
epochs=1
workers=16
framestack=2
l2=0.00001
data_dir=${ATARI_GC_DIR}
models_dir=models_pytorch_all
save_freq=1

for game in mspacman pinball qbert revenge spaceinvaders
do
    # Do 3 iterations of each training
    for i in 1 2 3
    do
        # Top 5%
        epochs=10
        percentile=95
        python3 train.py $data_dir $game $models_dir/${game}_95_${i} --epochs $epochs --workers $workers --percentile $percentile --framestack $framestack --l2 $l2 --save-freq $save_freq --merge
        python3 plot_history.py $models_dir/${game}_95_${i}-history --save

        # Top 50%
        epochs=5
        percentile=50
        python3 train.py $data_dir $game $models_dir/${game}_50_${i} --epochs $epochs --workers $workers --percentile $percentile --framestack $framestack --l2 $l2 --save-freq $save_freq --merge
        python3 plot_history.py $models_dir/${game}_50_${i}-history --save

        # All data
        epochs=5
        python3 train.py $data_dir $game $models_dir/${game}_all_${i} --epochs $epochs --workers $workers --framestack $framestack --l2 $l2 --save-freq $save_freq --merge
        python3 plot_history.py $models_dir/${game}_all_${i}-history --save

        # Top n
        for topn in 1 2 3
        do
            epochs=10
            python3 train.py $data_dir $game $models_dir/${game}_top${topn}_${i} --epochs $epochs --workers $workers --top-n ${topn} --framestack $framestack --l2 $l2 --save-freq $save_freq --merge
            python3 plot_history.py $models_dir/${game}_top${topn}_${i}-history --save
        done
    done
done

### Evaluation ###
savedir=results_pytorch_all/
action=sampling
processes=5
games=20
no_ops=30

python3 play_atari.py $models_dir/mspacman_* --env MsPacman-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
python3 play_atari.py $models_dir/qbert_* --env Qbert-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
python3 play_atari.py $models_dir/spaceinvaders_* --env SpaceInvaders-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
python3 play_atari.py $models_dir/revenge_* --env MontezumaRevenge-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge
python3 play_atari.py $models_dir/pinball_* --env VideoPinball-v0 --processes $processes --games $games --framestack $framestack --save $savedir --action $action --no-op $no_ops --merge