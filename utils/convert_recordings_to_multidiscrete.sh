# Convert bunch of game-recording trajectories
# (with buttons down) into trajectories with
# multi-discrete actions

if test -z "$1"
then
    echo "Usage: convert_recordings_to_multidiscrete.sh game_name input_dir output_dir"
    exit
fi

mkdir -p ${3}

for filepath in ${2}/*
do
  basename=$(basename $filepath)
  new_path=${3}/${basename}
  # Check for os type... On Linux we use python3,
  # but on windows  "py".
  # TODO missing Mac support (darwin)
  if [[ "$OSTYPE" == "linux-gnu" ]]; then
    python3 convert_recording_to_multidiscrete.py ${1} ${filepath} ${new_path}
  else
    py convert_recording_to_multidiscrete.py ${1} ${filepath} ${new_path}
  fi
done
