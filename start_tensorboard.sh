set -x
TRAIN_D="$1/train-d"
TRAIN_G="$1/train-g"
VALID_D="$1/valid-d"
PORT=$2

tensorboard --port $PORT --logdir train-d:"$TRAIN_D",valid-d:"$VALID_D",train-g:"$TRAIN_G"
