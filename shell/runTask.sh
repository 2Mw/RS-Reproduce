#!/bin/bash
# How to use
# This is a file used to execute training task. if gpu is busy it will block until gpu is available.
# Use command like this:
# 	runTask.sh.sh taskfile.txt

# Parameters
RUN_THRESHOLD=10240		# The minimum memory of GPU which allows programs to run.
SLEEP_GAP=1				# The interval of checking size of gpu memory when memory is insuffient.
# ENV INFO

# export getAvailableGPUs=3

# Run path,your project root path.
RUN_PATH='/data/amax/b510/yl/repo/33/22/rs/'

if [ $# -eq 0 ]; then
	echo 'task file is null.'
	exit 0
fi

taskfile="$(readlink -f ${1})"

echo "Run file is:$taskfile"

arr=()

function getGPUinfo(){
	str="$(nvidia-smi -q | grep 'FB Memory Usage' -A 2)"
	arr=("$(echo $str | grep -E '[0-9]+' -o)")
	arr=(${arr})
}

gpu=""
function getAvailableGPUs() {
	# get which GPU is available
	getGPUinfo
	gpu=""
	len=${#arr}
	for i in $(seq 0 2 $((len-2)))
	do
		# Caculate available memory, use it if space > RUN_THRESHOLD.
		left="$(( ${arr[i]} - ${arr[i+1]} ))"
		if [ $left -ge $RUN_THRESHOLD ]; then
			if [ ${#gpu} -eq 0 ]; then
				gpu="$gpu,$((i/2))"
			fi
		fi
	done

	if [ ${#gpu} -ne 0 ]; then
		gpu=${gpu:1}
	fi
}

cd $RUN_PATH

getGPUinfo

if [ ! -f $taskfile ]; then
	echo "$taskfile is not exists."
	exit 0
fi


# read tasks
cat $taskfile | while read line
do
	gpu=''
	while [ ${#gpu} -eq 0 ]
	do
		getAvailableGPUs
		sleep 1
		#echo "The $gpu"
	done
	echo "available gpus: $gpu"
	# set VARIABLE `CUDA_VISIBLE_DEVICES`
	export CUDA_VISIBLE_DEVICES=$gpu
	$line
	export CUDA_VISIBLE_DEVICES=""
	sleep 1
done

echo 'ALL task over.'
