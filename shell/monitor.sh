#!/bin/bash
arr=()

function getGPUinfo(){
	str="$(nvidia-smi -q | grep 'FB Memory Usage' -A 2)"
	arr=("$(echo $str | grep -E '[0-9]+' -o)")
}



function monitor() {
	echo +-----------------------------------------------------------------------------+
	nvidia-smi | grep 250W
	echo +-----------------------------------------------------------------------------+
	nvidia-smi | grep -i 'process name' -A 5
	
	arr=$(nvidia-smi | grep -i 'mib' | grep -v '250W' | grep -E '([0-9]{2,}) ' -o)
	arr=($arr)
	
	if [ ${#arr} -gt 0 ]; then
		echo -e "              \n\n=========  Who  ========\n\n"
		for i in $( seq 0 ${#arr} )
		do
			if [ ${#arr[i]} -gt 0 ]; then
				cmd="ps -eo '%p %U %a' | grep -E ${arr[i]} | grep -v 'grep'"
				bash -c "$cmd"
			fi
		done
	fi

	# monitor memory

	echo -e "              \n\n=========  Memory  ========\n\n"

	free -h

	# getGPUinfo

	users="$(w | grep -E "^\w+" -o | sort -u)"

	users=${users//\r\n/ }

	users=(${users//USER/ })

	echo -e "              \n\n=========  Users  ========\n\n"

	echo "Users(${#users[@]}): ${users[*]}"
	
}

monitor
