#/bin/bash

while true
do
{
	takeNum=`ps -aux | grep "python main.py train" | wc -l`
	if [ $takeNum -ge 2 ];then
		echo "takeNum: $takeNum"
		continue
	fi
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Data/ZZY/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Data/ZZY/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/Data/ZZY/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Data/ZZY/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
	conda activate cod
	cd /Data/ZZY/P_Edge_N
	python main.py train
}
done
