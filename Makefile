# Project path
proj=$(shell pwd)

# Intepreter and cli
py=/data/amax/b510/yl/.conda/envs/rs/bin/python
cli=$(proj)/cf/run/run_cli.py

# Configs
model=medcn
# Dataset (criteo, ml, avazu)
dataset=huawei
## evaluate config
config=/data/amax/b510/yl/repo/33/22/rs/cf/tune/dcn_me/avazu/0.yaml
weight=/data/amax/b510/yl/repo/33/22/rs/cf/result/dcn_me/20220713153831/weights.002-0.38175.hdf5
evcmd=$(py) $(cli) -m $(model) -c $(config) -t test -p $(weight) -d $(dataset)
## train config
t_cfg=/data/amax/b510/yl/repo/33/22/rs/cf/tune/medcn/huawei/0.yaml
t_weight=''
trcmd=$(py) $(cli) -m $(model) -c $(t_cfg) -p $(t_weight) -d $(dataset)
## pred config
p_cfg=/data/amax/b510/yl/repo/33/22/rs/cf/tune/medcn/huawei/0.yaml
p_weight=/data/amax/b510/yl/repo/33/22/rs/cf/result/medcn/20220727151629/weights.006-0.07807.hdf5
prcmd=$(py) $(cli) -m $(model) -c $(p_cfg) -t predict -p $(p_weight) -d $(dataset)
## Other
lastlog=$(shell ls -f $(proj)/log/$(model)-$(dataset)*.log | sort -r | head -n 1)
## profile dir
pdir=
port=6006
## Tune order
name=medcn
tune_cli=$(proj)/cf/utils/tune.py

evaluate:
	@cd $(proj)
	@$(evcmd)

train:
	@cd $(proj)
	@$(trcmd)

clear:
	@clear
	@rm $(proj)/log/*.log

peek:
	@clear
	@cat $(lastlog)

watch:
	@clear
	@tail -f $(lastlog)

profile:
	@tensorboard --logdir=$(pdir) --bind_all --port 6006

tune:
	@$(py) $(tune_cli) -m $(name)

show:
	@echo $(trcmd)

predict:
	@cd $(proj)
	@$(prcmd)
