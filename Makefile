# Project path
proj=$(shell pwd)

# Intepreter and cli
py=/data/amax/b510/yl/.conda/envs/rs/bin/python
cli=$(proj)/cf/run/run_cli.py

# Configs
model=autoint
# Dataset (criteo, ml, avazu)
dataset=criteo
## evaluate config
config=/data/amax/b510/yl/repo/33/22/rs/cf/tune/autoint/20220606172211/0.yaml
weight=/data/amax/b510/yl/repo/33/22/rs/cf/result/autoint/20220607113622/weights.0.812-0.81234-6500.hdf5
evcmd=$(py) $(cli) -m $(model) -c $(config) -t test -p $(weight) -d $(dataset)
## train config
t_cfg=/data/amax/b510/yl/repo/33/22/rs/cf/tune/autoint/20220606172211/0.yaml
t_weight=/data/amax/b510/yl/repo/33/22/rs/cf/result/autoint/20220607113622/weights.001-0.45292.hdf5
trcmd=$(py) $(cli) -m $(model) -c $(t_cfg) -p $(t_weight) -d $(dataset)
## Other
lastlog=$(shell ls -f $(proj)/log/$(model)-$(dataset)*.log | sort -r | head -n 1)
## profile dir
pdir=
port=6006
## Tune order
name=autoint
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