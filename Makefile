# Project path
proj=$(shell pwd)

# Intepreter and cli
py=/data/amax/b510/yl/.conda/envs/rs/bin/python
cli=$(proj)/cf/run/run_cli.py

# Configs
model=medcn
# Dataset (criteo, ml, avazu)
dataset=criteo
## evaluate config
config=/data/amax/b510/yl/repo/33/22/rs/cf/tune/dcnv2/20220607163757/0.yaml
weight=/data/amax/b510/yl/repo/33/22/rs/cf/result/dcnv2/20220607202118/weights.0.8115-0.81210-5700.hdf5
evcmd=$(py) $(cli) -m $(model) -c $(config) -t test -p $(weight) -d $(dataset)
## train config
t_cfg=/data/amax/b510/yl/repo/33/22/rs/cf/tune/medcn/20220614225934/0.yaml
t_weight=/data/amax/b510/yl/repo/33/22/rs/cf/result/medcn/20220617144121/weights.008-0.44764.hdf5
trcmd=$(py) $(cli) -m $(model) -c $(t_cfg) -p $(t_weight) -d $(dataset)
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