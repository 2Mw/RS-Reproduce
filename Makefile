# Project path
proj=$(shell pwd)

# Intepreter and cli
py=/data/amax/b510/yl/.conda/envs/rs/bin/python
cli=$(proj)/cf/run/run_cli.py

# Configs
model=dcnv2
## evaluate config
config=/data/amax/b510/yl/repo/33/22/rs/cf/result/dcnv2/20220527180953-/config.yaml
weight=/data/amax/b510/yl/repo/33/22/rs/cf/result/dcnv2/20220527180953-/weights.004-0.46825.hdf5
evcmd=$(py) $(cli) -m $(model) -c $(config) -t test -p $(weight)
## train config
t_cfg=/data/amax/b510/yl/repo/33/22/rs/cf/result/dcnv2/20220527180953-/config.yaml
t_weight=/data/amax/b510/yl/repo/33/22/rs/cf/result/dcnv2/20220527180953-/weights.004-0.46825.hdf5
trcmd=$(py) $(cli) -m $(model) -c $(t_cfg) -p $(t_weight)
## Other
lastlog=$(shell ls -f $(proj)/log/$(model)*.log | sort -r | head -n 1)
## profile dir
pdir=
port=6006
## Tune order
name=
tune_cli=$(proj)/cf/utils/tune.py

evaluate:
	@cd $(proj)
	$(evcmd)

train:
	@cd $(proj)
	$(trcmd)

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