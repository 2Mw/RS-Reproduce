# Project path
proj=$(shell pwd)

# Intepreter and cli
py=/data/amax/b510/yl/.conda/envs/rs/bin/python
cli=$(proj)/cf/run/run_cli.py

# Select GPU
gpu=0
gpucmd=CUDA_VISIBLE_DEVICES=$(gpu)
# Configs
model=addpp
# Dataset (criteo, ml, avazu)
dataset=criteo
## evaluate config
config=/data/amax/b510/yl/repo/33/22/rs/cf/tune/mime/20221126190906/0.yaml
weight=/data/amax/b510/yl/repo/33/22/rs/cf/result/mime/20230112143408/weights.008.hdf5
evcmd=$(py) $(cli) -m $(model) -c $(config) -t test -p $(weight) -d $(dataset)
## train config
t_cfg=/data/amax/b510/yl/repo/33/22/rs/cf/tune/addpp/20240313210616/0.yaml
t_weight=''
trcmd=$(py) $(cli) -m $(model) -c $(t_cfg) -p $(t_weight) -d $(dataset)
## pred config
p_cfg=/data/amax/b510/yl/repo/33/22/rs/cf/tune/addpp/20240313210616/0.yaml
p_weight=/data/amax/b510/yl/repo/33/22/rs/cf/result/addpp/20240314175553/model.tf
prcmd=$(py) $(cli) -m $(model) -c $(p_cfg) -t predict -p $(p_weight) -d $(dataset)
## Other
lastlog=$(shell ls -f $(proj)/log/$(model)-$(dataset)*.log | sort -r | head -n 1)
## profile dir
pdir=
port=6006
## Tune order
name=addpp
tune_cli=$(proj)/cf/utils/tune.py

evaluate:
	@cd $(proj)
	$(gpucmd) $(evcmd)

train:
	@cd $(proj)
	$(gpucmd) $(trcmd)

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
	@$(gpucmd) $(prcmd)
