# Project path
proj=/data/amax/b510/yl/repo/33/22/rs

# Intepreter and cli
py=/data/amax/b510/yl/.conda/envs/rs/bin/python
cli=/data/amax/b510/yl/repo/33/22/rs/cf/run/run_cli.py

# Configs
## evaluate config
model=autoint
config=/data/amax/b510/yl/repo/33/22/rs/cf/tune/autoint/20220531154531/0.yaml
weight=/data/amax/b510/yl/repo/33/22/rs/cf/result/autoint/20220531154925/weights.001-0.46823.hdf5
## train config
t_model=autoint
t_cfg=/data/amax/b510/yl/repo/33/22/rs/cf/tune/autoint/20220531154531/0.yaml
t_weight=''
## Other
lastlog=$(shell ls -f $(proj)/log/$(t_model)* | sort -r | head -n 1)
evaluate:
	@cd $(proj)
	$(py) $(cli) -m $(model) -c $(config) -t test -p $(weight)

train:
	@cd $(proj)
	$(py) $(cli) -m $(t_model) -c $(t_cfg) -p $(t_weight)

clear:
	@clear
	rm $(proj)/log/*.log

peek:
	@clear
	@cat $(lastlog)

watch:
	@clear
	@tail -f $(lastlog)