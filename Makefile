# Project path
proj=/data/amax/b510/yl/repo/33/22/rs

# Intepreter and cli
py=/data/amax/b510/yl/.conda/envs/rs/bin/python
cli=/data/amax/b510/yl/repo/33/22/rs/cf/run/run_cli.py

# Configs
## evaluate config
model=dcnv2
config=/data/amax/b510/yl/repo/33/22/rs/cf/result/dcnv2/20220528145839/config.yaml
weight=/data/amax/b510/yl/repo/33/22/rs/cf/result/dcnv2/20220527180953/weights.002-0.46903.hdf5
## train config
t_model=dcnv2
t_cfg=/data/amax/b510/yl/repo/33/22/rs/cf/result/dcnv2/20220527180953/config.yaml
t_weight=''
## Other
lastlog=$(shell ls -f $(proj)/log/* | sort -r | head -n 1)

evaluate:
	cd $(proj)
	$(py) $(cli) -m $(model) -c $(config) -t test -p $(weight)

train:
	cd $(proj)
	$(py) $(cli) -m $(t_model) -c $(t_cfg) -p $(t_weight)

clear:
	@rm $(proj)/log/*.log

peek:
	@cat $(lastlog)

watch:
	@tail -f $(lastlog)