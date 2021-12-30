# the following automate the main commands for our experiments.
n_maps = 4096
n_display = 24

cmd = '# MAIN ROBOT\n\n'

cmd += 'python mainrobot.py  --n_maps %s --PROJECT_NAME robot_gif\n\n'%(str(n_display))

for x in ['eval','train']:
    cmd+='python mainrobot.py --mode data --map_data_name robot%s.%s.data --n_maps %s\n'%(str(n_maps),str(x),str(n_maps))
cmd += '\n'

for label in ['A','B','C','D']:
    cmd+='python mainrobot.py --mode run_through --n_expt 4 --PROJECT_NAME PROJECT_%s --custom_tile_val 1 --custom_target 2.0  --train_data_name robot%s.train.data --map_data_name robot%s.eval.data\n'%(str(label),str(n_maps),str(n_maps))
    cmd+='python mainrobot.py --mode aggregate_result --PROJECT_NAME PROJECT_%s --n_expt 4\n'%(str(label))
cmd += '\n'

for label in ['A','B','C','D']:
    cmd+= 'python mainrobot.py --mode run_through --n_expt 4 --PROJECT_NAME COMPARE_%s --train_data_name robot%s.train.data --map_data_name robot%s.eval.data\n'%(str(label),str(n_maps),str(n_maps))
    cmd+= 'python mainrobot.py --mode aggregate_result --PROJECT_NAME COMPARE_%s --n_expt 4\n'%(str(label))
cmd+= '\n'

for label in ['A','B','C','D']:
    cmd+='python mainrobot.py  --mode robot_srd --n_maps %s --PROJECT_NAME PROJECT_%s\PROJECT_%s001.srd\n'%(str(n_display),str(label),str(label))
cmd+='\n'

for label in ['A','B','C','D']:
    cmd+='python mainrobot.py  --mode robot_srd --n_maps %s --PROJECT_NAME COMPARE_%s\COMPARE_%s001.srd\n'%(str(n_display),str(label),str(label))
cmd+='\n'

cmd += '\n\n# MAIN ROBOT WITH LAVA\n\n'
cmd+='python mainrobot.py  --n_maps %s --PROJECT_NAME robot_gif.lava --lava_fraction 0.1\n'%(str(n_display))
cmd+='python mainrobot.py  --n_maps %s --PROJECT_NAME robot_gif.lava.non --lava_fraction 0.1 --unknown_avoidance 0\n'%(str(n_display))
cmd+='\n'

cmd+='python mainrobot.py --mode data --map_data_name lava%s.eval.data --n_maps %s --lava_fraction 0.1\n'%(str(n_maps),str(n_maps))
cmd+='python mainrobot.py --mode data --map_data_name lava%s.train.data --n_maps %s --lava_fraction 0.1\n'%(str(n_maps),str(n_maps))
cmd+='\n'

for label in ['A','B','C','D']:
    cmd+='python mainrobot.py --mode run_through_lava --n_expt 4 --PROJECT_NAME WITH_LAVA_%s --train_data_name lava%s.train.data --map_data_name lava%s.eval.data\n'%(str(label),str(n_maps),str(n_maps))
    cmd+='python mainrobot.py --mode aggregate_result --PROJECT_NAME WITH_LAVA_%s --n_expt 4 --include_lava 1\n'%(str(label))
cmd+='\n'

for label in ['A','B','C','D']:
    cmd+='python mainrobot.py  --mode robot_srd --n_maps %s --PROJECT_NAME WITH_LAVA_%s\WITH_LAVA_%s001.srd --lava_fraction 0.1\n'%(str(n_display),str(label),str(label))
cmd+='\n'

for label in ['A','B','C','D']:
    cmd+='python mainrobot.py --mode run_through_lava --n_expt 4 --PROJECT_NAME WITH_LAVA_NOAV_%s --unknown_avoidance 0 --train_data_name lava%s.train.data --map_data_name lava%s.eval.data\n'%(str(label),str(n_maps),str(n_maps))
    cmd+='python mainrobot.py --mode aggregate_result --PROJECT_NAME WITH_LAVA_NOAV_%s --n_expt 4 --include_lava 1\n'%(str(label),)
cmd+='\n'

for label in ['A','B','C','D']:
    cmd+='python mainrobot.py  --mode robot_srd --n_maps %s --PROJECT_NAME WITH_LAVA_NOAV_%s\WITH_LAVA_NOAV_%s001.srd --lava_fraction 0.1 --unknown_avoidance 0\n'%(str(n_display),str(label),str(label))
cmd+='\n'


txt = open('commands.txt','w')
txt.write(cmd)
txt.close()