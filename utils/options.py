import os.path as osp
import json
import os
import sys

class Options:

	def __init__(self, path):

		self.def_opts = {\
		'stepsize': 11000,\
		'gpu':'0',\
		'max_iter':30000,\
		'base_lr':0.00004,\
		'size':300,\
		'split_id':1,\
		'diff_max':3,\
		'resume':True\
		}

		self.opts = self.set_opts(path)
		for k, v in self.def_opts.iteritems():
			if k in self.opts:
				continue
			else:
				self.opts[k] = v


	def set_opts(self, json_path):
		if json_path != '':
			return json.load(open(json_path, 'r'))
		else:
			temp = {}
			return temp


	def get_opts(self, opt_id):
		if opt_id not in self.opts:
			if opt_id in self.def_opts:
				return self.def_opts[opt_id]
			else:
				print '%s not found in opt or default opts' % opt_id
				sys.exit()
		else:
			return self.opts[opt_id]


	def get_avd_db_stem(self, split):
		out = 'split%d_diff%d_%s' % (self.get_opts('split_id'), self.get_opts('diff_max'), split)
		return out

	def add_kv(self, key, val):
		self.opts[key] = val

	def write_opt(self, path):
		with open(path, 'w') as out:
			json.dump(self.opts, out)

